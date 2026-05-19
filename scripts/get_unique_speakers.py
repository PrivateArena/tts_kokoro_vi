#!/usr/bin/env python3
import os
import sys
import json
import time
import fsspec
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, as_completed

PROGRESS_FILE = "data/extraction_progress.json"
OUTPUT_TXT = "data/unique_speakers.txt"
OUTPUT_JSON = "data/unique_speakers.json"

def get_parquet_urls():
    from huggingface_hub import HfApi
    print("Fetching file list from Hugging Face...", flush=True)
    api = HfApi()
    files = api.list_repo_files(repo_id="thivux/phoaudiobook", repo_type="dataset")
    parquet_files = sorted([f for f in files if f.endswith(".parquet") and "train-" in f])
    return parquet_files

def load_progress(parquet_files):
    os.makedirs("data", exist_ok=True)
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                progress = json.load(f)
                global_speakers = set(progress.get("unique_speakers", []))
                completed_files = set(progress.get("completed_files", []))
                print(f"Resuming progress: {len(completed_files)}/{len(parquet_files)} files processed. Unique speakers so far: {len(global_speakers)}", flush=True)
                return global_speakers, completed_files
        except Exception as e:
            print(f"Error loading progress file: {e}. Starting fresh.", flush=True)
            
    return set(), set()

def save_progress(global_speakers, completed_files):
    progress = {
        "unique_speakers": sorted(list(global_speakers)),
        "completed_files": list(completed_files)
    }
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)
        
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for spk in sorted(list(global_speakers)):
            f.write(f"{spk}\n")
            
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(sorted(list(global_speakers)), f, ensure_ascii=False, indent=2)

def extract_speakers_from_metadata(file_path):
    url = f"hf://datasets/thivux/phoaudiobook/{file_path}"
    fs = fsspec.filesystem("hf")
    retries = 5
    backoff = 2.0
    
    for attempt in range(retries):
        try:
            with fs.open(url, "rb") as f:
                metadata = pq.read_metadata(f)
                file_speakers = set()
                # Find column index for 'speaker'
                speaker_col_idx = -1
                for idx, name in enumerate(metadata.schema.names):
                    if name == "speaker":
                        speaker_col_idx = idx
                        break
                
                if speaker_col_idx == -1:
                    return set(), None
                
                for rg_idx in range(metadata.num_row_groups):
                    col_meta = metadata.row_group(rg_idx).column(speaker_col_idx)
                    if col_meta.is_stats_set and col_meta.statistics:
                        file_speakers.add(col_meta.statistics.min)
                        file_speakers.add(col_meta.statistics.max)
                
                return file_speakers, None
        except Exception as e:
            if attempt < retries - 1:
                sleep_time = backoff * (1.5 ** attempt)
                time.sleep(sleep_time)
                # Re-initialize filesystem
                fs = fsspec.filesystem("hf")
            else:
                return None, f"Failed {file_path} - Error: {e}"

def main():
    start_time = time.time()
    try:
        parquet_files = get_parquet_urls()
    except Exception as e:
        print(f"Error fetching file list: {e}", file=sys.stderr)
        sys.exit(1)
        
    total_files = len(parquet_files)
    print(f"Found {total_files} train parquet files.", flush=True)
    
    global_speakers, completed_files = load_progress(parquet_files)
    
    files_to_process = [f for f in parquet_files if f not in completed_files]
    if not files_to_process:
        print("All files have already been processed! Unique speakers list is complete.", flush=True)
        print(f"Total Unique Speakers found: {len(global_speakers)}", flush=True)
        return

    # Use moderate concurrency (6 workers) since we are only downloading kilobytes of metadata per file
    num_threads = 6
    print(f"Starting metadata-based extraction on {len(files_to_process)} remaining files with {num_threads} threads...", flush=True)
    
    completed_in_session = 0
    errors = []
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(extract_speakers_from_metadata, f): f for f in files_to_process}
        
        for future in as_completed(futures):
            f_name = futures[future]
            file_speakers, err = future.result()
            completed_in_session += 1
            
            if err:
                errors.append(err)
                print(f"\n[ERROR] {err}", flush=True)
            elif file_speakers:
                global_speakers.update(file_speakers)
                completed_files.add(f_name)
                # Save progress incrementally after every file
                save_progress(global_speakers, completed_files)
                
            total_done = len(completed_files)
            if total_done % 10 == 0 or total_done == total_files:
                pct = (total_done / total_files) * 100
                elapsed = time.time() - start_time
                speed = completed_in_session / elapsed if elapsed > 0 else 0
                eta = (len(files_to_process) - completed_in_session) / speed if speed > 0 else 0
                print(f"Progress: {total_done}/{total_files} ({pct:.1f}%) | "
                      f"Unique Speakers: {len(global_speakers)} | "
                      f"Speed: {speed:.2f} files/s | "
                      f"ETA: {eta:.0f}s", flush=True)
                      
    print("\nExtraction completed in {:.1f}s!".format(time.time() - start_time), flush=True)
    
    if errors:
        print(f"Encountered {len(errors)} errors during processing. Rerun to retry.", file=sys.stderr)
            
    print(f"Results saved to {OUTPUT_TXT} and {OUTPUT_JSON}", flush=True)
    print(f"Total Unique Speakers found: {len(global_speakers)}", flush=True)

if __name__ == "__main__":
    main()
