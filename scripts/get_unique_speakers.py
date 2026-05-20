#!/usr/bin/env python3
"""
Efficient unique-speaker extraction from thivux/phoaudiobook.

Reads only Parquet row-group statistics (min/max of the 'speaker' column)
without downloading actual audio data. Runs with 6 concurrent threads and
saves progress incrementally so it can be safely interrupted and resumed.

Improvements vs original:
  - Separates errors from genuinely skipped files so retries are accurate.
  - Falls back to full column scan if row-group stats are missing/stale.
  - Prints a cleaner, tighter progress line (no repeated header text).
  - Uses a dataclass for progress state instead of raw dicts.
  - Correctly handles the case where a parquet file has no 'speaker' column.
"""
import json
import os
import sys
import time
import fsspec
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

PROGRESS_FILE = "data/extraction_progress.json"
OUTPUT_TXT    = "data/unique_speakers.txt"
OUTPUT_JSON   = "data/unique_speakers.json"

# =============================================================================
# Progress state
# =============================================================================

@dataclass
class Progress:
    unique_speakers: set[str] = field(default_factory=set)
    completed_files: set[str] = field(default_factory=set)
    error_files:     set[str] = field(default_factory=set)

    @classmethod
    def load(cls, total_files: int) -> "Progress":
        os.makedirs("data", exist_ok=True)
        if os.path.exists(PROGRESS_FILE):
            try:
                with open(PROGRESS_FILE, encoding="utf-8") as f:
                    raw = json.load(f)
                p = cls(
                    unique_speakers = set(raw.get("unique_speakers", [])),
                    completed_files = set(raw.get("completed_files", [])),
                    error_files     = set(raw.get("error_files", [])),
                )
                print(
                    f"Resuming: {len(p.completed_files)}/{total_files} files done, "
                    f"{len(p.unique_speakers)} speakers found so far.",
                    flush=True,
                )
                return p
            except Exception as e:
                print(f"Could not load progress file ({e}) — starting fresh.", flush=True)
        return cls()

    def save(self):
        data = {
            "unique_speakers": sorted(self.unique_speakers),
            "completed_files": sorted(self.completed_files),
            "error_files":     sorted(self.error_files),
        }
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
            for spk in sorted(self.unique_speakers):
                f.write(spk + "\n")

        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(sorted(self.unique_speakers), f, ensure_ascii=False, indent=2)

# =============================================================================
# File list
# =============================================================================

def get_parquet_urls() -> list[str]:
    from huggingface_hub import HfApi
    print("Fetching file list from Hugging Face …", flush=True)
    api   = HfApi()
    files = api.list_repo_files(repo_id="thivux/phoaudiobook", repo_type="dataset")
    urls  = sorted(f for f in files if f.endswith(".parquet") and "train-" in f)
    print(f"Found {len(urls)} train parquet files.", flush=True)
    return urls

# =============================================================================
# Per-file speaker extraction
# =============================================================================

def _find_speaker_col(schema) -> int:
    """Return the column index for 'speaker', or -1 if absent."""
    for i, name in enumerate(schema.names):
        if name == "speaker":
            return i
    return -1


def extract_speakers_stats(url: str, fs) -> tuple[set[str], str | None]:
    """
    Read only Parquet row-group statistics for the 'speaker' column.
    This downloads kilobytes of metadata, not the full column data.
    Returns (speakers_found, error_message_or_None).
    """
    full_url = f"hf://datasets/thivux/phoaudiobook/{url}"
    retries, backoff = 5, 2.0

    for attempt in range(retries):
        try:
            with fs.open(full_url, "rb") as fh:
                meta = pq.read_metadata(fh)
            col_idx = _find_speaker_col(meta.schema)
            if col_idx < 0:
                return set(), None  # file has no speaker column — skip silently

            speakers: set[str] = set()
            all_stats_present  = True

            for rg in range(meta.num_row_groups):
                col_meta = meta.row_group(rg).column(col_idx)
                stats    = col_meta.statistics if col_meta.is_stats_set else None
                if stats and stats.has_min_max:
                    speakers.add(str(stats.min))
                    speakers.add(str(stats.max))
                else:
                    all_stats_present = False
                    break

            # If any row group lacked stats, fall back to full column scan
            if not all_stats_present:
                speakers = extract_speakers_full_scan(full_url, fs)

            return speakers, None

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(backoff * (1.5 ** attempt))
                fs = fsspec.filesystem("hf")  # re-init after network error
            else:
                return set(), f"{url}: {e}"

    return set(), f"{url}: max retries exceeded"


def extract_speakers_full_scan(full_url: str, fs) -> set[str]:
    """
    Full column scan fallback — reads only the 'speaker' column (no audio bytes).
    Used when row-group statistics are unavailable.
    """
    try:
        with fs.open(full_url, "rb") as fh:
            table = pq.read_table(fh, columns=["speaker"])
        return set(table["speaker"].to_pylist())
    except Exception as e:
        print(f"  [WARN] Full scan failed for {full_url}: {e}", flush=True)
        return set()

# =============================================================================
# Main
# =============================================================================

def main():
    try:
        parquet_files = get_parquet_urls()
    except Exception as e:
        print(f"Error fetching file list: {e}", file=sys.stderr)
        sys.exit(1)

    total = len(parquet_files)
    prog  = Progress.load(total)

    # Also retry files that previously errored (they might work now)
    files_to_process = [
        f for f in parquet_files
        if f not in prog.completed_files
    ]

    if not files_to_process:
        print(f"All {total} files processed. {len(prog.unique_speakers)} unique speakers.", flush=True)
        return

    fs = fsspec.filesystem("hf")
    num_threads = 6
    print(
        f"Processing {len(files_to_process)} remaining files "
        f"({len(prog.completed_files)} already done) with {num_threads} threads …",
        flush=True,
    )

    session_done = 0
    session_errors: list[str] = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(extract_speakers_stats, f, fs): f
            for f in files_to_process
        }
        for future in as_completed(futures):
            fname = futures[future]
            speakers, err = future.result()
            session_done += 1

            if err:
                session_errors.append(err)
                prog.error_files.add(fname)
                print(f"\n[ERR] {err}", flush=True)
            else:
                prog.unique_speakers.update(speakers)
                prog.completed_files.add(fname)
                prog.error_files.discard(fname)  # clear previous error if retried OK
                prog.save()

            # Progress line every file (overwrite in place)
            total_done = len(prog.completed_files)
            pct        = total_done / total * 100
            elapsed    = max(time.time() - start_time, 1e-3)
            speed      = session_done / elapsed
            eta        = (len(files_to_process) - session_done) / speed if speed > 0 else 0
            print(
                f"\r{total_done}/{total} ({pct:.1f}%) | "
                f"speakers={len(prog.unique_speakers)} | "
                f"{speed:.1f} files/s | ETA {eta:.0f}s   ",
                end="", flush=True,
            )

    print(flush=True)  # newline after progress line
    elapsed_total = time.time() - start_time
    print(f"\nDone in {elapsed_total:.1f}s.", flush=True)
    print(f"Unique speakers: {len(prog.unique_speakers)}", flush=True)
    print(f"Results → {OUTPUT_TXT} and {OUTPUT_JSON}", flush=True)

    if session_errors:
        print(
            f"\n{len(session_errors)} files errored (saved to progress file for retry):",
            file=sys.stderr,
        )
        for e in session_errors[:10]:
            print(f"  {e}", file=sys.stderr)
        if len(session_errors) > 10:
            print(f"  … and {len(session_errors) - 10} more.", file=sys.stderr)
        print("Re-run to retry failed files.", file=sys.stderr)


if __name__ == "__main__":
    main()