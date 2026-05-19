#!/usr/bin/env python3
"""
Metadata-first Northern Vietnamese dataset preparation pipeline.
Streams thivux/phoaudiobook, filters by speaker whitelist & dialect blocklists,
decodes audio, applies SNR check, and normalizes/G2P phonemizes.
"""
import os
import csv
import json
import re
import logging
import io
from pathlib import Path
import soundfile as sf
import librosa
import numpy as np
from datasets import load_dataset
from vinorm import TTSnorm
import datasets

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Config
# Note: These paths can be configured via environment variables or default to standard data paths
DATA_ROOT = Path(os.getenv("DATA_ROOT", "data"))
PROCESSED_DIR = DATA_ROOT / "processed"
MANIFEST_PATH = DATA_ROOT / "train_manifest.csv"
VERIFIED_SPEAKERS_FILE = DATA_ROOT / "verified_northern_speakers.txt"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR   = 24_000
MIN_DUR_S   = 1.0
MAX_DUR_S   = 15.0
MIN_SNR_DB  = 25.0

# Dialect blocklists (Lexical checks)
SOUTHERN_BLOCK = {
    "vầy", "bự", "hổng", "xài", "kêu", "nhậu", "mắc cười", "mèn ơi",
    "giùm", "dễ cưng", "hổm", "dzậy", "tui", "mầy", "ổng", "bả",
    "thổng", "dzìa", "nói dzậy", "dzô", "lổng", "hổng có",
    "mắc chi", "ghe", "bình thạnh", "sài gòn", "đa khoa", "chút xíu",
    "muống", "kiếm", "làm gì dữ vậy", "hú hồn", "miết", "uống nước ngọt"
}

CENTRAL_BLOCK = {
    "chi", "mô", "tê", "răng", "rứa", "nớ", "hỉ", "bây chừ",
    "đọi", "trốc", "eng", "ả", "mần", "hè nơ",
    "tội nghiệp", "răng rứa", "ngong", "bữa ni", "gửi vô", "trong nớ"
}

def load_verified_speakers():
    """Load and parse the verified speakers list from file."""
    if not VERIFIED_SPEAKERS_FILE.exists():
        log.warning("Verified speakers file %s not found. Pre-filtering disabled!", VERIFIED_SPEAKERS_FILE)
        return None
    
    speakers = set()
    with open(VERIFIED_SPEAKERS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                speakers.add(line)
    
    log.info("Loaded %d verified Northern speakers.", len(speakers))
    return speakers

def passes_dialect_filter(text: str) -> bool:
    """Check text against dialect blocklists."""
    tl = text.lower()
    if any(w in tl for w in SOUTHERN_BLOCK): return False
    if any(w in tl for w in CENTRAL_BLOCK):  return False
    return True

def estimate_snr(audio: np.ndarray) -> float:
    """Simple waveform-level SNR estimate (signal vs. quietest 10% frames)."""
    rms = np.sqrt(np.mean(audio**2))
    noise_floor = np.percentile(np.abs(audio), 10)
    if noise_floor < 1e-10: return 99.0
    return float(20 * np.log10(rms / noise_floor))

def to_ipa(text: str) -> str:
    """Normalize → G2P → IPA (Northern dialect)."""
    try:
        from viphoneme import vi2IPA
        normalized = TTSnorm(text, punc=True, unknown=False, lower=False)
        return vi2IPA(normalized, dialect="north")
    except Exception as e:
        log.warning("G2P failed for '%s': %s", text[:40], e)
        return ""

def process_audio(array: np.ndarray, orig_sr: int) -> np.ndarray | None:
    """Downmix, resample, peak-normalize, duration-gate, and SNR-gate."""
    # 1. Downmix to Mono if stereo
    if len(array.shape) > 1:
        array = np.mean(array, axis=1) if array.shape[1] == 2 else np.mean(array, axis=0)

    # 2. Resample to Target Sample Rate (24kHz)
    if orig_sr != TARGET_SR:
        audio = librosa.resample(array, orig_sr=orig_sr, target_sr=TARGET_SR)
    else:
        audio = array.copy()

    # 3. Peak-normalize to -0.5 dB (~0.95 amplitude) for uniform training volume
    max_val = np.max(np.abs(audio))
    if max_val > 1e-5:
        audio = (audio / max_val) * 0.95

    # 4. Duration Gate
    dur = len(audio) / TARGET_SR
    if not (MIN_DUR_S <= dur <= MAX_DUR_S):
        return None

    # 5. SNR Quality Gate
    if estimate_snr(audio) < MIN_SNR_DB:
        return None

    # 6. Convert to 16-bit PCM amplitude scale
    return (audio * 32767).clip(-32768, 32767).astype(np.int16)

def main():
    verified_speakers = load_verified_speakers()
    
    log.info("Streaming dataset 'thivux/phoaudiobook'...")
    # Load in streaming mode to keep memory low and download on the fly
    ds = load_dataset("thivux/phoaudiobook", split="train", streaming=True)
    
    # Crucial optimization: cast the audio column to disable automatic decoding
    # of rejected samples. This saves massive bandwidth and CPU overhead.
    ds_meta = ds.cast_column("audio", datasets.Audio(decode=False))
    
    records = []
    global_idx = 0
    retained_idx = 0
    
    # We iterate over the metadata first
    for item in ds_meta:
        global_idx += 1
        
        # 1. Fast speaker check
        spk = item.get("speaker", "")
        if verified_speakers is not None and spk not in verified_speakers:
            continue
            
        # 2. Text dialect filter
        transcript = (item.get("text") or "").strip()
        if not transcript or not passes_dialect_filter(transcript):
            continue
            
        # 3. Verified! Now fetch and decode the actual audio bytes
        audio_data = item.get("audio", {})
        try:
            audio_bytes = audio_data.get("bytes")
            if audio_bytes is None:
                continue
            
            with io.BytesIO(audio_bytes) as bio:
                array, orig_sr = sf.read(bio, dtype="float32")
        except Exception as e:
            log.warning("Audio decoding failed for item %d: %s", global_idx, e)
            continue
            
        # 4. Audio quality gates
        processed = process_audio(array, orig_sr)
        if processed is None:
            continue
            
        # 5. G2P conversion
        ipa = to_ipa(transcript)
        if not ipa:
            continue
            
        # 6. Save clip and record manifest
        fname = f"vi_north_{retained_idx:07d}.wav"
        fpath = PROCESSED_DIR / fname
        sf.write(str(fpath), processed, TARGET_SR, subtype="PCM_16")
        
        records.append((str(fpath), ipa, transcript))
        retained_idx += 1
        
        if retained_idx % 100 == 0:
            log.info("Processed %d / Retained %d samples", global_idx, retained_idx)
            
        # Hard limit for test/preview or stable GAN training bootstrap
        if os.getenv("SMOKE_TEST") == "true" and retained_idx >= 50:
            log.info("Smoke test: stopping data preparation early.")
            break

    # Write Manifest
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="|")
        w.writerows(records)
        
    log.info("Completed! Retained %d Northern samples -> %s", retained_idx, MANIFEST_PATH)

if __name__ == "__main__":
    main()
