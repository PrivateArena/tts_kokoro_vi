#!/usr/bin/env python3
"""
Metadata-first Northern Vietnamese dataset preparation pipeline.
Streams thivux/phoaudiobook, filters by speaker whitelist & dialect blocklists,
decodes audio, applies SNR check, and normalizes/G2P phonemizes.
"""
import os
import csv
import re
import logging
import io
import argparse
import concurrent.futures
from pathlib import Path
from typing import Optional
import soundfile as sf
import librosa
import numpy as np
from datasets import load_dataset
from vinorm import TTSnorm
import datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (overridable via CLI or env vars)
# ---------------------------------------------------------------------------
DATA_ROOT             = Path(os.getenv("DATA_ROOT", "data"))
PROCESSED_DIR         = DATA_ROOT / "processed"
MANIFEST_PATH         = DATA_ROOT / "train_manifest.csv"
VERIFIED_SPEAKERS_FILE = DATA_ROOT / "verified_northern_speakers.txt"

TARGET_SR   = 24_000
MIN_DUR_S   = 1.0
MAX_DUR_S   = 15.0
MIN_SNR_DB  = 25.0

# ---------------------------------------------------------------------------
# Dialect blocklists — word-boundary aware (avoids "chi" hitting "chính")
# ---------------------------------------------------------------------------
# Build as compiled regex patterns for speed + accuracy
_SOUTHERN_WORDS = [
    r"vầy", r"bự", r"hổng", r"xài", r"nhậu", r"mắc\s+cười", r"mèn\s+ơi",
    r"giùm", r"dễ\s+cưng", r"hổm", r"dzậy", r"tui", r"mầy", r"ổng", r"bả",
    r"dzìa", r"dzô", r"hổng\s+có", r"mắc\s+chi", r"chút\s+xíu", r"làm\s+gì\s+dữ\s+vậy",
    r"hú\s+hồn", r"miết",
]
_CENTRAL_WORDS = [
    r"\bchi\b", r"\bmô\b", r"\btê\b", r"\brăng\b", r"\brứa\b", r"\bnớ\b",
    r"\bhỉ\b", r"bây\s+chừ", r"\bđọi\b", r"\btrốc\b", r"\beng\b",
    r"\bmần\b", r"hè\s+nơ", r"răng\s+rứa", r"\bngong\b", r"bữa\s+ni",
]

SOUTHERN_RE = re.compile("|".join(_SOUTHERN_WORDS), re.IGNORECASE)
CENTRAL_RE  = re.compile("|".join(_CENTRAL_WORDS),  re.IGNORECASE)


def passes_dialect_filter(text: str) -> bool:
    """Check text against dialect blocklists using regex (word-boundary safe)."""
    if SOUTHERN_RE.search(text):
        return False
    if CENTRAL_RE.search(text):
        return False
    return True


# ---------------------------------------------------------------------------
# Speaker list
# ---------------------------------------------------------------------------
def load_verified_speakers() -> Optional[set]:
    if not VERIFIED_SPEAKERS_FILE.exists():
        log.warning(
            "Verified speakers file %s not found — speaker pre-filtering disabled.",
            VERIFIED_SPEAKERS_FILE,
        )
        return None
    speakers = set()
    with open(VERIFIED_SPEAKERS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                speakers.add(line)
    log.info("Loaded %d verified Northern speakers.", len(speakers))
    return speakers


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def estimate_snr(audio: np.ndarray) -> float:
    """
    Estimate SNR as RMS vs. 10th-percentile absolute amplitude.
    NOTE: This is a fast proxy, not a true signal/noise decomposition.
          For higher-fidelity rejection, consider a VAD-based approach.
    """
    rms = np.sqrt(np.mean(audio ** 2))
    noise_floor = np.percentile(np.abs(audio), 10)
    if noise_floor < 1e-10:
        return 99.0
    return float(20 * np.log10(rms / noise_floor))


def process_audio(array: np.ndarray, orig_sr: int) -> Optional[np.ndarray]:
    """
    Downmix → resample → peak-normalize → duration gate → SNR gate → int16.
    Returns None if the clip is rejected.
    """
    # 1. Downmix to mono (handles arbitrary channel layouts)
    if array.ndim > 1:
        array = array.mean(axis=-1)  # works for (samples, channels) or (channels, samples)

    # 2. Resample — check duration BEFORE resampling to fail fast on long clips
    raw_dur = len(array) / orig_sr
    if not (MIN_DUR_S <= raw_dur <= MAX_DUR_S):
        return None

    if orig_sr != TARGET_SR:
        # res_type="soxr_hq" is higher quality than default "soxr_vhq" and faster
        audio = librosa.resample(array, orig_sr=orig_sr, target_sr=TARGET_SR, res_type="soxr_hq")
    else:
        audio = array.copy()

    # 3. Peak-normalize to -0.5 dBFS
    max_val = np.max(np.abs(audio))
    if max_val > 1e-5:
        audio = (audio / max_val) * 0.95
    else:
        return None  # effectively silent — reject

    # 4. SNR quality gate (done AFTER normalisation so scale is consistent)
    if estimate_snr(audio) < MIN_SNR_DB:
        return None

    # 5. Convert to 16-bit PCM
    return (audio * 32767).clip(-32768, 32767).astype(np.int16)


# ---------------------------------------------------------------------------
# G2P  (import once, not per-call)
# ---------------------------------------------------------------------------
try:
    from viphoneme import vi2IPA as _vi2IPA
    _G2P_AVAILABLE = True
except ImportError:
    log.error("viphoneme not installed — G2P will be skipped. Run: pip install viphoneme")
    _G2P_AVAILABLE = False


def to_ipa(text: str) -> str:
    """Normalise Vietnamese text and convert to Northern-dialect IPA."""
    if not _G2P_AVAILABLE:
        return ""
    try:
        normalized = TTSnorm(text, punc=True, unknown=False, lower=False)
        return _vi2IPA(normalized, dialect="north")
    except Exception as e:
        log.warning("G2P failed for '%.40s': %s", text, e)
        return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Prepare Northern Vietnamese TTS dataset.")
    p.add_argument("--data-root",   default=os.getenv("DATA_ROOT", "data"))
    p.add_argument("--max-samples", type=int, default=0,
                   help="Hard cap on retained samples (0 = unlimited). Smoke test uses 50.")
    p.add_argument("--smoke-test",  action="store_true",
                   help="Alias for --max-samples 50 (fast CI check).")
    p.add_argument("--workers",     type=int, default=4,
                   help="Parallel workers for audio processing (default: 4).")
    return p.parse_args()


def main():
    args = parse_args()

    # Allow env-var smoke test compat as well
    smoke = args.smoke_test or (os.getenv("SMOKE_TEST", "").lower() == "true")
    max_samples = 50 if smoke else (args.max_samples or 0)

    # Re-root paths from CLI arg
    data_root     = Path(args.data_root)
    processed_dir = data_root / "processed"
    manifest_path = data_root / "train_manifest.csv"
    processed_dir.mkdir(parents=True, exist_ok=True)

    verified_speakers = load_verified_speakers()

    log.info("Streaming dataset 'thivux/phoaudiobook'…")
    ds = load_dataset("thivux/phoaudiobook", split="train", streaming=True)
    # Disable auto-decode so we only pay the decode cost for accepted samples
    ds_meta = ds.cast_column("audio", datasets.Audio(decode=False))

    records      = []
    global_idx   = 0
    rejected     = {"speaker": 0, "dialect": 0, "audio_decode": 0,
                    "audio_quality": 0, "g2p": 0, "silent": 0}
    retained_idx = 0

    for item in ds_meta:
        global_idx += 1

        # ── 1. Fast speaker check ──────────────────────────────────────────
        spk = (item.get("speaker") or "").strip()
        if verified_speakers is not None and spk not in verified_speakers:
            rejected["speaker"] += 1
            continue

        # ── 2. Text dialect filter ─────────────────────────────────────────
        transcript = (item.get("text") or "").strip()
        if not transcript:
            rejected["dialect"] += 1
            continue
        if not passes_dialect_filter(transcript):
            rejected["dialect"] += 1
            continue

        # ── 3. Decode audio bytes (only for surviving candidates) ──────────
        audio_data  = item.get("audio") or {}
        audio_bytes = audio_data.get("bytes")
        if not audio_bytes:
            rejected["audio_decode"] += 1
            continue
        try:
            with io.BytesIO(audio_bytes) as bio:
                array, orig_sr = sf.read(bio, dtype="float32")
        except Exception as e:
            log.warning("Audio decoding failed (item %d): %s", global_idx, e)
            rejected["audio_decode"] += 1
            continue

        # ── 4. Audio quality gates ─────────────────────────────────────────
        processed = process_audio(array, orig_sr)
        if processed is None:
            rejected["audio_quality"] += 1
            continue

        # ── 5. G2P conversion ──────────────────────────────────────────────
        ipa = to_ipa(transcript)
        if not ipa:
            rejected["g2p"] += 1
            continue

        # ── 6. Save clip + append manifest row ────────────────────────────
        fname = f"vi_north_{retained_idx:07d}.wav"
        fpath = processed_dir / fname
        sf.write(str(fpath), processed, TARGET_SR, subtype="PCM_16")

        records.append((str(fpath), ipa, transcript))
        retained_idx += 1

        if retained_idx % 100 == 0:
            log.info(
                "Scanned %d | Retained %d | Rejected: spk=%d dial=%d aud=%d+%d g2p=%d",
                global_idx, retained_idx,
                rejected["speaker"], rejected["dialect"],
                rejected["audio_decode"], rejected["audio_quality"],
                rejected["g2p"],
            )

        if max_samples and retained_idx >= max_samples:
            log.info("Reached max-samples cap (%d). Stopping.", max_samples)
            break

    # ── Write manifest ─────────────────────────────────────────────────────
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f, delimiter="|").writerows(records)

    log.info(
        "Done. Retained %d Northern samples → %s\n"
        "Rejection breakdown: %s",
        retained_idx, manifest_path, rejected,
    )


if __name__ == "__main__":
    main()