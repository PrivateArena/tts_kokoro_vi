#!/usr/bin/env python3
"""
Metadata-first Northern Vietnamese dataset preparation pipeline.
Streams thivux/phoaudiobook, filters by speaker whitelist & dialect blocklists,
decodes audio, applies SNR check, and normalizes/G2P phonemizes.
"""
import os
import json
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
import torch
from transformers import pipeline
from datasets import load_dataset

# Limit PyTorch CPU thread pool to prevent 50%+ CPU spikes on multi-core CPUs like 9700X
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
# ── Python 3.12 Compatibility Shim for vinorm ───────────────────────────
import sys
import types
import importlib.util
def mock_find_module(name):
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise ImportError(f"No module named '{name}'")
    path = os.path.dirname(spec.origin) if spec.origin else spec.submodule_search_locations[0]
    return (None, path, None)
imp_mock = types.ModuleType("imp")
imp_mock.find_module = mock_find_module
sys.modules["imp"] = imp_mock

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
    """Reject transcripts with obvious Southern or Central blank lexical markers."""
    if SOUTHERN_RE.search(text):
        return False
    if CENTRAL_RE.search(text):
        return False
    return True


# ---------------------------------------------------------------------------
# Speaker list
# ---------------------------------------------------------------------------
def load_verified_speakers(data_root: Path | None = None) -> Optional[set]:
    """Load the Northern speaker whitelist so the filter is actually active.
    
    Returns None only when the whitelist file is absent (unlimited pass-through).
    Pass data_root=DATA_ROOT to lock to this project's verified list.
    """
    fpath = Path(data_root) / "verified_northern_speakers.txt" if data_root else VERIFIED_SPEAKERS_FILE
    if not fpath.exists():
        log.warning(
            "verified_northern_speakers.txt not found at %s — speaker whitelist active, "
            "UNLIMITED pass-accept (no filtering). Create the file or pass --bypass-speaker-filter.",
            fpath,
        )
        return None
    speakers = set()
    with open(fpath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                speakers.add(line)
    if not speakers:
        log.warning("verified_northern_speakers.txt is empty — unlimited pass-accept.")
        return None
    log.info("Speaker whitelist active: %d verified Northern speakers.", len(speakers))
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


def to_ipa(text: str, dialect: str = "north") -> str:
    """Normalise Vietnamese text and convert to Northern or Southern dialect IPA."""
    if not _G2P_AVAILABLE:
        return ""
    try:
        normalized = TTSnorm(text, punc=True, unknown=False, lower=False)
        return _vi2IPA(normalized, dialect=dialect)
    except Exception as e:
        log.warning("G2P failed for '%.40s' with dialect %s: %s", text, dialect, e)
        return ""


# Helper to load Northern speakers for dynamic tagging
def load_northern_speakers(data_root: Path) -> set:
    fpath = data_root / "verified_northern_speakers.txt"
    if not fpath.exists():
        log.warning("Northern speaker whitelist %s not found. All speakers default to Southern G2P.", fpath)
        return set()
    speakers = set()
    with open(fpath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                speakers.add(line)
    log.info("Loaded %d verified Northern speakers for G2P dialect tagging.", len(speakers))
    return speakers


SPEAKER_DIALECTS_FILE = Path("data/speaker_dialects.json")
_CLASSIFIER = None


class DialectAuditor:
    """Manages regional dialect tagging, cache loading/saving, and company speaker filtering."""
    def __init__(self, data_root: Path, northern_speakers: set, all_dialects: bool = False):
        self.cache_file = data_root / "speaker_dialects.json"
        self.northern_speakers = northern_speakers
        self.all_dialects    = all_dialects
        self.dialects        = self._load_cache()
        self.votes           = defaultdict(list)
        self.processed_counts = defaultdict(int)
        # Exponential indices to sample spread-out chapters/narrators
        self.audit_indices = {0, 9, 29, 69, 149}

    def _load_cache(self) -> dict:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                log.warning("Failed to load speaker dialects file: %s", e)
        return {}

    def save_cache(self):
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.dialects, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning("Failed to save speaker dialects: %s", e)

    def get_dialect_or_skip(self, spk: str, audio: np.ndarray, sample_rate: int) -> Optional[str]:
        """
        Retrieves the dialect for a speaker.
        Returns 'north', 'south', or 'central'.
        Returns None if the speaker is flagged as a mixed-bag company speaker (should be skipped).
        """
        # 1. Check cache first
        if spk in self.dialects:
            cached_val = self.dialects[spk]
            return None if cached_val == "mixed" else cached_val

        # 2. Whitelisted Northern speakers bypass classification
        if spk in self.northern_speakers:
            return "north"

        # 3. Dynamic Exponential Auditing
        idx = self.processed_counts[spk]
        self.processed_counts[spk] += 1

        if idx in self.audit_indices:
            predicted = classify_audio_dialect(audio, sample_rate)
            self.votes[spk].append(predicted)
            dialect = predicted

            # Audit and lock when we hit 5 spread-out samples (at index 149)
            if len(self.votes[spk]) >= 5:
                votes = self.votes[spk]
                unique_labels = set(votes)
                if len(unique_labels) == 1:
                    self.dialects[spk] = list(unique_labels)[0]
                    self.save_cache()
                    log.info("Locked speaker '%s' as consistent '%s' accent (5/5 spread votes agreement).", spk, self.dialects[spk])
                else:
                    self.dialects[spk] = "mixed"
                    self.save_cache()
                    log.warning("Detected Mixed-Bag Company Speaker for '%s' (votes: %s). Dropping speaker to prevent training noise.", spk, votes)
                
                # Clear votes for this speaker
                del self.votes[spk]
        else:
            # Reuse the last prediction to avoid redundant Wav2Vec2 calls
            dialect = self.votes[spk][-1] if self.votes[spk] else "north"

        return dialect

    def finalize_sweep(self):
        """Final sweep audit for remaining active speakers with pending votes."""
        swept_any = False
        for spk, votes in list(self.votes.items()):
            if votes:
                unique_labels = set(votes)
                if len(votes) >= 2 and len(unique_labels) > 1:
                    self.dialects[spk] = "mixed"
                    log.warning("Final Sweep: Flagged mixed-bag speaker '%s' (votes: %s).", spk, votes)
                else:
                    self.dialects[spk] = votes[-1]
                    log.info("Final Sweep: Locked speaker '%s' as consistent '%s' (votes: %s).", spk, self.dialects[spk], votes)
                swept_any = True
        
        if swept_any:
            self.save_cache()


from collections import defaultdict


def classify_audio_dialect(audio_array: np.ndarray, sample_rate: int) -> str:
    """Dynamically classify the speech regional accent on the fly using a Wav2Vec2 classifier."""
    global _CLASSIFIER
    if _CLASSIFIER is None:
        log.info("Initializing Hugging Face Wav2Vec2 regional dialect classifier (thangquang09)...")
        # Auto-detect CUDA if available
        device = 0 if torch.cuda.is_available() else -1
        _CLASSIFIER = pipeline(
            "audio-classification",
            model="thangquang09/wav2vec2-base-vi-accent-classification",
            device=device
        )

    try:
        # Resample to 16kHz for Wav2Vec2 model if needed
        if sample_rate != 16000:
            audio_16k = librosa.resample(audio_array.astype(np.float32), orig_sr=sample_rate, target_sr=16000)
        else:
            audio_16k = audio_array.astype(np.float32)

        res = _CLASSIFIER(audio_16k)
        best_label = res[0]["label"]
        mapping = {"Bắc": "north", "Nam": "south", "Trung": "central"}
        pred = mapping.get(best_label, "north")
        return pred
    except Exception as e:
        log.warning("Dialect model prediction failed, falling back to 'north': %s", e)
        return "north"


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
    p.add_argument("--bypass-speaker-filter", action="store_true",
                   help="Skip speaker whitelist — accept all speakers.")
    p.add_argument("--all-dialects", action="store_true",
                   help="Accept South + Central dialects with dialect-correct G2P.")
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

    # Speaker & dialect setup
    if args.bypass_speaker_filter:
        verified_speakers = None
        log.info("Speaker whitelist BYPASSED (--bypass-speaker-filter).")
    else:
        verified_speakers = load_verified_speakers(data_root)

    northern_speakers = load_northern_speakers(data_root)

    # DialectAuditor: when --bypass-speaker-filter is active, whitelist is empty so
    # Wav2Vec2 classifier handles tagging; when it's off, whitelisted speakers skip Wav2Vec2.
    # With --all-dialects, South/Central speakers are retained with their own dialect G2P.
    if not args.all_dialects:
        log.info("Northern-only mode: South & Central speakers will be rejected by Wav2Vec2 tag.")
    else:
        log.info("--all-dialects: South & Central speakers retained with dialect-correct G2P.")

    auditor = DialectAuditor(data_root, northern_speakers, all_dialects=args.all_dialects)

    log.info("Streaming dataset 'thivux/phoaudiobook'…")
    ds = load_dataset("thivux/phoaudiobook", split="train", streaming=True)
    # Disable auto-decode so we only pay the decode cost for accepted samples
    ds_meta = ds.cast_column("audio", datasets.Audio(decode=False))

    records      = []
    global_idx   = 0
    rejected     = {"speaker": 0, "dialect": 0, "audio_decode": 0,
                    "audio_quality": 0, "g2p": 0, "silent": 0, "company_speaker": 0}
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

        # ── 5. Dynamic Sentence-Level Dialect G2P Tagging with Company Speaker Filtering ──
        dialect = auditor.get_dialect_or_skip(spk, processed, TARGET_SR)
        if dialect is None:
            rejected["company_speaker"] += 1
            continue

        ipa = to_ipa(transcript, dialect=dialect)
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

    # ── Final Sweep Audit for remaining active speakers ───────────────────
    auditor.finalize_sweep()

    # ── Split and write train/val manifests ──────────────────────────────────
    import random
    random.seed(42)  # ensure reproducible train/val splits
    random.shuffle(records)

    val_size = max(1, int(len(records) * 0.05)) if len(records) >= 20 else 1
    val_records = records[:val_size]
    train_records = records[val_size:]

    val_manifest_path = manifest_path.parent / "val_manifest.csv"

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f, delimiter="|").writerows(train_records)

    with open(val_manifest_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f, delimiter="|").writerows(val_records)

    log.info(
        "Done. Saved %d training samples to %s and %d validation samples to %s\n"
        "Rejection breakdown: %s",
        len(train_records), manifest_path, len(val_records), val_manifest_path, rejected,
    )


if __name__ == "__main__":
    main()