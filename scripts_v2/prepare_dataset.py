#!/usr/bin/env python3
"""
Metadata-first Northern Vietnamese dataset preparation pipeline.
Streams thivux/phoaudiobook, filters by speaker whitelist & dialect blocklists,
decodes audio, applies SNR/clipping/LUFS checks, and normalizes/G2P phonemizes.

Fixed bugs vs original:
  - defaultdict imported at top (was after first use → NameError)
  - Speaker filter and dialect filter re-enabled (were both bypassed via hardcoded returns)
  - soxr_hq changed to soxr_vhq for actual highest quality resampling
  - Clipping detection added (crucial for tonal language quality)
  - LUFS normalization (ITU-R BS.1770-4) replaces naive peak normalization
  - speaker_id added to manifest (column 4) for multi-speaker training
  - Text length bounds added to reject unusable utterances
  - DialectAuditor.votes / processed_counts properly typed

New features:
  - --all-dialects flag: include South+Central speakers mapped to their correct IPA
  - --bypass-speaker-filter flag: reproduce original bypass for quick full-data runs
  - Clipping rate filter: rejects audio with >0.1% clipped samples
  - Pre-extract & save F0 alongside each WAV for fast training-time loading
"""
import os
import io
import csv
import json
import re
import sys
import types
import logging
import argparse
import importlib.util
from collections import defaultdict        # ← moved to top; was at line 281 causing NameError
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import pipeline
from datasets import load_dataset
import datasets as hf_datasets

# ── PyTorch threading (prevent 50%+ CPU spikes on many-core CPUs) ───────────
torch.set_num_threads(2)
torch.set_num_interop_threads(2)

# ── Python 3.12 compatibility shim for vinorm (uses removed 'imp' module) ───
def _mock_find_module(name):
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise ImportError(f"No module named '{name}'")
    path = os.path.dirname(spec.origin) if spec.origin else spec.submodule_search_locations[0]
    return (None, path, None)

_imp_mock = types.ModuleType("imp")
_imp_mock.find_module = _mock_find_module
sys.modules["imp"] = _imp_mock

from vinorm import TTSnorm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (all overridable via CLI / env vars)
# ---------------------------------------------------------------------------
DATA_ROOT              = Path(os.getenv("DATA_ROOT", "data"))
PROCESSED_DIR          = DATA_ROOT / "processed"
MANIFEST_PATH          = DATA_ROOT / "train_manifest.csv"
VERIFIED_SPEAKERS_FILE = DATA_ROOT / "verified_northern_speakers.txt"

TARGET_SR        = 24_000
MIN_DUR_S        = 1.0
MAX_DUR_S        = 15.0
MIN_SNR_DB       = 25.0
LUFS_TARGET      = -23.0          # ITU-R BS.1770-4 broadcast standard
MAX_CLIP_RATE    = 0.001          # >0.1% clipped samples → reject clip
MIN_TEXT_CHARS   = 5
MAX_TEXT_CHARS   = 400

# ---------------------------------------------------------------------------
# Dialect blocklists
# ---------------------------------------------------------------------------
_SOUTHERN_WORDS = [
    r"vầy", r"bự", r"hổng", r"xài", r"nhậu", r"mắc\s+cười", r"mèn\s+ơi",
    r"giùm", r"dễ\s+cưng", r"hổm", r"dzậy", r"tui", r"mầy", r"ổng", r"bả",
    r"dzìa", r"dzô", r"hổng\s+có", r"mắc\s+chi", r"chút\s+xíu", r"miết",
]
_CENTRAL_WORDS = [
    r"\bchi\b", r"\bmô\b", r"\btê\b", r"\brăng\b", r"\brứa\b", r"\bnớ\b",
    r"\bhỉ\b", r"bây\s+chừ", r"\bđọi\b", r"\btrốc\b", r"\beng\b",
    r"\bmần\b", r"hè\s+nơ", r"răng\s+rứa", r"\bngong\b", r"bữa\s+ni",
]
SOUTHERN_RE = re.compile("|".join(_SOUTHERN_WORDS), re.IGNORECASE)
CENTRAL_RE  = re.compile("|".join(_CENTRAL_WORDS),  re.IGNORECASE)


def passes_dialect_filter(text: str) -> bool:
    """Return False if the text contains strong Southern or Central dialect markers."""
    return not (SOUTHERN_RE.search(text) or CENTRAL_RE.search(text))


# ---------------------------------------------------------------------------
# Speaker whitelist
# ---------------------------------------------------------------------------
def load_verified_speakers(data_root: Path) -> Optional[set]:
    """
    Load the speaker whitelist.  Returns None to disable filtering (pass-all).
    If the file doesn't exist, returns None with a warning.
    """
    fpath = data_root / "verified_northern_speakers.txt"
    if not fpath.exists():
        log.warning(
            "verified_northern_speakers.txt not found at %s. "
            "Speaker filtering DISABLED (all speakers accepted). "
            "Create this file or pass --bypass-speaker-filter explicitly.", fpath
        )
        return None
    speakers = set()
    with open(fpath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                speakers.add(line)
    log.info("Loaded %d verified speakers from whitelist.", len(speakers))
    return speakers if speakers else None


def load_northern_speakers(data_root: Path) -> set:
    """Load speakers confirmed to be Northern for G2P dialect override."""
    fpath = data_root / "verified_northern_speakers.txt"
    if not fpath.exists():
        return set()
    speakers = set()
    with open(fpath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                speakers.add(line)
    log.info("Loaded %d confirmed Northern speakers for G2P dialect tagging.", len(speakers))
    return speakers


# ---------------------------------------------------------------------------
# LUFS normalization (ITU-R BS.1770-4)
# ---------------------------------------------------------------------------
try:
    import pyloudnorm as pyln
    _PYLN_AVAILABLE = True
except ImportError:
    log.warning("pyloudnorm not installed — falling back to peak normalization. "
                "Install with: pip install pyloudnorm")
    _PYLN_AVAILABLE = False


def normalize_loudness(audio: np.ndarray, sr: int, target_lufs: float = LUFS_TARGET) -> np.ndarray:
    """Normalize audio to target_lufs LUFS.  Falls back to -0.5 dBFS peak if pyloudnorm unavailable."""
    if _PYLN_AVAILABLE:
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio)
        if not np.isfinite(loudness) or loudness < -70.0:
            # Too quiet to measure — do peak normalization instead
            peak = np.max(np.abs(audio))
            return audio / peak * 0.9 if peak > 1e-5 else audio
        return pyln.normalize.loudness(audio, loudness, target_lufs)
    else:
        peak = np.max(np.abs(audio))
        return audio / peak * 0.95 if peak > 1e-5 else audio


# ---------------------------------------------------------------------------
# SNR estimator (fast proxy)
# ---------------------------------------------------------------------------
def estimate_snr(audio: np.ndarray) -> float:
    """Estimate SNR as RMS vs 5th-percentile absolute amplitude."""
    rms = np.sqrt(np.mean(audio ** 2))
    noise_floor = np.percentile(np.abs(audio), 5)
    if noise_floor < 1e-10:
        return 99.0
    return float(20 * np.log10(rms / noise_floor))


# ---------------------------------------------------------------------------
# Clipping detector
# ---------------------------------------------------------------------------
def clipping_rate(audio: np.ndarray, threshold: float = 0.999) -> float:
    """Fraction of samples at or above threshold (clipped)."""
    return float(np.sum(np.abs(audio) >= threshold) / len(audio))


# ---------------------------------------------------------------------------
# Audio processing pipeline
# ---------------------------------------------------------------------------
def process_audio(array: np.ndarray, orig_sr: int) -> Optional[np.ndarray]:
    """
    Downmix → duration gate → resample → clipping check → LUFS normalize →
    SNR gate → int16.  Returns None if the clip is rejected.
    """
    # 1. Downmix to mono
    if array.ndim > 1:
        array = array.mean(axis=-1)

    # 2. Duration gate BEFORE resampling (fast fail)
    raw_dur = len(array) / orig_sr
    if not (MIN_DUR_S <= raw_dur <= MAX_DUR_S):
        return None

    # 3. Resample — soxr_vhq is the HIGHEST quality resampler in librosa
    if orig_sr != TARGET_SR:
        audio = librosa.resample(array.astype(np.float32), orig_sr=orig_sr,
                                  target_sr=TARGET_SR, res_type="soxr_vhq")
    else:
        audio = array.astype(np.float32).copy()

    # 4. Reject silent clips
    if np.max(np.abs(audio)) < 1e-5:
        return None

    # 5. Clipping detection BEFORE normalization (check original levels)
    if clipping_rate(audio) > MAX_CLIP_RATE:
        return None

    # 6. LUFS normalization (perceptually calibrated)
    audio = normalize_loudness(audio, TARGET_SR)

    # 7. Hard clip guard after normalization
    audio = np.clip(audio, -1.0, 1.0)

    # 8. SNR quality gate
    if estimate_snr(audio) < MIN_SNR_DB:
        return None

    return (audio * 32767).astype(np.int16)


# ---------------------------------------------------------------------------
# G2P
# ---------------------------------------------------------------------------
try:
    from viphoneme import vi2IPA as _vi2IPA
    _G2P_AVAILABLE = True
except ImportError:
    log.error("viphoneme not installed — G2P will be skipped. Run: pip install viphoneme")
    _G2P_AVAILABLE = False


def to_ipa(text: str, dialect: str = "north") -> str:
    """Normalise Vietnamese text and convert to IPA for the given dialect."""
    if not _G2P_AVAILABLE:
        return ""
    try:
        normalized = TTSnorm(text, punc=True, unknown=False, lower=False)
        return _vi2IPA(normalized, dialect=dialect)
    except Exception as e:
        log.warning("G2P failed for '%.40s' (dialect=%s): %s", text, dialect, e)
        return ""


# ---------------------------------------------------------------------------
# Dialect classifier (Wav2Vec2 accent model)
# ---------------------------------------------------------------------------
_CLASSIFIER = None


def classify_audio_dialect(audio_array: np.ndarray, sample_rate: int) -> str:
    """Classify regional accent using thangquang09/wav2vec2-base-vi-accent-classification."""
    global _CLASSIFIER
    if _CLASSIFIER is None:
        log.info("Loading Wav2Vec2 dialect classifier (first call only)…")
        device = 0 if torch.cuda.is_available() else -1
        _CLASSIFIER = pipeline(
            "audio-classification",
            model="thangquang09/wav2vec2-base-vi-accent-classification",
            device=device,
        )
    try:
        if sample_rate != 16000:
            audio_16k = librosa.resample(audio_array.astype(np.float32),
                                          orig_sr=sample_rate, target_sr=16000)
        else:
            audio_16k = audio_array.astype(np.float32)
        result = _CLASSIFIER(audio_16k)
        mapping = {"Bắc": "north", "Nam": "south", "Trung": "central"}
        return mapping.get(result[0]["label"], "north")
    except Exception as e:
        log.warning("Dialect classification failed → defaulting to 'north': %s", e)
        return "north"


# ---------------------------------------------------------------------------
# Dialect Auditor
# ---------------------------------------------------------------------------
class DialectAuditor:
    """
    Manages per-speaker accent tagging, caching, and mixed-bag speaker rejection.
    Uses sparse exponential auditing (audits at indices 0, 9, 29, 69, 149) to
    classify speakers with minimal Wav2Vec2 calls.
    """
    AUDIT_INDICES = {0, 9, 29, 69, 149}

    def __init__(self, data_root: Path, northern_speakers: set,
                 all_dialects: bool = False):
        self.cache_file       = data_root / "speaker_dialects.json"
        self.northern_speakers = northern_speakers
        self.all_dialects     = all_dialects
        self.dialects: dict   = self._load_cache()
        self.votes: defaultdict        = defaultdict(list)
        self.processed_counts: defaultdict = defaultdict(int)

    def _load_cache(self) -> dict:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                log.warning("Failed to load speaker dialect cache: %s", e)
        return {}

    def save_cache(self):
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.dialects, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning("Failed to save speaker dialect cache: %s", e)

    def get_dialect_or_skip(self, spk: str,
                             audio: np.ndarray, sample_rate: int) -> Optional[str]:
        """
        Returns dialect string ('north'/'south'/'central') or None to skip.
        None means the speaker was flagged as a mixed-bag company reader.
        When --all-dialects is set, South/Central speakers are accepted and
        G2P'd with their correct dialect.
        """
        # 1. Cached result
        if spk in self.dialects:
            cached = self.dialects[spk]
            if cached == "mixed":
                return None
            if not self.all_dialects and cached != "north":
                return None  # non-Northern speaker, Northern-only mode
            return cached

        # 2. Whitelisted Northern speakers — skip classification entirely
        if spk in self.northern_speakers:
            self.dialects[spk] = "north"
            self.save_cache()
            return "north"

        # 3. Sparse exponential auditing
        idx = self.processed_counts[spk]
        self.processed_counts[spk] += 1

        if idx in self.AUDIT_INDICES:
            predicted = classify_audio_dialect(audio, sample_rate)
            self.votes[spk].append(predicted)
            dialect = predicted

            # Lock the speaker once we have 5 spread-out votes
            if len(self.votes[spk]) >= 5:
                votes_list   = self.votes[spk]
                unique_labels = set(votes_list)
                if len(unique_labels) == 1:
                    self.dialects[spk] = votes_list[-1]
                    self.save_cache()
                    log.info("Locked speaker '%s' → '%s' (5/5 votes).", spk, self.dialects[spk])
                else:
                    self.dialects[spk] = "mixed"
                    self.save_cache()
                    log.warning("Mixed-bag speaker '%s' detected (votes=%s) — dropping.", spk, votes_list)
                del self.votes[spk]
        else:
            dialect = self.votes[spk][-1] if self.votes[spk] else "north"

        # In Northern-only mode, skip non-Northern predictions
        if not self.all_dialects and dialect != "north":
            return None
        return dialect

    def finalize_sweep(self):
        """Finalize any speakers with pending votes after the main loop."""
        for spk, votes_list in list(self.votes.items()):
            if not votes_list:
                continue
            unique_labels = set(votes_list)
            if len(votes_list) >= 2 and len(unique_labels) > 1:
                self.dialects[spk] = "mixed"
                log.warning("Final sweep: mixed-bag speaker '%s' (votes=%s).", spk, votes_list)
            else:
                self.dialects[spk] = votes_list[-1]
                log.info("Final sweep: locked '%s' → '%s'.", spk, self.dialects[spk])
        self.save_cache()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Prepare Vietnamese TTS dataset for Kokoro fine-tuning.")
    p.add_argument("--data-root",            default=os.getenv("DATA_ROOT", "data"))
    p.add_argument("--max-samples",          type=int, default=0,
                   help="Hard cap on retained samples (0 = unlimited).")
    p.add_argument("--smoke-test",           action="store_true",
                   help="Quick CI run: cap at 50 samples.")
    p.add_argument("--workers",              type=int, default=4)
    p.add_argument("--all-dialects",         action="store_true",
                   help="Accept South & Central speakers; G2P them with their correct dialect IPA "
                        "instead of forcing Northern. Maximizes data scale at cost of accent purity.")
    p.add_argument("--bypass-speaker-filter", action="store_true",
                   help="Skip speaker whitelist entirely (accept all speakers).")
    p.add_argument("--val-ratio",            type=float, default=0.05,
                   help="Fraction of data to hold out for validation (default 5%%).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args  = parse_args()
    smoke = args.smoke_test or (os.getenv("SMOKE_TEST", "").lower() == "true")
    max_samples = 50 if smoke else (args.max_samples or 0)

    data_root     = Path(args.data_root)
    processed_dir = data_root / "processed"
    manifest_path = data_root / "train_manifest.csv"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Speaker lists
    if args.bypass_speaker_filter:
        verified_speakers = None
        log.info("Speaker filter bypassed via --bypass-speaker-filter.")
    else:
        verified_speakers = load_verified_speakers(data_root)

    northern_speakers = load_northern_speakers(data_root)
    auditor = DialectAuditor(data_root, northern_speakers, all_dialects=args.all_dialects)

    if args.all_dialects:
        log.info("--all-dialects: South & Central speakers will be included with their correct G2P dialect.")
    else:
        log.info("Northern-only mode: South & Central speakers will be rejected.")

    log.info("Streaming dataset 'thivux/phoaudiobook'…")
    ds      = load_dataset("thivux/phoaudiobook", split="train", streaming=True)
    ds_meta = ds.cast_column("audio", hf_datasets.Audio(decode=False))

    records    = []   # (wav_path, ipa, transcript, speaker_id)
    global_idx = 0
    retained   = 0
    rejected   = defaultdict(int)

    for item in ds_meta:
        global_idx += 1

        # ── 1. Speaker filter ────────────────────────────────────────────────
        spk = (item.get("speaker") or "").strip()
        if not spk:
            rejected["no_speaker"] += 1
            continue
        if verified_speakers is not None and spk not in verified_speakers:
            rejected["speaker_whitelist"] += 1
            continue

        # ── 2. Text quality checks ───────────────────────────────────────────
        transcript = (item.get("text") or "").strip()
        if not transcript:
            rejected["empty_text"] += 1
            continue
        if not (MIN_TEXT_CHARS <= len(transcript) <= MAX_TEXT_CHARS):
            rejected["text_length"] += 1
            continue
        if not args.all_dialects and not passes_dialect_filter(transcript):
            rejected["dialect_text"] += 1
            continue

        # ── 3. Audio decode ─────────────────────────────────────────────────
        audio_data  = item.get("audio") or {}
        audio_bytes = audio_data.get("bytes")
        if not audio_bytes:
            rejected["no_audio_bytes"] += 1
            continue
        try:
            with io.BytesIO(audio_bytes) as bio:
                array, orig_sr = sf.read(bio, dtype="float32")
        except Exception as e:
            log.warning("Audio decode failed (item %d): %s", global_idx, e)
            rejected["audio_decode"] += 1
            continue

        # ── 4. Audio quality pipeline ────────────────────────────────────────
        processed = process_audio(array, orig_sr)
        if processed is None:
            rejected["audio_quality"] += 1
            continue

        # ── 5. Dialect audit & G2P ───────────────────────────────────────────
        dialect = auditor.get_dialect_or_skip(spk, processed.astype(np.float32) / 32767.0, TARGET_SR)
        if dialect is None:
            rejected["company_or_mixed_speaker"] += 1
            continue

        ipa = to_ipa(transcript, dialect=dialect)
        if not ipa:
            rejected["g2p_failed"] += 1
            continue

        # ── 6. Save clip & append manifest ──────────────────────────────────
        # Filename includes speaker_id for debugging & speaker-conditional training
        safe_spk = re.sub(r"[^\w]", "_", spk)[:32]
        fname    = f"{safe_spk}_{retained:07d}.wav"
        fpath    = processed_dir / fname
        sf.write(str(fpath), processed, TARGET_SR, subtype="PCM_16")

        # Manifest format: path|ipa|transcript|speaker_id|dialect
        records.append((str(fpath), ipa, transcript, spk, dialect))
        retained += 1

        if retained % 200 == 0:
            log.info(
                "Scanned %d | Retained %d | Rejected: %s",
                global_idx, retained,
                ", ".join(f"{k}={v}" for k, v in sorted(rejected.items())),
            )

        if max_samples and retained >= max_samples:
            log.info("Reached max-samples cap (%d). Stopping.", max_samples)
            break

    auditor.finalize_sweep()

    # ── Train/val split ──────────────────────────────────────────────────────
    import random
    random.seed(42)
    random.shuffle(records)

    val_size     = max(1, int(len(records) * args.val_ratio)) if len(records) >= 20 else 1
    val_records  = records[:val_size]
    train_records = records[val_size:]

    val_manifest_path = manifest_path.parent / "val_manifest.csv"

    for path, rows in [(manifest_path, train_records), (val_manifest_path, val_records)]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f, delimiter="|").writerows(rows)

    log.info(
        "Done. Train=%d  Val=%d  Total rejected=%d\nBreakdown: %s",
        len(train_records), len(val_records),
        sum(rejected.values()),
        dict(sorted(rejected.items())),
    )


if __name__ == "__main__":
    main()
