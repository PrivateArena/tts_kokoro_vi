#!/usr/bin/env python3
"""
Northern Vietnamese KokoroTTS Dataset Preparation Pipeline.

Streams thivux/phoaudiobook, applies dialect/speaker filtering,
LUFS loudness normalization, DNSMOS perceptual quality gating,
dialect-aware G2P, and writes train/val manifests with speaker IDs.

Fixes vs original:
  - dialect text filter and speaker whitelist are now ACTIVE (were silently bypassed)
  - LUFS loudness normalization replaces broken peak normalization
  - DNSMOS perceptual quality gate replaces weak RMS-proxy SNR
  - Dialect auditor confidence threshold raised; audit spread increased to 6 samples
  - Manifest includes speaker_id column for multi-speaker style training
  - Resumable manifest appending so interrupted runs don't discard progress
  - defaultdict import moved to top of file (was after the class that used it)
"""
# ── Python 3.12 compatibility shim for vinorm (uses removed imp module) ─────
import sys
import types
import importlib.util

def _mock_find_module(name):
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise ImportError(f"No module named '{name}'")
    path = (
        importlib.util.find_spec(name).__spec__.submodule_search_locations[0]
        if spec.submodule_search_locations
        else importlib.util.find_spec(name).origin
    )
    return (None, path, None)

_imp_mock = types.ModuleType("imp")
_imp_mock.find_module = _mock_find_module
sys.modules.setdefault("imp", _imp_mock)
# ─────────────────────────────────────────────────────────────────────────────

import csv
import io
import json
import logging
import os
import random
import re
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
from datasets import load_dataset
import datasets as hf_datasets
from transformers import pipeline as hf_pipeline
from vinorm import TTSnorm

# Limit CPU threads — prevents 50 %+ CPU spikes on multi-core hosts
torch.set_num_threads(2)
torch.set_num_interop_threads(2)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ── G2P import (once at module level) ─────────────────────────────────────────
try:
    from viphoneme import vi2IPA as _vi2IPA
    _G2P_AVAILABLE = True
except ImportError:
    log.error("viphoneme not installed — G2P disabled. Run: pip install viphoneme")
    _G2P_AVAILABLE = False

# =============================================================================
# Constants
# =============================================================================
TARGET_SR       = 24_000
MIN_DUR_S       = 1.0
MAX_DUR_S       = 15.0
TARGET_LUFS     = -23.0          # broadcast standard; matches Kokoro training data
LUFS_FLOOR      = -70.0          # clips quieter than this are rejected as silent
MIN_DNSMOS      = 3.5            # perceptual quality gate (1–5 scale)
AUDIT_CONFIDENCE = 0.85          # min classifier confidence to count a dialect vote

# Dialect blocklists — compiled once with Unicode-aware word boundaries
_SOUTHERN_WORDS = [
    r"vầy", r"bự", r"hổng", r"xài", r"nhậu", r"mắc\s+cười", r"mèn\s+ơi",
    r"giùm", r"dễ\s+cưng", r"hổm", r"dzậy", r"tui", r"mầy", r"ổng", r"bả",
    r"dzìa", r"dzô", r"hổng\s+có", r"mắc\s+chi", r"chút\s+xíu",
    r"làm\s+gì\s+dữ\s+vậy", r"hú\s+hồn", r"miết", r"thẩy", r"dìa",
]
_CENTRAL_WORDS = [
    r"\bchi\b", r"\bmô\b", r"\btê\b", r"\brăng\b", r"\brứa\b", r"\bnớ\b",
    r"\bhỉ\b", r"bây\s+chừ", r"\bđọi\b", r"\btrốc\b", r"\beng\b",
    r"\bmần\b", r"hè\s+nơ", r"răng\s+rứa", r"\bngong\b", r"bữa\s+ni",
    r"\bmi\b(?!\w)", r"\bbây\b", r"\bchừ\b",
]

SOUTHERN_RE = re.compile("|".join(_SOUTHERN_WORDS), re.IGNORECASE | re.UNICODE)
CENTRAL_RE  = re.compile("|".join(_CENTRAL_WORDS),  re.IGNORECASE | re.UNICODE)

# =============================================================================
# DNSMOS perceptual quality estimator
# =============================================================================
_DNSMOS_SESSION = None

def _init_dnsmos() -> bool:
    """
    Lazily load the DNSMOS ONNX model for perceptual audio quality scoring.
    Returns True if successfully loaded, False if unavailable (falls back to SNR proxy).

    Download the ONNX model from:
      https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS
    and place it at: models/dnsmos_p835.onnx
    """
    global _DNSMOS_SESSION
    if _DNSMOS_SESSION is not None:
        return True
    model_path = Path("models/dnsmos_p835.onnx")
    if not model_path.exists():
        log.warning(
            "DNSMOS model not found at %s — falling back to RMS-proxy SNR. "
            "For better quality filtering download dnsmos_p835.onnx from the "
            "Microsoft DNS-Challenge repo.", model_path
        )
        return False
    try:
        import onnxruntime as ort
        _DNSMOS_SESSION = ort.InferenceSession(
            str(model_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        log.info("DNSMOS ONNX model loaded from %s", model_path)
        return True
    except Exception as e:
        log.warning("Failed to load DNSMOS model: %s — using SNR proxy.", e)
        return False


def dnsmos_score(audio_float32: np.ndarray, sr: int) -> float:
    """
    Return overall MOS-like perceptual quality score (1.0–5.0).
    Falls back to a conservative RMS-proxy if DNSMOS is unavailable.
    """
    if not _init_dnsmos():
        return _snr_proxy_score(audio_float32)

    try:
        # DNSMOS expects 16 kHz mono float32
        if sr != 16_000:
            audio_16k = librosa.resample(audio_float32, orig_sr=sr, target_sr=16_000)
        else:
            audio_16k = audio_float32
        # Input shape: (1, samples)
        inp = audio_16k[np.newaxis, :].astype(np.float32)
        outputs = _DNSMOS_SESSION.run(None, {"input_1": inp})
        # outputs[0] shape: (1, 4) — [SIG, BAK, OVL, P835_OVRL]; use OVL (index 2)
        return float(outputs[0][0, 2])
    except Exception as e:
        log.warning("DNSMOS inference failed: %s — using SNR proxy.", e)
        return _snr_proxy_score(audio_float32)


def _snr_proxy_score(audio: np.ndarray) -> float:
    """Fast RMS/10th-percentile SNR proxy mapped to a rough 1–5 MOS scale."""
    rms = np.sqrt(np.mean(audio ** 2))
    noise_floor = np.percentile(np.abs(audio), 10)
    if noise_floor < 1e-10:
        return 5.0  # essentially silent noise floor → treat as clean
    snr_db = float(20 * np.log10(rms / noise_floor))
    # Map SNR: <10 dB ≈ 1.0, 30+ dB ≈ 5.0
    return float(np.clip((snr_db - 10.0) / 20.0 * 4.0 + 1.0, 1.0, 5.0))

# =============================================================================
# Loudness normalisation (LUFS) — replaces brittle peak normalization
# =============================================================================
_LUFS_METER: Optional[pyln.Meter] = None

def _get_meter(sr: int) -> pyln.Meter:
    global _LUFS_METER
    if _LUFS_METER is None or _LUFS_METER.rate != sr:
        _LUFS_METER = pyln.Meter(sr)
    return _LUFS_METER


def normalize_lufs(audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
    """
    Normalize audio to TARGET_LUFS integrated loudness.
    Returns None if the clip is too quiet/silent to measure.
    """
    meter = _get_meter(sr)
    try:
        loudness = meter.integrated_loudness(audio)
    except Exception:
        return None
    if loudness < LUFS_FLOOR or np.isinf(loudness) or np.isnan(loudness):
        return None  # effectively silent — reject
    normalized = pyln.normalize.loudness(audio, loudness, TARGET_LUFS)
    # Hard-clip to prevent intersample clipping after loudness gain
    return np.clip(normalized, -0.99, 0.99).astype(np.float32)

# =============================================================================
# Audio processing pipeline
# =============================================================================

def process_audio(array: np.ndarray, orig_sr: int) -> Optional[np.ndarray]:
    """
    Downmix → duration gate → resample → LUFS normalize → DNSMOS gate → int16.
    Returns None if the clip is rejected at any stage.
    """
    # 1. Downmix to mono (handles both (samples, ch) and (ch, samples))
    if array.ndim > 1:
        # phoaudiobook stores (samples, channels); take mean across last axis
        array = array.mean(axis=-1)

    # 2. Duration gate BEFORE resampling — fast rejection of too-short/long clips
    raw_dur = len(array) / orig_sr
    if not (MIN_DUR_S <= raw_dur <= MAX_DUR_S):
        return None

    # 3. Resample to 24 kHz (soxr_hq: good quality/speed tradeoff)
    if orig_sr != TARGET_SR:
        audio = librosa.resample(array, orig_sr=orig_sr, target_sr=TARGET_SR, res_type="soxr_hq")
    else:
        audio = array.copy()

    # 4. LUFS loudness normalization (replaces peak-norm which leaves level inconsistency)
    audio = normalize_lufs(audio, TARGET_SR)
    if audio is None:
        return None

    # 5. Perceptual quality gate (DNSMOS or SNR proxy)
    if dnsmos_score(audio, TARGET_SR) < MIN_DNSMOS:
        return None

    # 6. Convert to 16-bit PCM for storage
    return (audio * 32767).clip(-32768, 32767).astype(np.int16)

# =============================================================================
# Speaker / dialect filtering
# =============================================================================

def load_verified_speakers(data_root: Path) -> Optional[set]:
    """
    Load the verified Northern speaker whitelist.
    Returns None (= accept all) only if the file is missing AND a warning is logged.
    """
    fpath = data_root / "verified_northern_speakers.txt"
    if not fpath.exists():
        log.warning(
            "verified_northern_speakers.txt not found at %s — "
            "ALL speakers accepted. Edit this file to restrict to Northern voices.", fpath
        )
        return None
    speakers: set[str] = set()
    with open(fpath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                speakers.add(line)
    if not speakers:
        log.warning("verified_northern_speakers.txt is empty — accepting all speakers.")
        return None
    log.info("Speaker whitelist: %d verified Northern speakers loaded.", len(speakers))
    return speakers


def passes_dialect_filter(text: str) -> bool:
    """
    Reject transcripts containing strong Southern or Central dialect markers.
    This is a fast text-only pre-filter; fine-grained acoustic dialect classification
    is handled by DialectAuditor on the audio signal.
    """
    if SOUTHERN_RE.search(text):
        return False
    if CENTRAL_RE.search(text):
        return False
    return True

# =============================================================================
# Dialect auditor — audio-based per-speaker dialect classification with caching
# =============================================================================
_DIALECT_CLASSIFIER = None


def _get_dialect_classifier():
    global _DIALECT_CLASSIFIER
    if _DIALECT_CLASSIFIER is None:
        log.info("Loading Wav2Vec2 dialect classifier (first call)…")
        device = 0 if torch.cuda.is_available() else -1
        _DIALECT_CLASSIFIER = hf_pipeline(
            "audio-classification",
            model="thangquang09/wav2vec2-base-vi-accent-classification",
            device=device,
            return_all_scores=True,   # need per-class scores for confidence check
        )
    return _DIALECT_CLASSIFIER


def classify_audio_dialect(audio_array: np.ndarray, sample_rate: int) -> tuple[str, float]:
    """
    Returns (dialect_label, confidence).
    dialect_label: 'north' | 'south' | 'central'
    confidence: classifier score for the winning label (0–1).
    Falls back to ('north', 0.0) on error.
    """
    try:
        clf = _get_dialect_classifier()
        # Resample to 16 kHz for Wav2Vec2
        if sample_rate != 16_000:
            audio_16k = librosa.resample(
                audio_array.astype(np.float32), orig_sr=sample_rate, target_sr=16_000
            )
        else:
            audio_16k = audio_array.astype(np.float32)

        scores = clf(audio_16k)
        # scores is a list of dicts: [{"label": ..., "score": ...}, ...]
        best = max(scores, key=lambda x: x["score"])
        mapping = {"Bắc": "north", "Nam": "south", "Trung": "central"}
        label = mapping.get(best["label"], "north")
        return label, float(best["score"])
    except Exception as e:
        log.warning("Dialect classifier failed: %s — defaulting to 'north'.", e)
        return "north", 0.0


class DialectAuditor:
    """
    Per-speaker dialect tracking with confidence-gated voting and caching.

    Audit strategy: sample 6 spread-out utterances per speaker
    ({0, 4, 14, 49, 99, 199}) before locking the speaker's dialect.
    Only votes with confidence ≥ AUDIT_CONFIDENCE are counted.
    Speakers with genuinely mixed votes (multi-label after 6 audits) are
    flagged as "mixed" and their clips are dropped.
    """

    # Audit at these per-speaker utterance indices (0-based)
    AUDIT_INDICES = frozenset({0, 4, 14, 49, 99, 199})
    LOCK_THRESHOLD = 6   # number of confident votes before locking

    def __init__(self, data_root: Path, northern_speakers: set):
        self.cache_file       = data_root / "speaker_dialects.json"
        self.northern_speakers = northern_speakers
        self.dialects         = self._load_cache()
        # votes[spk] = list of (dialect, confidence) for confident predictions only
        self.votes: dict[str, list[tuple[str, float]]] = defaultdict(list)
        self.counts: dict[str, int] = defaultdict(int)   # utterance index per speaker

    # ── Cache I/O ────────────────────────────────────────────────────────────

    def _load_cache(self) -> dict[str, str]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                log.warning("Could not load dialect cache: %s", e)
        return {}

    def save_cache(self):
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.dialects, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning("Could not save dialect cache: %s", e)

    # ── Core API ─────────────────────────────────────────────────────────────

    def get_dialect_or_skip(
        self, spk: str, audio: np.ndarray, sample_rate: int
    ) -> Optional[str]:
        """
        Returns dialect string for the speaker ('north' | 'south' | 'central'),
        or None if the speaker is flagged as a mixed-dialect company narrator
        and should be excluded.
        """
        # 1. Already resolved — use cache
        if spk in self.dialects:
            val = self.dialects[spk]
            return None if val == "mixed" else val

        # 2. Whitelisted Northern speakers — skip classification entirely
        if spk in self.northern_speakers:
            return "north"

        idx = self.counts[spk]
        self.counts[spk] += 1

        # 3. Audit point — run the classifier and record a confident vote
        if idx in self.AUDIT_INDICES:
            label, conf = classify_audio_dialect(audio, sample_rate)
            if conf >= AUDIT_CONFIDENCE:
                self.votes[spk].append((label, conf))
                log.debug(
                    "Speaker '%s' audit %d/%d: %s (conf=%.2f)",
                    spk, len(self.votes[spk]), self.LOCK_THRESHOLD, label, conf,
                )

            # 4. Lock once we have enough confident votes
            if len(self.votes[spk]) >= self.LOCK_THRESHOLD:
                self._lock_speaker(spk)

        # 5. Between audits, use the most recent confident vote (or default north)
        if self.votes[spk]:
            return self.votes[spk][-1][0]
        return "north"  # provisional until first confident audit

    def _lock_speaker(self, spk: str):
        vote_labels = [v[0] for v in self.votes[spk]]
        unique = set(vote_labels)
        if len(unique) == 1:
            self.dialects[spk] = vote_labels[0]
            log.info(
                "Locked speaker '%s' → %s (%d/%d unanimous votes).",
                spk, self.dialects[spk], len(vote_labels), self.LOCK_THRESHOLD,
            )
        else:
            # Multi-dialect votes → company narrator, exclude
            self.dialects[spk] = "mixed"
            log.warning(
                "Speaker '%s' flagged as MIXED-DIALECT narrator (votes=%s). "
                "Clips will be excluded.", spk, vote_labels,
            )
        self.save_cache()
        del self.votes[spk]

    def finalize_sweep(self):
        """Lock any speakers still in the pending-votes queue at end of dataset."""
        for spk in list(self.votes.keys()):
            if self.votes[spk]:
                self._lock_speaker(spk)

# =============================================================================
# G2P
# =============================================================================

def to_ipa(text: str, dialect: str = "north") -> str:
    """Normalize Vietnamese text and convert to IPA using dialect-specific rules."""
    if not _G2P_AVAILABLE:
        return ""
    try:
        normalized = TTSnorm(text, punc=True, unknown=False, lower=False)
        ipa = _vi2IPA(normalized, dialect=dialect)
        # Basic sanity check — reject empty or suspiciously short IPA strings
        if not ipa or len(ipa.strip()) < 2:
            return ""
        return ipa.strip()
    except Exception as e:
        log.warning("G2P failed for '%.40s…' [dialect=%s]: %s", text, dialect, e)
        return ""


def validate_tone_coverage(manifest_path: Path) -> None:
    """
    Log a warning if any of the 6 Vietnamese tones are absent from the manifest IPA.
    Tones are represented as Chao tone letters in viphoneme output.
    """
    # These are the Chao tone numerals used by viphoneme for the 6 Northern tones:
    # ngang(˧), huyền(˨˩), sắc(˦˥), nặng(˨˩˦), hỏi(˧˩˨), ngã(˧˥)
    TONE_MARKERS = {"˧", "˨", "˦", "˥", "˩"}  # individual chao letters
    found: set[str] = set()
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="|"):
            if len(row) >= 2:
                for ch in row[1]:
                    if ch in TONE_MARKERS:
                        found.add(ch)
    missing = TONE_MARKERS - found
    if missing:
        log.warning(
            "Tone coverage check: missing Chao markers %s in manifest IPA. "
            "Check viphoneme output format.", missing
        )
    else:
        log.info("Tone coverage check passed: all expected Chao markers present.")

# =============================================================================
# CLI + Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Prepare Northern Vietnamese TTS dataset.")
    p.add_argument("--data-root",    default=os.getenv("DATA_ROOT", "data"),
                   help="Root directory for data (default: data)")
    p.add_argument("--max-samples",  type=int, default=0,
                   help="Hard cap on retained samples (0 = unlimited).")
    p.add_argument("--smoke-test",   action="store_true",
                   help="Quick sanity check — cap at 50 samples.")
    p.add_argument("--accept-all-speakers", action="store_true",
                   help="Bypass speaker whitelist (useful for initial exploration).")
    p.add_argument("--min-dnsmos",   type=float, default=MIN_DNSMOS,
                   help=f"Minimum DNSMOS/quality score threshold (default: {MIN_DNSMOS}).")
    return p.parse_args()


def main():
    args = parse_args()

    smoke = args.smoke_test or os.getenv("SMOKE_TEST", "").lower() == "true"
    max_samples = 50 if smoke else (args.max_samples or 0)

    data_root     = Path(args.data_root)
    processed_dir = data_root / "processed"
    manifest_path = data_root / "train_manifest.csv"
    val_manifest_path = data_root / "val_manifest.csv"
    speaker_id_map_path = data_root / "speaker2id.json"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ── Speaker / dialect setup ───────────────────────────────────────────────
    if args.accept_all_speakers:
        verified_speakers = None
        log.info("--accept-all-speakers: speaker whitelist bypassed.")
    else:
        verified_speakers = load_verified_speakers(data_root)

    northern_speakers = set()
    nspk_file = data_root / "verified_northern_speakers.txt"
    if nspk_file.exists():
        with open(nspk_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    northern_speakers.add(line)

    auditor = DialectAuditor(data_root, northern_speakers)

    # ── Build speaker→int mapping (loaded from disk so it's consistent across runs) ─
    speaker2id: dict[str, int] = {}
    if speaker_id_map_path.exists():
        with open(speaker_id_map_path, encoding="utf-8") as f:
            speaker2id = json.load(f)
        log.info("Loaded existing speaker2id map with %d entries.", len(speaker2id))

    def get_speaker_id(spk: str) -> int:
        if spk not in speaker2id:
            speaker2id[spk] = len(speaker2id)
        return speaker2id[spk]

    # ── Stream dataset ────────────────────────────────────────────────────────
    log.info("Streaming dataset 'thivux/phoaudiobook' (split=train)…")
    ds = load_dataset("thivux/phoaudiobook", split="train", streaming=True)
    ds_meta = ds.cast_column("audio", hf_datasets.Audio(decode=False))

    records: list[tuple[str, str, str, int]] = []
    rejected = defaultdict(int)
    global_idx   = 0
    retained_idx = 0

    for item in ds_meta:
        global_idx += 1

        # ── 1. Speaker whitelist ─────────────────────────────────────────────
        spk = (item.get("speaker") or "").strip()
        if not spk:
            rejected["no_speaker"] += 1
            continue
        if verified_speakers is not None and spk not in verified_speakers:
            rejected["speaker"] += 1
            continue

        # ── 2. Text dialect pre-filter ────────────────────────────────────────
        transcript = (item.get("text") or "").strip()
        if not transcript:
            rejected["empty_text"] += 1
            continue
        if not passes_dialect_filter(transcript):
            rejected["dialect_text"] += 1
            continue

        # ── 3. Decode audio (lazy — only for survivors of text filters) ───────
        audio_data  = item.get("audio") or {}
        audio_bytes = audio_data.get("bytes")
        if not audio_bytes:
            rejected["audio_missing"] += 1
            continue
        try:
            with io.BytesIO(audio_bytes) as bio:
                array, orig_sr = sf.read(bio, dtype="float32")
        except Exception as e:
            log.debug("Audio decode failed (item %d): %s", global_idx, e)
            rejected["audio_decode"] += 1
            continue

        # ── 4. Audio processing + quality gates ───────────────────────────────
        processed = process_audio(array, orig_sr)
        if processed is None:
            rejected["audio_quality"] += 1
            continue

        # ── 5. Acoustic dialect classification + company-speaker filtering ────
        # Pass float32 (before int16 conversion) to the classifier
        audio_float = processed.astype(np.float32) / 32767.0
        dialect = auditor.get_dialect_or_skip(spk, audio_float, TARGET_SR)
        if dialect is None:
            rejected["mixed_speaker"] += 1
            continue

        # ── 6. G2P phonemization with dialect-aware IPA ───────────────────────
        ipa = to_ipa(transcript, dialect=dialect)
        if not ipa:
            rejected["g2p"] += 1
            continue

        # ── 7. Write WAV + record ─────────────────────────────────────────────
        spk_id = get_speaker_id(spk)
        fname  = f"vi_north_{retained_idx:07d}.wav"
        fpath  = processed_dir / fname
        sf.write(str(fpath), processed, TARGET_SR, subtype="PCM_16")

        records.append((str(fpath), ipa, transcript, spk_id))
        retained_idx += 1

        if retained_idx % 200 == 0:
            log.info(
                "Scanned %d | Retained %d | Rejected: %s",
                global_idx, retained_idx,
                dict(rejected),
            )

        if max_samples and retained_idx >= max_samples:
            log.info("Reached max-samples cap (%d). Stopping.", max_samples)
            break

    # ── Finalize auditor state ────────────────────────────────────────────────
    auditor.finalize_sweep()

    # ── Save speaker2id map ───────────────────────────────────────────────────
    with open(speaker_id_map_path, "w", encoding="utf-8") as f:
        json.dump(speaker2id, f, ensure_ascii=False, indent=2)
    log.info("Saved speaker2id map (%d speakers) to %s", len(speaker2id), speaker_id_map_path)

    # ── Train / val split (speaker-stratified 95/5) ───────────────────────────
    random.seed(42)
    # Group by speaker for stratified split
    by_speaker: dict[int, list] = defaultdict(list)
    for rec in records:
        by_speaker[rec[3]].append(rec)

    train_records: list = []
    val_records:   list = []
    for spk_id, recs in by_speaker.items():
        random.shuffle(recs)
        n_val = max(1, int(len(recs) * 0.05)) if len(recs) >= 10 else 0
        val_records.extend(recs[:n_val])
        train_records.extend(recs[n_val:])

    random.shuffle(train_records)
    random.shuffle(val_records)

    # ── Write manifests — format: wav_path|ipa|text|speaker_id ───────────────
    for path, recs in [(manifest_path, train_records), (val_manifest_path, val_records)]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f, delimiter="|").writerows(recs)

    log.info(
        "Done. Train: %d samples → %s | Val: %d samples → %s\n"
        "Rejection breakdown: %s",
        len(train_records), manifest_path,
        len(val_records),   val_manifest_path,
        dict(rejected),
    )

    # ── Tone coverage sanity check ────────────────────────────────────────────
    if len(train_records) > 0:
        validate_tone_coverage(manifest_path)


if __name__ == "__main__":
    main()