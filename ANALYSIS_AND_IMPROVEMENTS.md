# KokoroTTS Vietnamese Fine-Tuning — Deep Analysis & Improvement Plan

> Target: Beat `dangtr0408/StyleTTS2-lite-vi` (trained ~120k steps, batch 3, ~1 A100-day).
> Hardware: Ryzen AI MAX 395+ · 128 GB unified VRAM (ROCm 6.2+).

---

## Executive Summary

The pipeline architecture is solid — dialect-aware G2P, audio quality gating, incremental speaker auditing, and vocabulary surgery are all the right ideas. However there are **4 critical correctness bugs** and **several major quality gaps** that would prevent beating a properly trained StyleTTS2 baseline even with 10× the compute. The current `run_train.py` is training a *proxy mel-regression model*, not Kokoro/StyleTTS2 itself. This is the single most important thing to fix.

---

## CRITICAL BUGS (Fix These First)

### Bug 1 — `run_train.py` is NOT fine-tuning Kokoro
**Severity: Fatal.**

`run_train.py` loads `XPhoneBERT` and trains a custom `ViNorthAcousticModel` (a tiny GRU+conv head) with L1 mel loss. This has nothing to do with the actual Kokoro/StyleTTS2 model architecture. The `kokoro-vi-north-extended.pth` checkpoint produced by `extend_vocab.py` is never loaded into training. The whole stage 4 essentially trains a throwaway regression probe, then discards it.

**Fix:** Wire the extended checkpoint into a real StyleTTS2 fine-tuning loop. The StyleTTS2 training script is available at `https://github.com/yl4579/StyleTTS2` — clone it into `/opt/StyleTTS2` in your Docker image and call its `train_first.py` (Stage 1, acoustic) properly, passing your extended checkpoint as `--pretrained_model`.

```python
# In run_train.py — replace the model block with:
sys.path.insert(0, "/opt/StyleTTS2")
from models import build_model
from utils import load_checkpoint

config_path = "/workspace/config_vi.json"
ckpt_path   = "/workspace/checkpoints/kokoro-vi-north-extended.pth"

with open(config_path) as f:
    config = yaml.safe_load(f)

model = build_model(config["model_params"])
load_checkpoint(model, ckpt_path)  # loads extended weights
```

### Bug 2 — `prepare_dataset.py`: Speaker and dialect filters are silently disabled
**Severity: High.**

`load_verified_speakers()` always returns `None` (line 93) and `passes_dialect_filter()` always returns `True` (line 85). The entire filtering stack — the carefully written `verified_northern_speakers.txt` system and the `SOUTHERN_RE`/`CENTRAL_RE` blocklists — is completely bypassed. Every sample from every speaker goes through regardless of dialect.

This means you are training on Southern and Central Vietnamese audio with Northern IPA labels. That mismatch is catastrophic: the model learns to associate the *wrong acoustic features* with each phoneme.

**Fix (prepare_dataset.py):**
```python
def load_verified_speakers() -> Optional[set]:
    """Load the verified Northern speaker whitelist."""
    if not VERIFIED_SPEAKERS_FILE.exists():
        log.warning("No verified speakers file found at %s — accepting all speakers.", VERIFIED_SPEAKERS_FILE)
        return None
    speakers = set()
    with open(VERIFIED_SPEAKERS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                speakers.add(line)
    log.info("Loaded %d verified Northern speakers.", len(speakers))
    return speakers if speakers else None

def passes_dialect_filter(text: str) -> bool:
    if SOUTHERN_RE.search(text):
        return False
    if CENTRAL_RE.search(text):
        return False
    return True
```

### Bug 3 — `extend_vocab.py`: Reads phoneme *characters*, not phoneme *tokens*
**Severity: High.**

`vi_symbols.update(list(row[1]))` iterates over individual Unicode characters in the IPA string. Vietnamese IPA has multi-character phonemes (e.g., `kʷ`, `ŋ͡m`, tonal diacritics stacked on vowels). Splitting character-by-character produces a broken symbol inventory — you may add 80 single characters while missing 20 actual phoneme tokens.

**Fix:** Use `viphoneme`'s known symbol set directly, or split on whitespace if your G2P produces space-separated tokens:
```python
# Option A — use viphoneme's declared symbol inventory (preferred)
from viphoneme import PHONE_SET  # check the actual export name
vi_symbols = set(PHONE_SET)

# Option B — if manifest stores space-separated IPA tokens:
vi_symbols.update(row[1].split())  # instead of list(row[1])
```

### Bug 4 — Mel spectrogram parameters mismatch StyleTTS2
**Severity: High.**

`compute_mel_spectrogram` in `run_train.py` uses `n_fft=1024, hop_length=256, n_mels=80, fmax=8000`. Kokoro/StyleTTS2 uses `n_fft=2048, hop_length=300, n_mels=80, fmax=None` (full bandwidth). Mismatched Mel parameters mean the acoustic targets your model is being trained toward are systematically different from what Kokoro was pre-trained to produce.

**Fix:** Use StyleTTS2's exact `meldataset.py` mel extractor from the repo, or match parameters exactly:
```python
S = librosa.feature.melspectrogram(
    y=audio, sr=sr,
    n_fft=2048, hop_length=300, win_length=1200,
    n_mels=80, fmin=0, fmax=None, power=1.0
)
```

---

## MAJOR QUALITY IMPROVEMENTS

### 1. Data Scale & Quality Ceiling

The referenced model trained ~120k steps at batch size 3 ≈ ~360k utterance-steps. With your 128 GB unified VRAM you can run **batch size 32–64** on StyleTTS2 Stage 1. To outperform it you need both more data and more training budget.

**Recommended targets:**
- Minimum 200h of verified Northern speech for Stage 1 acoustic pre-training.
- 50h of your best-quality multi-speaker Northern data for Stage 2 adversarial.
- 10–20h of a single target speaker for Stage 3 style fine-tuning (if targeting a specific voice).

`phoaudiobook` has ~1,000h total. Even after Northern filtering you should have 200–400h available.

### 2. SNR Estimator is Too Weak

Your SNR estimator (`rms / 10th-percentile`) is a crude proxy that passes a lot of noisy audiobook recordings. Books are often recorded in non-ideal conditions with room echo, mic noise, and breath artifacts.

**Recommended: Silero VAD + DNSMOS**
```python
# DNSMOS gives a perceptual quality score (1–5) without a clean reference
# pip install onnxruntime
# Model: https://github.com/microsoft/DNS-Challenge (DNSMOS P.835)

def dnsmos_score(audio_16k: np.ndarray) -> float:
    """Returns overall perceptual quality score (aim for > 3.5)."""
    ...

MIN_DNSMOS = 3.5  # reject anything below this
```

Also add **Silero VAD** to reject clips with >15% silence or with speech segments shorter than 0.8s:
```python
# pip install silero-vad
from silero_vad import load_silero_vad, get_speech_timestamps
```

### 3. G2P Dialect Tagging — Improve Confidence

The `DialectAuditor` audits at indices `{0, 9, 29, 69, 149}` — only 5 samples before locking. For audiobooks with inconsistent narrators (e.g., books voiced by voice actors who switch between tones), 5 samples may not be enough.

**Improvements:**
- Increase `audit_indices` to `{0, 4, 14, 49, 99, 199}` (6 checks across 200 samples).
- Add a confidence threshold: only lock if classifier confidence ≥ 0.85 on all votes (use `res[0]["score"]` from the pipeline).
- Add a "provisional" phase: stream with `dialect=north` for first 5 samples, then retroactively re-G2P if locked as `south`/`central` and skip those samples.

### 4. Audio Normalization — Use Loudness (LUFS), Not Peak

Peak normalization (`audio / max_val * 0.95`) leaves loudness inconsistency across clips. Kokoro's training data is loudness-normalized to **-23 LUFS** (broadcast standard). Inconsistent loudness confuses the style encoder.

```python
import pyloudnorm as pyln
meter = pyln.Meter(TARGET_SR)  # create once

def normalize_loudness(audio: np.ndarray, sr: int, target_lufs=-23.0) -> np.ndarray:
    loudness = meter.integrated_loudness(audio)
    if np.isinf(loudness) or loudness < -70:
        return None  # reject silent/broken clips
    normalized = pyln.normalize.loudness(audio, loudness, target_lufs)
    # Hard clip to prevent intersample clipping
    return np.clip(normalized, -0.99, 0.99)
```

### 5. ROCm / AMD GPU — Critical Environment Variables

Your Dockerfile should export these before PyTorch initialization. Missing them can cause 30–50% throughput loss on Strix Halo:

```dockerfile
ENV HSA_OVERRIDE_GFX_VERSION=11.5.0
ENV HSA_ENABLE_SDMA=0
ENV PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
ENV ROCR_VISIBLE_DEVICES=0
# For unified memory efficiency on Strix Halo (iGPU shares system RAM):
ENV HIP_VISIBLE_DEVICES=0
ENV GPU_MAX_ALLOC_PERCENT=100
ENV GPU_MAX_HEAP_SIZE=100
```

Also: use `torch.set_float32_matmul_precision("high")` at the top of `run_train.py` — this enables TF32 on ROCm and gives ~20% throughput boost with no quality loss.

### 6. Training Stage Sequencing

StyleTTS2 has 3 training stages. The referenced model likely only ran Stage 1. To beat it:

| Stage | What trains | Steps (recommend) | LR |
|-------|-------------|-------------------|----|
| Stage 1 | Acoustic decoder + text encoder (no discriminator) | 200k | 1e-4 |
| Stage 2 | Full adversarial (JCU discriminator + SLM loss) | 100k | 2e-5 |
| Stage 3 | Style / duration fine-tune on target speaker | 50k | 5e-6 |

At batch 32 on your hardware, 200k steps ≈ 18–24 hours. This is 4× more training than the reference model.

### 7. Manifest Format — Add Speaker ID Column

Your manifest is `wav_path|ipa_text|raw_text`. StyleTTS2's multi-speaker training expects a speaker ID column:

```
wav_path|ipa_text|raw_text|speaker_id
```

Add speaker ID during `prepare_dataset.py`:
```python
records.append((str(fpath), ipa, transcript, spk))
```

And build a `speaker2id.json` mapping. This enables the style encoder to learn per-speaker embeddings, which is what gives Kokoro's 8 built-in voices their distinct character.

### 8. Tonal Accuracy — Vietnamese-Specific IPA Concern

Vietnamese has 6 tones and the IPA representation of tones varies by G2P library. `viphoneme` encodes tones as diacritic combinations. Verify that your tokenization in `extend_vocab.py` correctly identifies all tonal diacritics as part of their host vowels, not as separate tokens. Incorrectly split tones are the #1 cause of flat/toneless Vietnamese TTS output.

**Validation script to run after G2P:**
```python
# Check IPA coverage of all 6 tones in your manifest
tone_markers = {'˧', '˨˩', '˦˥', '˧˩˨', '˧˥', '˨˩˦'}  # adjust per viphoneme output
found_tones = set()
with open("data/train_manifest.csv") as f:
    for row in csv.reader(f, delimiter="|"):
        for marker in tone_markers:
            if marker in row[1]:
                found_tones.add(marker)
print("Tones covered:", found_tones)
assert found_tones == tone_markers, f"Missing tones: {tone_markers - found_tones}"
```

---

## TRAINING CONFIGURATION IMPROVEMENTS

### `run_train.py` — After fixing the architecture

```python
# Replace the optimizer block with proper differential LR for StyleTTS2:
optimizer_params = [
    # Text encoder (BERT/XPhoneBERT) — very low LR, catastrophic forgetting prevention
    {"params": model.text_encoder.parameters(), "lr": 1e-5},
    # Acoustic decoder — moderate LR
    {"params": model.decoder.parameters(), "lr": 5e-5},
    # Style encoder — standard LR
    {"params": model.style_encoder.parameters(), "lr": 1e-4},
]
optimizer = torch.optim.AdamW(optimizer_params, weight_decay=1e-2, betas=(0.9, 0.98), eps=1e-9)

# Use CosineAnnealingWarmRestarts instead of one-shot cosine for longer training:
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50_000, T_mult=2, eta_min=1e-6)
```

### Batch size on 128 GB Unified VRAM

With 128 GB unified memory (no VRAM/system RAM boundary), you can push:
- Stage 1: batch size **48–64** at 24kHz/80mel
- Stage 2: batch size **24–32** (discriminator overhead)

Set `DataLoader` workers to `min(12, os.cpu_count())` — the 395+ has 16 cores, which can feed the iGPU without bottleneck.

---

## PIPELINE CHECKLIST (Priority Order)

1. **[CRITICAL]** Fix `run_train.py` to actually load and fine-tune the Kokoro model
2. **[CRITICAL]** Re-enable dialect filtering in `prepare_dataset.py`
3. **[CRITICAL]** Fix `extend_vocab.py` to use phoneme tokens, not characters
4. **[CRITICAL]** Fix mel spectrogram parameters to match StyleTTS2's spec
5. **[HIGH]** Switch from peak to LUFS loudness normalization
6. **[HIGH]** Add DNSMOS quality filtering (target > 3.5)
7. **[HIGH]** Add speaker ID column to manifest for multi-speaker training
8. **[HIGH]** Add ROCm environment variables to Dockerfile
9. **[MEDIUM]** Increase dialect auditor confidence threshold and sample count
10. **[MEDIUM]** Validate Vietnamese tone coverage in IPA output
11. **[MEDIUM]** Add `torch.set_float32_matmul_precision("high")` for ROCm throughput
12. **[LOW]** Extend training to 200k+ Stage 1 steps, then run Stage 2 + Stage 3

---

## Expected Outcome

With the critical bugs fixed and the quality improvements applied, your training on 128 GB unified memory should achieve:
- **4–6× more training compute** than the reference model (200k steps vs 120k, batch 32+ vs 3)
- **Better data quality** (LUFS normalization, DNSMOS filtering, correctly tagged dialect IPA)
- **Proper multi-speaker embeddings** (style encoder trained with speaker IDs)
- **MOS score** competitive with SparkTTS (target > 4.0 for Northern dialect)

The reference model's author estimated "~$1000 of compute" for Kokoro's original training. With your hardware, the equivalent runs locally for free — the bottleneck is now data quality and pipeline correctness, not compute.
