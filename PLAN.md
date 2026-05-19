# Northern Vietnamese KokoroTTS Fine-Tuning Execution Plan

**Target:** Giọng Hà Nội — High-Purity Northern Vietnamese Voice  
**Architecture:** Kokoro-82M (StyleTTS2 + iSTFTNet) with XPhoneBERT multilingual encoder  
**Compute:** Ryzen 7 9700X (prep/verify) → Ryzen AI MAX 395+ Strix Halo (full train)  
**License target:** All source data CC-BY-4.0 or CC-0 cleared for commercial redistribution

---

## Architecture Map

```
Raw Audio Corpus
       │
       ▼
[Phase 1] High-Purity Northern Filter
       │  · Speaker metadata → accent field & blocklist lexicon
       │  · SNR gate (≥ 25 dB), duration gate (1.0 – 15.0 s)
       ▼
[Phase 2] Text Normalization (vinorm) → G2P (viphoneme) → IPA tokens
       │  · Northern consonant rules applied (d/gi/r → /z/)
       │  · 6-tone suffix encoding (1–6)
       ▼
[Phase 3] Model Weight Surgery
       │  · Load kokoro-v1.0-zh.pth (best multilingual base)
       │  · Swap PL-BERT → XPhoneBERT (multilingual, covers vi)
       │  · Extend embedding dict to cover Vietnamese IPA set
       ▼
[Phase 4] Workstation Smoke Test (9700X, CPU)
       │  · 32-sample micro-batch, 20 steps
       │  · Assert loss ↓ and audio output is non-silent
       ▼
[Phase 5] Full Training — Rack (Ryzen AI MAX 395+, ROCm 6.2+)
       │  · Stage 1: Acoustic pre-training (mel recon, aligner)
       │  · Stage 2: TTS adversarial fine-tuning (GAN + diffusion)
       │  · Stage 3: Style encoder extraction
       ▼
[Phase 6] Northern Style Vector Export + Inference Validation
```

---

## Phase 1: Dataset Acquisition & High-Purity Northern Filtering

### 1.1 Approved Corpora (Priority Order)

| Dataset | HF ID | License | Est. Clean Hours | Notes |
|---|---|---|---|---|
| InfoRe1 25h | `doof-ferb/infore1_25hours` | CC-BY-4.0 | ~24h | Studio Hanoi — **Primary backbone** |
| FPT Open Speech | `doof-ferb/fpt_fosd` | CC-BY-4.0 | ~25h | Read speech, high SNR |
| InfoRe2 Audiobooks | `doof-ferb/infore2_audiobooks` | CC-BY-4.0 | ~20h | Expressive narration — prosody injector |
| Common Voice 17.0 | `mozilla-foundation/common_voice_17_0` | CC-0 | variable | Filter strictly — see §1.2 |

**Data volume target:** ≥ 30 hours clean post-filter (50+ hours preferred for stable GAN convergence).

### 1.2 Filtering Strategy

**Layer 1 — Metadata (Common Voice):**
```python
# CV17 'accent' field is free-text; normalize before matching
NORTH_ACCENTS = {"hà nội", "hanoi", "northern", "bắc", "north", "miền bắc"}
keep = accent_raw.lower().strip() in NORTH_ACCENTS or accent_raw == ""
# When accent is empty: fall through to Layer 2 lexical check only
```

**Layer 2 — Lexical Blocklist (all corpora):**
```python
SOUTHERN_BLOCK = {
    "vầy", "bự", "hổng", "xài", "kêu", "nhậu", "mắc cười",
    "mèn ơi", "giùm", "dễ cưng", "hổm", "dzậy", "tui", "mầy",
    "ổng", "bả", "thổng", "dzìa", "nói dzậy"
}
CENTRAL_BLOCK = {
    "chi", "mô", "tê", "răng", "rứa", "nớ", "hỉ", "bây chừ",
    "đọi", "trốc", "eng", "ả", "mần"
}
```

> **Do NOT use regex character presence** (e.g., matching `[đrgi]`) as a Northern proxy — those characters appear in all Vietnamese dialects equally.

**Layer 3 — Audio Quality Gate:**
- SNR ≥ 25 dB (measured via `pyloudnorm` / `speechbrain`)
- Duration: 1.0 s ≤ length ≤ 15.0 s
- Silence ratio < 30% of total clip duration

### 1.3 Audio Standardization
- Resample to exactly **24,000 Hz** (Kokoro's native rate) using `librosa.resample` (kaiser_best)
- Format: **16-bit Mono PCM WAV**
- Output manifest: `data/train_manifest.csv` (LJSpeech-compatible: `filename|text|normalized_text`)

### 1.4 Data Split
```
Total clean → 97% train / 2% validation / 1% held-out test
```
Stratify by speaker to avoid speaker leakage across splits.

---

## Phase 2: Text Normalization & Northern Vietnamese G2P

### 2.1 Toolchain

| Tool | Role | Install |
|---|---|---|
| `vinorm` | Vietnamese text normalization (numbers, dates, abbrevs) | `pip install vinorm` |
| `underthesea` | Word segmentation boundary detection | `pip install underthesea` |
| `viphoneme` | Vietnamese G2P → IPA output | `pip install viphoneme` |
| `phonemizer` | Fallback espeak-ng wrapper | `pip install phonemizer` |

### 2.2 Northern Phoneme Rules Applied in G2P

| Orthography | Northern /IPA/ | Incorrect Southern |
|---|---|---|
| `d`, `gi`, `r` | /z/ | /j/ (Southern) |
| `ch`, `tr` | /tɕ/ | /c/ (Central) |
| `s`, `x` | /s/ | merged (Southern) |
| `v` | /v/ | /j/ (Southern) |
| `Ngã (ã)` tone | high glottalized /˧ˀ˥/ | merged with Hỏi (Southern) |

**The 6 tones indexed as token suffixes:**

| # | Tone | Name | Contour | Suffix |
|---|---|---|---|---|
| 1 | Ngang | Mid Level | /˧˧/ | `1` |
| 2 | Huyền | Low Falling | /˨˩/ | `2` |
| 3 | Sắc | High Rising | /˧˥/ | `3` |
| 4 | Hỏi | Mid-Low Dip | /˧˩˨/ | `4` |
| 5 | Ngã | High Glottalized | /˧ˀ˥/ | `5` |
| 6 | Nặng | Low Glottalized | /˨˩ˀ/ | `6` |

### 2.3 Normalization Pipeline (per sample)
```python
from vinorm import TTSnorm
from viphoneme import vi2IPA

def process_text(raw_text: str) -> str:
    # Step 1: Normalize (numbers, dates, abbreviations)
    normalized = TTSnorm(raw_text, punc=True, unknown=False, lower=False)
    # Step 2: G2P to IPA with Northern dialect flag
    ipa = vi2IPA(normalized, dialect="north")
    return ipa
```

---

## Phase 3: Model Architecture & Weight Surgery

### 3.1 Base Checkpoint Selection

> **Use `kokoro-v1.0-zh.pth`** (Mandarin-included checkpoint), NOT the English-only base.  
> The zh checkpoint already has CJK+tone-aware embedding slots that are structurally closer to Vietnamese tonal phonology.

Download: `https://huggingface.co/hexgrad/Kokoro-82M`

### 3.2 Critical Component Replacements

**Problem:** Default Kokoro uses English-trained PL-BERT. It will fail on Vietnamese phonemes.  
**Solution:** Replace with `XPhoneBERT` — pre-trained on 100+ languages including Vietnamese.

```python
# In models.py — swap the BERT encoder
from transformers import AutoModel, AutoTokenizer

# Replace: bert = PLBERT(...)
bert = AutoModel.from_pretrained("vinai/xphonebert-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/xphonebert-base")
```

**Text Aligner:** The ASR-based aligner in StyleTTS2 supports English/Japanese/Chinese. For Vietnamese, use the aligner in **train-only mode** initialized from scratch on the Vietnamese dataset (included in Stage 1 training below).

### 3.3 Vocabulary Extension
```python
# Target: extend from base vocab to cover all Vietnamese IPA glyphs + tone suffixes
# Full Vietnamese IPA set from viphoneme output: approximately 120–150 unique tokens
# Add: ă, ơ, ư, ê, ô, tonal diacritics as discrete tokens

new_vocab_size = len(existing_symbols) + len(vi_ipa_symbols)  # typically ~170–195 total
# Initialize new embeddings as mean of existing weights (no cold-start instability)
new_embed = nn.Embedding(new_vocab_size, embed_dim)
new_embed.weight.data[:old_vocab_size] = old_embed.weight.data
new_embed.weight.data[old_vocab_size:] = old_embed.weight.data.mean(0)
```

Save as: `kokoro-vi-north-extended.pth` + updated `config_vi.json`

---

## Phase 4: Workstation Smoke Test (Ryzen 7 9700X)

### 4.1 Environment Setup
```bash
conda create -n kokoro-vi python=3.10
conda activate kokoro-vi
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install vinorm underthesea viphoneme soundfile librosa datasets transformers tqdm pyloudnorm
```

### 4.2 Verification Checklist

```bash
# [CHECK 1] Audio format compliance
python -c "
import soundfile as sf, glob
errors = []
for f in glob.glob('data/processed/*.wav'):
    info = sf.info(f)
    if info.samplerate != 24000 or info.channels != 1:
        errors.append(f)
print(f'Format errors: {len(errors)}')
"

# [CHECK 2] Manifest integrity
python -c "
import csv, os
with open('data/train_manifest.csv') as f:
    rows = list(csv.reader(f, delimiter='|'))
missing = [r[0] for r in rows if not os.path.exists(r[0])]
empty_phonemes = [r[1] for r in rows if not r[2].strip()]
print(f'Missing files: {len(missing)}, Empty phonemes: {len(empty_phonemes)}')
"

# [CHECK 3] Micro-batch overfitting test (CPU, 20 steps)
python train.py \
    --config config_vi.json \
    --data_path data/train_manifest.csv \
    --batch_size 4 \
    --num_steps 20 \
    --device cpu \
    --smoke_test

# Expected: loss should decrease from step 1 → 20 (not stuck or NaN)
```

---

## Phase 5: Full Training on Rack (Ryzen AI MAX 395+)

### 5.1 Hardware Configuration

| Parameter | Value |
|---|---|
| iGPU | RDNA4 40 CU (integrated, Strix Halo) |
| VRAM model | Unified LPDDR5x shared pool |
| GTT allocation | 96 GB (`amdgpu.gttsize=98304`) |
| ROCm version | 6.2.x |
| `HSA_OVERRIDE_GFX_VERSION` | `11.5.0` (required for RDNA4 iGPU detection) |
| Docker shm | 32 GB (`--shm-size=32g`) |

> **Set `amdgpu.gttsize=98304` in your kernel boot parameters** (GRUB/systemd-boot), not at runtime, for reliability.  
> Add to `/etc/default/grub`: `GRUB_CMDLINE_LINUX_DEFAULT="... amdgpu.gttsize=98304"`

### 5.2 Training Stages (StyleTTS2 protocol)

**Stage 1 — Acoustic Pre-training (~100k steps)**
- Trains: Mel decoder, text aligner, duration predictor
- Loss: mel reconstruction + CTC alignment
- LR: `1e-4`, batch size: `16–24` (tune to fill VRAM)

**Stage 2 — Adversarial TTS Fine-tuning (~300k steps)**
- Trains: Full model + discriminators (MPD + MSD) + style diffusion
- Loss: GAN + feature matching + style diffusion
- LR: `2e-5` (AdamW, weight_decay=`0.01`)
- AMP: `torch.bfloat16` preferred over `float16` on ROCm (more stable)
- **Enable scheduled sampling** after step 50k to prevent exposure bias

**Stage 3 — Style Encoder Fine-tuning (~50k steps)**
- Freeze acoustic modules; only train style encoder
- Reference: 10–30 second Northern Vietnamese reference clips

### 5.3 Checkpoint Strategy
```python
# Save every 5000 steps; keep last 5 checkpoints
# Resume: python train.py --resume checkpoints/step_XXXXX.pth
```

### 5.4 Monitoring
- Use `wandb` (`pip install wandb`) — log: mel loss, duration loss, GAN loss per step
- Alert threshold: if mel loss does not decrease by step 5000, **stop and debug aligner**

---

## Phase 6: Style Vector Export & Inference Validation

### 6.1 Northern Style Vector Extraction
```python
import torch, numpy as np
model.eval()
# Use a 10–30s clean Hanoi female/male reference clip
ref_audio, sr = librosa.load("reference_hanoi.wav", sr=24000)
with torch.no_grad():
    style_vec = model.style_encoder(torch.FloatTensor(ref_audio).unsqueeze(0))
np.save("style_vi_north.npy", style_vec.cpu().numpy())
print("Style vector shape:", style_vec.shape)  # expected: (1, 128)
```

### 6.2 Inference Smoke Test
```python
# Test 5 representative Northern sentences
test_sentences = [
    "Hà Nội mùa thu, cây cơm nguội vàng.",
    "Anh ấy đang làm việc ở Hà Nội.",
    "Bầu trời hôm nay thật trong xanh.",
    "Chúng tôi đi ăn phở buổi sáng.",
    "Cô giáo dạy học rất tận tâm.",
]
for sent in test_sentences:
    audio = model.inference(sent, style_vec=np.load("style_vi_north.npy"))
    # Save and verify: no clicks, correct tones, Northern /z/ for d/r/gi
    sf.write(f"test_{hash(sent)}.wav", audio, 24000)
```

### 6.3 Acceptance Criteria
- [ ] No NaN/silence in inference output
- [ ] `d`, `gi`, `r` sounds as /z/ (not /j/)
- [ ] Ngã tone (ã) is distinct from Hỏi (ả) 
- [ ] MOS score ≥ 3.5 on blind listening test (5 raters)

---

## Quick Reference: Key Decisions

| Decision | Rationale |
|---|---|
| XPhoneBERT over PL-BERT | PL-BERT is English-only; XPhoneBERT natively covers Vietnamese |
| zh checkpoint as base | Tone-aware architecture better suited than English base |
| `bfloat16` over `float16` | More numerically stable on ROCm; avoids GAN underflow |
| viphoneme over espeak-ng | espeak-ng vi G2P is poor quality; viphoneme is dialect-aware |
| vinorm for normalization | Handles Vietnamese-specific numbers/dates/abbreviations |
| Lexical blocklist + accent metadata | Dual-layer filter is far more reliable than any single heuristic |
| Stage-by-stage training | Follows validated StyleTTS2 protocol; skipping stages causes mode collapse |
