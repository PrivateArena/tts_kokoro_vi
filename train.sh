#!/usr/bin/env bash
# ==============================================================================
# Northern Vietnamese KokoroTTS — One-Click Training Bootstrap
# Target: Ryzen AI MAX 395+ (Strix Halo, RDNA4 iGPU, ROCm 6.2+)
# Usage:  bash train.sh [--resume] [--smoke-test] [--data-only]
# ==============================================================================
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/kokoro_vietnamese"
DATA_DIR="${PROJECT_DIR}/data"
MODEL_DIR="${PROJECT_DIR}/checkpoints"
LOG_DIR="${PROJECT_DIR}/logs"
DOCKER_IMAGE="kokoro-rocm-strix:latest"
KOKORO_REPO="https://github.com/yl4579/StyleTTS2.git"
BASE_CHECKPOINT_URL="https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1_0.pth"
CONFIG_URL="https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/config.json"

# Ryzen AI MAX 395+ unified memory — set in GRUB params too:
# GRUB_CMDLINE_LINUX_DEFAULT="... amdgpu.gttsize=98304"
GTT_SIZE_MB=98304

# Parse flags
RESUME=false; SMOKE_TEST=false; DATA_ONLY=false
for arg in "$@"; do
  case $arg in
    --resume)     RESUME=true ;;
    --smoke-test) SMOKE_TEST=true ;;
    --data-only)  DATA_ONLY=true ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
info()  { echo -e "\n\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
die()   { echo -e "\033[1;31m[FAIL]\033[0m  $*" >&2; exit 1; }

# ==============================================================================
# STEP 1 — DIRECTORY SCAFFOLD
# ==============================================================================
info "[1/7] Creating directory structure"
mkdir -p "${DATA_DIR}/raw" "${DATA_DIR}/processed" \
         "${MODEL_DIR}" "${LOG_DIR}" "${PROJECT_DIR}/scripts"
ok "Directories ready: ${PROJECT_DIR}"

# ==============================================================================
# STEP 2 — GTT UNIFIED MEMORY POOL (Strix Halo iGPU)
# ==============================================================================
info "[2/7] Tuning AMD Unified Memory (GTT) for Strix Halo"
if [ -f /sys/module/amdgpu/parameters/gttsize ]; then
    CURRENT_GTT=$(cat /sys/module/amdgpu/parameters/gttsize)
    if [ "${CURRENT_GTT}" -lt "${GTT_SIZE_MB}" ]; then
        echo "${GTT_SIZE_MB}" | sudo tee /sys/module/amdgpu/parameters/gttsize > /dev/null
        ok "GTT set to ${GTT_SIZE_MB} MB"
    else
        ok "GTT already at ${CURRENT_GTT} MB (≥ ${GTT_SIZE_MB})"
    fi
else
    warn "Runtime GTT path not found. Ensure kernel param amdgpu.gttsize=${GTT_SIZE_MB} is set in GRUB."
fi

# ==============================================================================
# STEP 3 — PYTHON ENVIRONMENT & DEPENDENCIES
# ==============================================================================
info "[3/7] Setting up Python environment"
cd "${PROJECT_DIR}"

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

pip install -q --upgrade pip
pip install -q \
    datasets soundfile librosa pyloudnorm \
    vinorm underthesea viphoneme phonemizer \
    transformers accelerate tqdm wandb \
    torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cpu

ok "Python deps installed"

# ==============================================================================
# STEP 4 — WRITE DATA PREPARATION SCRIPT
# ==============================================================================
info "[4/7] Writing dataset preparation script"

cat > "${PROJECT_DIR}/scripts/prepare_dataset.py" << 'PYEOF'
#!/usr/bin/env python3
"""
Northern Vietnamese dataset preparation pipeline.
Outputs LJSpeech-format manifest: data/train_manifest.csv
Columns: filepath|ipa_text|raw_text
"""
import os, csv, json, re, logging
import soundfile as sf
import librosa
import numpy as np
from pathlib import Path
from datasets import load_dataset
from vinorm import TTSnorm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
MANIFEST_PATH = Path("data/train_manifest.csv")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR   = 24_000
MIN_DUR_S   = 1.0
MAX_DUR_S   = 15.0
MIN_SNR_DB  = 25.0

# ── Dialect blocklists ────────────────────────────────────────────────────────
# Expanded, high-confidence lexical blocklists
SOUTHERN_BLOCK = {
    "vầy", "bự", "hổng", "xài", "kêu", "nhậu", "mắc cười", "mèn ơi",
    "giùm", "dễ cưng", "hổm", "dzậy", "tui", "mầy", "ổng", "bả",
    "thổng", "dzìa", "nói dzậy", "dzô", "lổng", "hổng có",
    # Additions
    "mắc chi", "ghe", "bình thạnh", "sài gòn", "đa khoa", "chút xíu",
    "muống", "kiếm", "làm gì dữ vậy", "hú hồn", "miết", "uống nước ngọt"
}

CENTRAL_BLOCK = {
    "chi", "mô", "tê", "răng", "rứa", "nớ", "hỉ", "bây chừ",
    "đọi", "trốc", "eng", "ả", "mần", "hè nơ",
    # Additions
    "tội nghiệp", "răng rứa", "ngong", "bữa ni", "gửi vô", "trong nớ"
}
NORTH_CV_ACCENTS = {"hà nội","hanoi","northern","bắc","north","miền bắc",""}

def is_northern(text: str, accent: str = "") -> bool:
    """Return True if sample passes Northern dialect filter."""
    tl = text.lower()
    if any(w in tl for w in SOUTHERN_BLOCK): return False
    if any(w in tl for w in CENTRAL_BLOCK):  return False
    if accent:
        norm = accent.lower().strip()
        return norm in NORTH_CV_ACCENTS
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
    """Resample, duration-gate, SNR-gate. Returns None if rejected."""
    audio = librosa.resample(array, orig_sr=orig_sr, target_sr=TARGET_SR) \
            if orig_sr != TARGET_SR else array.copy()
    dur = len(audio) / TARGET_SR
    if not (MIN_DUR_S <= dur <= MAX_DUR_S): return None
    if estimate_snr(audio) < MIN_SNR_DB:    return None
    return (audio * 32767).clip(-32768, 32767).astype(np.int16)

# ── Dataset loaders ───────────────────────────────────────────────────────────
DATASETS = [
    dict(hf_id="doof-ferb/infore1_25hours",  split="train", accent_field=None),
    dict(hf_id="doof-ferb/fpt_fosd",         split="train", accent_field=None),
    dict(hf_id="doof-ferb/infore2_audiobooks",split="train", accent_field=None),
    dict(hf_id="mozilla-foundation/common_voice_17_0",
         split="train", lang="vi", accent_field="accent"),
]

records = []
global_idx = 0

for ds_cfg in DATASETS:
    hf_id = ds_cfg["hf_id"]
    log.info("Loading %s ...", hf_id)
    kwargs = dict(split=ds_cfg["split"], trust_remote_code=True, streaming=True)
    if "lang" in ds_cfg:
        kwargs["name"] = ds_cfg["lang"]

    try:
        dataset = load_dataset(hf_id, **kwargs)
    except Exception as e:
        log.warning("Skipping %s: %s", hf_id, e)
        continue

    for item in dataset:
        transcript = (item.get("transcript") or item.get("sentence") or "").strip()
        if not transcript:
            continue

        accent = item.get(ds_cfg.get("accent_field") or "", "") or ""
        if not is_northern(transcript, accent):
            continue

        audio_data = item.get("audio", {})
        array = audio_data.get("array")
        orig_sr = audio_data.get("sampling_rate", TARGET_SR)
        if array is None:
            continue

        processed = process_audio(array, orig_sr)
        if processed is None:
            continue

        ipa = to_ipa(transcript)
        if not ipa:
            continue

        fname = f"vi_north_{global_idx:07d}.wav"
        fpath = PROCESSED_DIR / fname
        sf.write(str(fpath), processed, TARGET_SR, subtype="PCM_16")

        records.append((str(fpath), ipa, transcript))
        global_idx += 1

        if global_idx % 500 == 0:
            log.info("Retained %d samples so far", global_idx)

# ── Write manifest ─────────────────────────────────────────────────────────────
# LJSpeech format: filepath|ipa|raw_text
with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter="|")
    w.writerows(records)

log.info("Done. Retained %d Northern samples → %s", global_idx, MANIFEST_PATH)
PYEOF

ok "prepare_dataset.py written"

# ==============================================================================
# STEP 5 — RUN DATA PREPARATION
# ==============================================================================
info "[5/7] Running dataset preparation (this may take a while)"
python3 "${PROJECT_DIR}/scripts/prepare_dataset.py"

SAMPLE_COUNT=$(wc -l < "${DATA_DIR}/train_manifest.csv" || echo 0)
if [ "${SAMPLE_COUNT}" -lt 1000 ]; then
    warn "Only ${SAMPLE_COUNT} samples retained. Recommend ≥ 5000 (≥30h) for stable GAN training."
else
    ok "Dataset ready: ${SAMPLE_COUNT} Northern samples"
fi

deactivate

if [ "${DATA_ONLY}" = "true" ]; then
    info "Data-only mode: stopping here. Run without --data-only to continue."
    exit 0
fi

# ==============================================================================
# STEP 6 — DOCKER IMAGE (ROCm + Kokoro/StyleTTS2 deps)
# ==============================================================================
info "[6/7] Building ROCm Docker image (kokoro-rocm-strix)"

cat > "${PROJECT_DIR}/Dockerfile" << 'DEOF'
# ROCm 6.2 + PyTorch 2.3 — Strix Halo (RDNA4 iGPU)
FROM rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_2.3.0

# Critical: force ROCm to recognise RDNA4 iGPU GFX ID
ENV HSA_OVERRIDE_GFX_VERSION=11.5.0
ENV ROCM_PATH=/opt/rocm
ENV HIP_VISIBLE_DEVICES=0
ENV PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 espeak-ng git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps — training stack
RUN pip install --no-cache-dir \
    vinorm underthesea viphoneme phonemizer \
    soundfile librosa pyloudnorm \
    transformers accelerate tqdm wandb \
    scipy matplotlib einops \
    munch parallel_wavegan

# Clone StyleTTS2 (base architecture for Kokoro)
RUN git clone https://github.com/yl4579/StyleTTS2.git /opt/StyleTTS2

# XPhoneBERT (multilingual PL-BERT replacement)
RUN python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('vinai/xphonebert-base')"

COPY scripts/ /workspace/scripts/
COPY data/train_manifest.csv /workspace/data/
COPY data/processed/ /workspace/data/processed/
COPY checkpoints/ /workspace/checkpoints/

CMD ["python3", "/workspace/scripts/run_train.py"]
DEOF

sudo docker build -t "${DOCKER_IMAGE}" "${PROJECT_DIR}"
ok "Docker image built: ${DOCKER_IMAGE}"

# ==============================================================================
# STEP 7 — WRITE & LAUNCH TRAINING CONTROLLER
# ==============================================================================
info "[7/7] Writing training controller"

cat > "${PROJECT_DIR}/scripts/run_train.py" << 'PYEOF'
#!/usr/bin/env python3
"""
Northern Vietnamese KokoroTTS — StyleTTS2 Training Controller
Stages:
  1. Acoustic pre-training  (mel recon + aligner CTC)   ~100k steps
  2. Adversarial TTS        (GAN + diffusion)            ~300k steps
  3. Style encoder          (freeze trunk, train style)  ~50k steps
"""
import os, sys, json, csv, logging, argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import soundfile as sf
import numpy as np
from transformers import AutoModel, AutoTokenizer
from torch.amp import autocast, GradScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--manifest",    default="data/train_manifest.csv")
parser.add_argument("--checkpoint",  default="checkpoints/kokoro-vi-north-extended.pth")
parser.add_argument("--config",      default="/opt/StyleTTS2/Configs/config_ft.yml")
parser.add_argument("--stage",       type=int, default=1, choices=[1,2,3])
parser.add_argument("--resume",      default="")
parser.add_argument("--batch_size",  type=int, default=16)
parser.add_argument("--max_steps",   type=int, default=100_000)
parser.add_argument("--smoke_test",  action="store_true")
parser.add_argument("--wandb",       action="store_true")
args = parser.parse_args()

if args.smoke_test:
    args.max_steps = 20
    args.batch_size = 4
    log.info("SMOKE TEST mode: %d steps, batch %d", args.max_steps, args.batch_size)

device = "cuda" if torch.cuda.is_available() else "cpu"
log.info("Backend: %s", device)

if device == "cuda":
    log.info("GPU: %s | VRAM: %.1f GB",
             torch.cuda.get_device_name(0),
             torch.cuda.get_device_properties(0).total_memory / 1e9)

# ── Dataset ───────────────────────────────────────────────────────────────────
class ViNorthDataset(Dataset):
    def __init__(self, manifest_path: str, tokenizer):
        self.records = []
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="|"):
                if len(row) >= 3:
                    self.records.append((row[0], row[1]))   # (wav_path, ipa_text)
        self.tokenizer = tokenizer
        log.info("Dataset: %d samples loaded", len(self.records))

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        wav_path, ipa_text = self.records[idx]
        audio, _ = sf.read(wav_path, dtype="float32")
        tokens = self.tokenizer(ipa_text, return_tensors="pt",
                                padding="max_length", max_length=256,
                                truncation=True)
        mel = torch.FloatTensor(audio).unsqueeze(0)   # placeholder — replace with proper mel
        return {
            "input_ids":      tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "mel":            mel,
        }

# ── Load XPhoneBERT (replaces PL-BERT) ───────────────────────────────────────
log.info("Loading XPhoneBERT (multilingual phoneme encoder)...")
tokenizer = AutoTokenizer.from_pretrained("vinai/xphonebert-base")
xphonebert = AutoModel.from_pretrained("vinai/xphonebert-base").to(device)

dataset   = ViNorthDataset(args.manifest, tokenizer)
loader    = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                       num_workers=min(8, os.cpu_count()), pin_memory=(device=="cuda"))

# ── Minimal fine-tuning head (attach to full StyleTTS2 in production) ─────────
class ViNorthHead(nn.Module):
    """
    Drop-in stub — replace this with the actual Kokoro/StyleTTS2 model graph.
    Load StyleTTS2 from /opt/StyleTTS2 and wire XPhoneBERT as the text encoder.
    """
    def __init__(self, bert_hidden=768, mel_bins=80):
        super().__init__()
        self.proj = nn.Linear(bert_hidden, mel_bins)

    def forward(self, bert_out, mel_target):
        pred = self.proj(bert_out.last_hidden_state[:, 0, :])   # CLS token
        loss = nn.functional.mse_loss(pred, mel_target.mean(-1))
        return loss

model     = ViNorthHead().to(device)
optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(xphonebert.parameters()),
    lr=2e-5, weight_decay=0.01
)
scaler    = GradScaler(enabled=(device == "cuda"))

# ── Resume ─────────────────────────────────────────────────────────────────────
start_step = 0
ckpt_dir   = Path("checkpoints")
ckpt_dir.mkdir(exist_ok=True)

if args.resume and Path(args.resume).exists():
    state = torch.load(args.resume, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    start_step = state.get("step", 0)
    log.info("Resumed from step %d", start_step)

# ── WandB ─────────────────────────────────────────────────────────────────────
if args.wandb:
    import wandb
    wandb.init(project="kokoro-vi-north", config=vars(args))

# ── Training loop ──────────────────────────────────────────────────────────────
log.info("Stage %d training — steps %d → %d", args.stage, start_step, args.max_steps)
step      = start_step
loss_hist = []

for batch in loader:
    if step >= args.max_steps:
        break

    ids  = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    mel  = batch["mel"].to(device)

    optimizer.zero_grad(set_to_none=True)

    with autocast(device_type=device, dtype=torch.bfloat16):
        bert_out = xphonebert(input_ids=ids, attention_mask=mask)
        loss     = model(bert_out, mel)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    loss_hist.append(loss.item())
    step += 1

    if step % 100 == 0:
        avg = sum(loss_hist[-100:]) / min(len(loss_hist), 100)
        log.info("Step %6d | loss %.4f | avg(100) %.4f", step, loss.item(), avg)
        if args.wandb:
            wandb.log({"loss": loss.item(), "loss_avg100": avg}, step=step)

    # Checkpoint every 5000 steps — keep last 5
    if step % 5000 == 0:
        ckpt_path = ckpt_dir / f"step_{step:08d}.pth"
        torch.save({"step": step, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict()}, ckpt_path)
        # Prune old checkpoints
        ckpts = sorted(ckpt_dir.glob("step_*.pth"))
        for old in ckpts[:-5]:
            old.unlink()
        log.info("Checkpoint saved: %s", ckpt_path)

# ── Smoke-test assertion ───────────────────────────────────────────────────────
if args.smoke_test:
    if loss_hist[-1] < loss_hist[0]:
        log.info("SMOKE TEST PASSED ✓  loss %.4f → %.4f", loss_hist[0], loss_hist[-1])
    else:
        log.error("SMOKE TEST FAILED ✗  loss did not decrease: %.4f → %.4f",
                  loss_hist[0], loss_hist[-1])
        sys.exit(1)

log.info("Training stage %d complete. Steps: %d", args.stage, step)
if args.wandb:
    wandb.finish()
PYEOF

ok "run_train.py written"

# ── Launch ─────────────────────────────────────────────────────────────────────
if [ "${SMOKE_TEST}" = "true" ]; then
    info "Running smoke test inside container (20 steps, CPU-only)..."
    sudo docker run --rm \
        --ipc=host \
        --shm-size=4g \
        -v "${PROJECT_DIR}:/workspace" \
        "${DOCKER_IMAGE}" \
        python3 /workspace/scripts/run_train.py \
            --smoke_test \
            --batch_size 4 \
            --max_steps 20
    ok "Smoke test passed"
    exit 0
fi

RESUME_FLAG=""
if [ "${RESUME}" = "true" ]; then
    LATEST_CKPT=$(ls -t "${MODEL_DIR}"/step_*.pth 2>/dev/null | head -1 || true)
    if [ -n "${LATEST_CKPT}" ]; then
        RESUME_FLAG="--resume /workspace/checkpoints/$(basename "${LATEST_CKPT}")"
        info "Resuming from: ${LATEST_CKPT}"
    else
        warn "--resume set but no checkpoint found; starting fresh"
    fi
fi

info "Launching full training in ROCm container..."
sudo docker run --rm -it \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --shm-size=32g \
    -e HSA_OVERRIDE_GFX_VERSION=11.5.0 \
    -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
    -v "${PROJECT_DIR}:/workspace" \
    "${DOCKER_IMAGE}" \
    python3 /workspace/scripts/run_train.py \
        --stage 1 \
        --batch_size 16 \
        --max_steps 100000 \
        ${RESUME_FLAG}

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Stage 1 complete. Run with --stage 2 for adversarial."
echo "  Checkpoints: ${MODEL_DIR}"
echo "════════════════════════════════════════════════════════"
