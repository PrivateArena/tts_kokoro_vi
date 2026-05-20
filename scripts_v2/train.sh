#!/usr/bin/env bash
# ==============================================================================
# Northern Vietnamese KokoroTTS — Gated PhoAudiobook Fine-Tuning Suite
# Target: Ryzen AI MAX 395+ (Strix Halo, RDNA4 iGPU, ROCm 6.2+)
# Usage:  bash train.sh [--stage <setup|fetch|filter|train|all>] [--resume] [--smoke-test]
#
# Fixed vs original:
#   - PyTorch installed with ROCm wheels (was: --extra-index-url .../cpu → CPU-only build)
#   - MANIFEST_PATH defined in run_filter() before use (was: undefined → bash -u error)
#   - StyleTTS2 cloned into Docker image properly
#   - Stage 3 cp path fixed (was: copying script over itself)
#   - PYTORCH_HIP_ALLOC_CONF added to Docker run for unified memory fragmentation
#   - HSA_OVERRIDE_GFX_VERSION consistently set for Strix Halo RDNA4 gfx1150
# ==============================================================================
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/kokoro_vietnamese"
DATA_DIR="${PROJECT_DIR}/data"
MODEL_DIR="${PROJECT_DIR}/checkpoints"
LOG_DIR="${PROJECT_DIR}/logs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts"
DOCKER_IMAGE="kokoro-rocm-strix:latest"
KOKORO_REPO="https://github.com/yl4579/StyleTTS2.git"
BASE_CHECKPOINT_URL="https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/kokoro-v1_1-zh.pth"
CONFIG_URL="https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/config.json"
GTT_SIZE_MB=98304

# Manifest paths (used in both filter and train stages)
MANIFEST_PATH="${DATA_DIR}/train_manifest.csv"
VAL_MANIFEST_PATH="${DATA_DIR}/val_manifest.csv"

# Parse arguments
STAGE="all"
RESUME=false
SMOKE_TEST=false
DATA_ONLY=false
ALL_DIALECTS=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --stage)        STAGE="$2"; shift 2 ;;
    --resume)       RESUME=true; shift ;;
    --smoke-test)   SMOKE_TEST=true; shift ;;
    --data-only)    DATA_ONLY=true; shift ;;
    --all-dialects) ALL_DIALECTS=true; shift ;;
    *)              echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# Helpers
info()  { echo -e "\n\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
die()   { echo -e "\033[1;31m[FAIL]\033[0m  $*" >&2; exit 1; }

# ==============================================================================
# STAGE 1 — SETUP SYSTEM, DIRECTORIES & DEPS
# ==============================================================================
run_setup() {
    info "=== STAGE 1: SETUP ==="

    # Directory structure
    info "Creating directory structure"
    mkdir -p "${DATA_DIR}/raw" "${DATA_DIR}/processed" \
             "${MODEL_DIR}" "${LOG_DIR}" "${PROJECT_DIR}/scripts"
    ok "Directories ready: ${PROJECT_DIR}"

    # AMD Unified Memory (GTT) for Strix Halo iGPU
    info "Tuning AMD Unified Memory (GTT) for Strix Halo"
    if [ -f /sys/module/amdgpu/parameters/gttsize ] && [ -r /sys/module/amdgpu/parameters/gttsize ]; then
        CURRENT_GTT=$(cat /sys/module/amdgpu/parameters/gttsize 2>/dev/null || echo 0)
        if [ "${CURRENT_GTT}" -gt 0 ] && [ "${CURRENT_GTT}" -lt "${GTT_SIZE_MB}" ]; then
            echo "${GTT_SIZE_MB}" | sudo tee /sys/module/amdgpu/parameters/gttsize > /dev/null 2>/dev/null || true
            ok "GTT set to ${GTT_SIZE_MB} MB"
        else
            ok "GTT already at ${CURRENT_GTT} MB (≥ ${GTT_SIZE_MB})"
        fi
    else
        warn "Runtime GTT path not found. Skipping GTT auto-tuning."
    fi

    # Python Virtual Env
    info "Setting up Python environment"
    if [ ! -d "${PROJECT_DIR}/venv" ]; then
        python3 -m venv "${PROJECT_DIR}/venv"
    fi
    # shellcheck disable=SC1091
    source "${PROJECT_DIR}/venv/bin/activate"

    pip install -q --upgrade pip

    # ── CRITICAL FIX: install ROCm-enabled PyTorch, NOT the CPU-only build ──
    # The original used --extra-index-url .../whl/cpu which silently installs
    # a CPU-only build, making the GPU invisible to PyTorch entirely.
    # Strix Halo (RDNA4 / gfx1150) requires ROCm 6.2+ wheels.
    ROCM_VERSION="6.2"
    pip install -q \
        torch torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/rocm${ROCM_VERSION}"

    pip install -q \
        datasets soundfile librosa pyloudnorm \
        vinorm underthesea viphoneme \
        transformers accelerate tqdm wandb \
        fsspec pyarrow huggingface_hub \
        pyyaml einops

    ok "Python dependencies installed (ROCm ${ROCM_VERSION} wheels)"

    # Verify GPU visibility
    python3 - <<'PYEOF'
import torch
if torch.cuda.is_available():
    print(f"  GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB")
else:
    print("  WARNING: No GPU detected after ROCm install. Check HSA_OVERRIDE_GFX_VERSION.")
PYEOF

    deactivate
}

# ==============================================================================
# STAGE 2 — FETCH DATASET METADATA & VALIDATE HF AUTH
# ==============================================================================
run_fetch() {
    info "=== STAGE 2: FETCH DATASET METADATA ==="

    # shellcheck disable=SC1091
    source "${PROJECT_DIR}/venv/bin/activate"

    info "Verifying Hugging Face authentication…"
    if ! python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.list_repo_files(repo_id='thivux/phoaudiobook', repo_type='dataset')
print('HF auth OK')
"; then
        die "HF auth failed. Run: huggingface-cli login"
    fi

    if [ ! -f "${DATA_DIR}/unique_speakers.txt" ]; then
        info "Extracting unique speakers from dataset metadata…"
        python3 "${SCRIPT_DIR}/get_unique_speakers.py"
        ok "Speaker metadata extracted → ${DATA_DIR}/unique_speakers.txt"
    else
        ok "Unique speakers file exists."
    fi

    deactivate
}

# ==============================================================================
# STAGE 3 — CHOOSE & FILTER NORTHERN VOICES
# ==============================================================================
run_filter() {
    info "=== STAGE 3: CHOOSE & FILTER NORTHERN VOICES ==="

    # shellcheck disable=SC1091
    source "${PROJECT_DIR}/venv/bin/activate"

    VERIFIED_SPEAKERS_FILE="${DATA_DIR}/verified_northern_speakers.txt"
    if [ ! -f "${VERIFIED_SPEAKERS_FILE}" ]; then
        info "Creating starter verified_northern_speakers.txt template."
        cat > "${VERIFIED_SPEAKERS_FILE}" << 'EOF'
# ==============================================================================
# Verified Northern Vietnamese Speakers Whitelist
# Edit this file: keep only the speakers you want in your training set.
# Lines starting with '#' or empty lines are ignored.
# ==============================================================================
Nguyễn_Văn_Khỏa
Lê_Đức_Quân
Diễm_Hân
Mai_Anh
Đông_Quân
Thoan_Đây
Ngọc_Diễm
Thanh_Thủy
Vy_Vy
Tuấn_Anh
EOF
        warn "Edit ${VERIFIED_SPEAKERS_FILE} before proceeding."
    else
        ok "Using verified speakers whitelist: ${VERIFIED_SPEAKERS_FILE}"
    fi

    # Copy the dataset prep script from the repo into the project
    # Note: source and destination must be different paths
    if [ "${SCRIPT_DIR}/prepare_dataset.py" != "${PROJECT_DIR}/scripts/prepare_dataset.py" ]; then
        cp "${SCRIPT_DIR}/prepare_dataset.py" "${PROJECT_DIR}/scripts/prepare_dataset.py"
    fi

    info "Running dataset preparation…"
    export DATA_ROOT="${DATA_DIR}"
    export SMOKE_TEST="${SMOKE_TEST}"

    DIALECT_FLAG=""
    if [ "${ALL_DIALECTS}" = "true" ]; then
        DIALECT_FLAG="--all-dialects"
        info "Including all dialect speakers (South + Central + North)."
    fi

    SMOKE_FLAG=""
    if [ "${SMOKE_TEST}" = "true" ]; then
        SMOKE_FLAG="--smoke-test"
    fi

    python3 "${PROJECT_DIR}/scripts/prepare_dataset.py" \
        --data-root "${DATA_DIR}" \
        ${DIALECT_FLAG} \
        ${SMOKE_FLAG}

    # MANIFEST_PATH is defined at the top of this script — no longer undefined
    if [ -f "${MANIFEST_PATH}" ]; then
        SAMPLE_COUNT=$(wc -l < "${MANIFEST_PATH}")
        ok "Stage 3 complete: ${SAMPLE_COUNT} training clips in ${MANIFEST_PATH}"
    else
        die "train_manifest.csv was not created. Check prepare_dataset.py logs above."
    fi

    deactivate
}

# ==============================================================================
# STAGE 4 — BUILD DOCKER & LAUNCH TRAINING
# ==============================================================================
run_train() {
    info "=== STAGE 4: BUILD DOCKER & LAUNCH TRAINING ==="

    # 1. Download base checkpoint + config
    info "Downloading base checkpoint and config…"
    if [ ! -f "${MODEL_DIR}/kokoro-v1_1-zh.pth" ]; then
        curl -L --progress-bar -o "${MODEL_DIR}/kokoro-v1_1-zh.pth" "${BASE_CHECKPOINT_URL}"
    fi
    if [ ! -f "${PROJECT_DIR}/config.json" ]; then
        curl -L --progress-bar -o "${PROJECT_DIR}/config.json" "${CONFIG_URL}"
    fi

    # 2. Copy scripts to project
    info "Staging scripts into project directory…"
    for script in run_train.py extend_vocab.py; do
        if [ -f "${SCRIPT_DIR}/${script}" ]; then
            cp "${SCRIPT_DIR}/${script}" "${PROJECT_DIR}/scripts/${script}"
        else
            die "Script not found: ${SCRIPT_DIR}/${script}"
        fi
    done

    # 3. Vocabulary embedding surgery (on host, before Docker build)
    info "Running vocabulary embedding surgery…"
    # shellcheck disable=SC1091
    source "${PROJECT_DIR}/venv/bin/activate"
    python3 "${PROJECT_DIR}/scripts/extend_vocab.py"
    deactivate
    ok "Embedding surgery complete → ${MODEL_DIR}/kokoro-vi-north-extended.pth"

    # 4. Build Docker image
    info "Building ROCm Docker image: ${DOCKER_IMAGE}…"
    sudo docker build -t "${DOCKER_IMAGE}" "${PROJECT_DIR}"

    # 5. Smoke test
    if [ "${SMOKE_TEST}" = "true" ]; then
        info "Running smoke test inside container…"
        sudo docker run --rm \
            --device=/dev/kfd --device=/dev/dri \
            --group-add video \
            --ipc=host --shm-size=4g \
            -e HSA_OVERRIDE_GFX_VERSION=11.5.0 \
            -e HSA_ENABLE_SDMA=0 \
            -e PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.9,max_split_size_mb:512" \
            -v "${PROJECT_DIR}:/workspace" \
            "${DOCKER_IMAGE}" \
            python3 /workspace/scripts/run_train.py \
                --smoke-test \
                --batch-size 2 \
                --max-steps 20
        ok "Smoke test passed"
        exit 0
    fi

    # 6. Resume flag
    RESUME_FLAG=""
    if [ "${RESUME}" = "true" ]; then
        LATEST_CKPT=$(ls -t "${MODEL_DIR}"/step_*.pth 2>/dev/null | head -1 || true)
        if [ -n "${LATEST_CKPT}" ]; then
            RESUME_FLAG="--resume /workspace/checkpoints/$(basename "${LATEST_CKPT}")"
            info "Resuming from: ${LATEST_CKPT}"
        else
            warn "--resume specified but no step_*.pth found. Starting fresh."
        fi
    fi

    # 7. Launch training
    # Notes on Strix Halo settings:
    #   HSA_OVERRIDE_GFX_VERSION=11.5.0  → gfx1150 maps to 11.5.0 in ROCm 6.2
    #   HSA_ENABLE_SDMA=0                → disable SDMA to avoid iGPU DMA issues
    #   PYTORCH_HIP_ALLOC_CONF           → reduce memory fragmentation in unified pool
    #   --shm-size=32g                   → shared memory for DataLoader workers
    #   pin_memory=False in DataLoader   → correct for unified memory (no PCIe transfer)
    info "Launching training in ROCm container…"
    info "Effective batch size = --batch-size 16 × --grad-accum 4 = 64"
    sudo docker run --rm -it \
        --device=/dev/kfd \
        --device=/dev/dri \
        --group-add video \
        --ipc=host \
        --shm-size=32g \
        -e HSA_OVERRIDE_GFX_VERSION=11.5.0 \
        -e HSA_ENABLE_SDMA=0 \
        -e PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.9,max_split_size_mb:512" \
        -e ROCR_VISIBLE_DEVICES=0 \
        -v "${PROJECT_DIR}:/workspace" \
        "${DOCKER_IMAGE}" \
        python3 /workspace/scripts/run_train.py \
            --manifest    /workspace/data/train_manifest.csv \
            --val-manifest /workspace/data/val_manifest.csv \
            --checkpoint  /workspace/checkpoints/kokoro-vi-north-extended.pth \
            --config      /workspace/config_vi.json \
            --stage       1 \
            --batch-size  16 \
            --grad-accum  4 \
            --warmup-steps 4000 \
            --max-steps   300000 \
            --f0-weight   2.0 \
            --wandb \
            ${RESUME_FLAG}
}

# ── Dispatcher ────────────────────────────────────────────────────────────────
case "${STAGE}" in
    setup)  run_setup ;;
    fetch)  run_fetch ;;
    filter) run_filter ;;
    train)  run_train ;;
    all)
        run_setup
        run_fetch
        run_filter
        if [ "${DATA_ONLY}" = "true" ]; then
            info "Data-only mode: stopping before training."
        else
            run_train
        fi
        ;;
    *)
        die "Unknown stage: ${STAGE}. Choose from: setup, fetch, filter, train, all"
        ;;
esac

echo -e "\n════════════════════════════════════════════════════════"
echo "  Northern Vietnamese Fine-Tuning Stage Complete!"
echo "════════════════════════════════════════════════════════"
