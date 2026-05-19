#!/usr/bin/env bash
# ==============================================================================
# Northern Vietnamese KokoroTTS — Gated PhoAudiobook Fine-Tuning Suite
# Target: Ryzen AI MAX 395+ (Strix Halo, RDNA4 iGPU, ROCm 6.2+)
# Usage:  bash train.sh [--stage <setup|fetch|filter|train>] [--resume] [--smoke-test]
# ==============================================================================
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/kokoro_vietnamese"
DATA_DIR="${PROJECT_DIR}/data"
MODEL_DIR="${PROJECT_DIR}/checkpoints"
LOG_DIR="${PROJECT_DIR}/logs"
DOCKER_IMAGE="kokoro-rocm-strix:latest"
KOKORO_REPO="https://github.com/yl4579/StyleTTS2.git"
BASE_CHECKPOINT_URL="https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/kokoro-v1_1-zh.pth"
CONFIG_URL="https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/config.json"
GTT_SIZE_MB=98304

# Parse arguments
STAGE="all"
RESUME=false
SMOKE_TEST=false
DATA_ONLY=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --stage)      STAGE="$2"; shift 2 ;;
    --resume)     RESUME=true; shift ;;
    --smoke-test) SMOKE_TEST=true; shift ;;
    --data-only)  DATA_ONLY=true; shift ;;
    *)            echo "Unknown argument: $1"; exit 1 ;;
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
    
    # 1. Directory structure
    info "Creating directory structure"
    mkdir -p "${DATA_DIR}/raw" "${DATA_DIR}/processed" \
             "${MODEL_DIR}" "${LOG_DIR}" "${PROJECT_DIR}/scripts"
    ok "Directories ready: ${PROJECT_DIR}"

    # 2. AMD Unified Memory (GTT) for Strix Halo iGPU
    info "Tuning AMD Unified Memory (GTT) for Strix Halo"
    if [ -f /sys/module/amdgpu/parameters/gttsize ] && [ -r /sys/module/amdgpu/parameters/gttsize ]; then
        CURRENT_GTT=$(cat /sys/module/amdgpu/parameters/gttsize 2>/dev/null || echo 0)
        if [ "${CURRENT_GTT}" -gt 0 ] && [ "${CURRENT_GTT}" -lt "${GTT_SIZE_MB}" ]; then
            echo "${GTT_SIZE_MB}" | sudo tee /sys/module/amdgpu/parameters/gttsize > /dev/null 2>/dev/null || true
            ok "GTT set to ${GTT_SIZE_MB} MB (if sudo authorized)"
        else
            ok "GTT already at ${CURRENT_GTT} MB (≥ ${GTT_SIZE_MB})"
        fi
    else
        warn "Runtime GTT path not found or not readable. Skipping GTT parameter auto-tuning."
    fi

    # 3. Python Virtual Env
    info "Setting up Python environment"
    if [ ! -d "${PROJECT_DIR}/venv" ]; then
        python3 -m venv "${PROJECT_DIR}/venv"
    fi
    # shellcheck disable=SC1091
    source "${PROJECT_DIR}/venv/bin/activate"

    pip install -q --upgrade pip
    pip install -q \
        datasets soundfile librosa pyloudnorm \
        vinorm underthesea viphoneme phonemizer \
        transformers accelerate tqdm wandb \
        torch torchvision torchaudio \
        --extra-index-url https://download.pytorch.org/whl/cpu
        
    ok "Python dependencies installed in virtual environment"
    deactivate
}

# ==============================================================================
# STAGE 2 — FETCH DATASET METADATA & VALIDATE HF AUTH
# ==============================================================================
run_fetch() {
    info "=== STAGE 2: FETCH DATASET METADATA ==="
    
    # shellcheck disable=SC1091
    source "${PROJECT_DIR}/venv/bin/activate"
    
    # 1. Check HF Authentication
    info "Verifying Hugging Face dataset authentication..."
    if ! python3 -c "
from huggingface_hub import HfApi
api = HfApi()
try:
    api.list_repo_files(repo_id='thivux/phoaudiobook', repo_type='dataset')
    print('HF authentication verified successfully!')
except Exception as e:
    raise RuntimeError('Hugging Face authentication failed or dataset is gated. Please run \"huggingface-cli login\" first. Error: ' + str(e))
"; then
        die "Hugging Face authentication failed. Run 'huggingface-cli login' in your shell."
    fi
    
    # 2. Check or run speaker list generation
    if [ ! -f "${DATA_DIR}/unique_speakers.txt" ]; then
        info "Unique speakers file not found. Generating unique speakers metadata..."
        python3 scripts/get_unique_speakers.py
        ok "Unique speakers metadata extracted!"
    else
        ok "Unique speakers file already exists at ${DATA_DIR}/unique_speakers.txt"
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
    
    # 1. Check or generate starter template for verified northern speakers
    VERIFIED_SPEAKERS_FILE="${DATA_DIR}/verified_northern_speakers.txt"
    if [ ! -f "${VERIFIED_SPEAKERS_FILE}" ]; then
        info "Creating starter verified_northern_speakers.txt template."
        cat > "${VERIFIED_SPEAKERS_FILE}" << 'EOF'
# ==============================================================================
# Verified Northern Vietnamese Speakers Whitelist
# Keep only the speaker names you want to include in the training dataset.
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
        warn "Generated starter template at: ${VERIFIED_SPEAKERS_FILE}"
        warn "Please edit this file to include/exclude the speakers of your choice!"
    else
        ok "Using verified speakers whitelist: ${VERIFIED_SPEAKERS_FILE}"
    fi

    # 2. Copy prepare_dataset.py with metadata-first optimization
    info "Copying dataset preparation script"
    cp scripts/prepare_dataset.py "${PROJECT_DIR}/scripts/prepare_dataset.py"

    info "Running dataset preparation..."
    export SMOKE_TEST="${SMOKE_TEST}"
    python3 "${PROJECT_DIR}/scripts/prepare_dataset.py"
    
    SAMPLE_COUNT=$(wc -l < "${MANIFEST_PATH}" || echo 0)
    ok "Stage 3 Complete: ${SAMPLE_COUNT} Northern clips prepared."
    deactivate
}

# ==============================================================================
# STAGE 4 — SET UP DOCKER & LAUNCH TRAINING CONTROLLER
# ==============================================================================
run_train() {
    info "=== STAGE 4: BUILD DOCKER & LAUNCH TRAINING ==="
    
    # 1. Preload base checkpoint and StyleTTS2 configs
    info "Downloading base checkpoints & files..."
    if [ ! -f "${MODEL_DIR}/kokoro-v1_1-zh.pth" ]; then
        curl -L -o "${MODEL_DIR}/kokoro-v1_1-zh.pth" "${BASE_CHECKPOINT_URL}"
    fi
    if [ ! -f "${PROJECT_DIR}/config.json" ]; then
        curl -L -o "${PROJECT_DIR}/config.json" "${CONFIG_URL}"
    fi

    # 2. Copy Dockerfile
    info "Copying Dockerfile"
    cp Dockerfile "${PROJECT_DIR}/Dockerfile"

    # 3. Copy Training Controller Script
    info "Copying run_train.py"
    cp scripts/run_train.py "${PROJECT_DIR}/scripts/run_train.py"

    # 4. Build Docker container
    info "Building ROCm Docker image: ${DOCKER_IMAGE}..."
    sudo docker build -t "${DOCKER_IMAGE}" "${PROJECT_DIR}"

    # 5. Launch Training
    if [ "${SMOKE_TEST}" = "true" ]; then
        info "Running smoke test inside container..."
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
            info "Resuming training from checkpoint: ${LATEST_CKPT}"
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
}

# ── Main Suite Dispatcher ──────────────────────────────────────────────────────
case "${STAGE}" in
    setup)
        run_setup
        ;;
    fetch)
        run_fetch
        ;;
    filter)
        run_filter
        ;;
    train)
        run_train
        ;;
    all)
        run_setup
        run_fetch
        run_filter
        if [ "${DATA_ONLY}" = "true" ]; then
            info "Data-only mode: stopping before training. Run without --data-only to train."
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
