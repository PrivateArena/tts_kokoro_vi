#!/usr/bin/env bash
# ==============================================================================
# Northern Vietnamese KokoroTTS — Fine-Tuning Suite
# Target: Ryzen AI MAX 395+ (Strix Halo, RDNA4 iGPU, ROCm 6.2+)
#
# Usage:
#   bash train.sh [--stage <setup|fetch|filter|train|all>]
#                 [--resume] [--smoke-test] [--data-only]
#
# Stages:
#   setup   — create dirs, Python venv, install deps
#   fetch   — validate HF auth, extract speaker metadata
#   filter  — filter/prepare dataset (prepare_dataset.py)
#   train   — build Docker, run embedding surgery, launch training
#   all     — run all stages in sequence (default)
# ==============================================================================
set -euo pipefail

# ── Project layout ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/kokoro_vietnamese"
DATA_DIR="${PROJECT_DIR}/data"
MODEL_DIR="${PROJECT_DIR}/checkpoints"
LOG_DIR="${PROJECT_DIR}/logs"
VENV_DIR="${PROJECT_DIR}/venv"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"

DOCKER_IMAGE="kokoro-rocm-strix:latest"

# Base Kokoro checkpoint (Chinese model — best multilingual starting point)
BASE_CHECKPOINT_URL="https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/kokoro-v1_1-zh.pth"
CONFIG_URL="https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/config.json"

# Strix Halo unified memory — request 96 GB GTT (leave headroom for CPU)
GTT_SIZE_MB=98304

# ── Argument parsing ───────────────────────────────────────────────────────────
STAGE="all"
RESUME=false
SMOKE_TEST=false
DATA_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)      STAGE="$2";  shift 2 ;;
        --resume)     RESUME=true; shift   ;;
        --smoke-test) SMOKE_TEST=true; shift ;;
        --data-only)  DATA_ONLY=true; shift  ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Logging helpers ────────────────────────────────────────────────────────────
info()  { echo -e "\n\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
die()   { echo -e "\033[1;31m[FAIL]\033[0m  $*" >&2; exit 1; }

# ── Activate venv helper ───────────────────────────────────────────────────────
activate_venv() {
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
}

# ==============================================================================
# STAGE 1 — SETUP
# ==============================================================================
run_setup() {
    info "=== STAGE 1: SETUP ==="

    # 1. Directory tree
    info "Creating directory structure …"
    mkdir -p \
        "${DATA_DIR}/raw" \
        "${DATA_DIR}/processed" \
        "${MODEL_DIR}" \
        "${LOG_DIR}" \
        "${SCRIPTS_DIR}" \
        "${PROJECT_DIR}/models"
    ok "Directories ready: ${PROJECT_DIR}"

    # 2. AMD GTT (unified memory) tuning for Strix Halo iGPU
    info "Checking AMD GTT (unified memory) configuration …"
    GTT_PARAM="/sys/module/amdgpu/parameters/gttsize"
    if [[ -r "${GTT_PARAM}" ]]; then
        CURRENT_GTT=$(cat "${GTT_PARAM}" 2>/dev/null || echo 0)
        if [[ "${CURRENT_GTT}" -lt "${GTT_SIZE_MB}" ]]; then
            if echo "${GTT_SIZE_MB}" | sudo tee "${GTT_PARAM}" > /dev/null 2>&1; then
                ok "GTT set to ${GTT_SIZE_MB} MB (was ${CURRENT_GTT} MB)"
            else
                warn "Could not set GTT — run as root or add to /etc/modprobe.d/amdgpu.conf:"
                warn "  options amdgpu gttsize=${GTT_SIZE_MB}"
            fi
        else
            ok "GTT already at ${CURRENT_GTT} MB (≥ ${GTT_SIZE_MB})"
        fi
    else
        warn "GTT parameter not found — skipping. Normal on non-AMD systems."
    fi

    # 3. Python virtual environment
    info "Setting up Python virtual environment …"
    if [[ ! -d "${VENV_DIR}" ]]; then
        python3 -m venv "${VENV_DIR}"
        ok "Created venv at ${VENV_DIR}"
    else
        ok "Venv already exists at ${VENV_DIR}"
    fi
    activate_venv
    pip install -q --upgrade pip

    # Core TTS/audio dependencies
    pip install -q \
        datasets \
        soundfile \
        librosa \
        pyloudnorm \
        vinorm \
        underthesea \
        viphoneme \
        phonemizer \
        grapheme \
        regex

    # Quality filtering (DNSMOS)
    pip install -q onnxruntime

    # ML / training
    pip install -q \
        transformers \
        accelerate \
        tqdm \
        wandb \
        PyYAML

    # PyTorch — ROCm build for Strix Halo (gfx1150 / RDNA4)
    # If ROCm 6.2 wheels are not yet published for your exact gfx version,
    # install with HSA_OVERRIDE_GFX_VERSION=11.5.0 at runtime (set in Dockerfile).
    pip install -q \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/rocm6.2 \
        || pip install -q torch torchvision torchaudio  # CPU fallback for dev machines

    # HuggingFace hub (needed for speaker extraction)
    pip install -q huggingface_hub fsspec pyarrow

    ok "Python dependencies installed."
    deactivate
}

# ==============================================================================
# STAGE 2 — FETCH
# ==============================================================================
run_fetch() {
    info "=== STAGE 2: FETCH DATASET METADATA ==="
    activate_venv

    # 1. HF authentication check
    info "Verifying Hugging Face authentication …"
    if ! python3 - <<'PYEOF'
from huggingface_hub import HfApi
api = HfApi()
try:
    files = list(api.list_repo_files(repo_id="thivux/phoaudiobook", repo_type="dataset"))
    print(f"  Auth OK — {len(files)} files visible in dataset.")
except Exception as e:
    raise SystemExit(f"HF auth failed: {e}\nRun: huggingface-cli login")
PYEOF
    then
        deactivate
        die "Hugging Face authentication failed. Run: huggingface-cli login"
    fi

    # 2. DNSMOS model download prompt
    DNSMOS_PATH="${PROJECT_DIR}/models/dnsmos_p835.onnx"
    if [[ ! -f "${DNSMOS_PATH}" ]]; then
        warn "DNSMOS quality model not found at ${DNSMOS_PATH}."
        warn "Download from: https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS"
        warn "Place the ONNX file at: ${DNSMOS_PATH}"
        warn "Without it, a coarser SNR-proxy quality gate will be used."
    else
        ok "DNSMOS model found."
    fi

    # 3. Speaker metadata extraction
    if [[ ! -f "${DATA_DIR}/unique_speakers.txt" ]]; then
        info "Extracting unique speaker list (metadata-only, no audio download) …"
        # Copy script to project dir and run from there
        cp "${SCRIPT_DIR}/get_unique_speakers.py" "${SCRIPTS_DIR}/"
        pushd "${PROJECT_DIR}" > /dev/null
        python3 scripts/get_unique_speakers.py
        popd > /dev/null
        ok "Speaker list saved to ${DATA_DIR}/unique_speakers.txt"
    else
        NSPK=$(wc -l < "${DATA_DIR}/unique_speakers.txt")
        ok "Speaker list already exists (${NSPK} speakers). Delete to regenerate."
    fi

    deactivate
}

# ==============================================================================
# STAGE 3 — FILTER / PREPARE DATASET
# ==============================================================================
run_filter() {
    info "=== STAGE 3: FILTER & PREPARE DATASET ==="
    activate_venv

    # 1. Generate verified Northern speaker whitelist template if missing
    VERIFIED="${DATA_DIR}/verified_northern_speakers.txt"
    if [[ ! -f "${VERIFIED}" ]]; then
        info "Creating starter verified_northern_speakers.txt …"
        cat > "${VERIFIED}" << 'EOF'
# ==============================================================================
# Verified Northern Vietnamese Speakers Whitelist
# Edit this file before running the filter stage.
# Lines starting with '#' or blank lines are ignored.
# Speaker names must match exactly what appears in the dataset.
# Run Stage 2 (fetch) first to get the full speaker list.
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
        warn "EDIT ${VERIFIED} to select your Northern speakers, then re-run --stage filter"
        deactivate
        exit 0
    fi

    NSPK=$(grep -v '^#' "${VERIFIED}" | grep -v '^$' | wc -l)
    ok "Using ${NSPK} whitelisted Northern speakers."

    # 2. Copy and run dataset preparation script
    cp "${SCRIPT_DIR}/prepare_dataset.py" "${SCRIPTS_DIR}/"

    SMOKE_FLAG=""
    [[ "${SMOKE_TEST}" == "true" ]] && SMOKE_FLAG="--smoke-test"

    info "Running dataset preparation pipeline …"
    pushd "${PROJECT_DIR}" > /dev/null
    DATA_ROOT="${DATA_DIR}" python3 scripts/prepare_dataset.py ${SMOKE_FLAG}
    popd > /dev/null

    if [[ -f "${DATA_DIR}/train_manifest.csv" ]]; then
        NSAMP=$(wc -l < "${DATA_DIR}/train_manifest.csv")
        ok "Dataset ready: ${NSAMP} training samples in train_manifest.csv"
    else
        die "train_manifest.csv not created — check prepare_dataset.py output above."
    fi

    deactivate
}

# ==============================================================================
# STAGE 4 — BUILD DOCKER & LAUNCH TRAINING
# ==============================================================================
run_train() {
    info "=== STAGE 4: BUILD DOCKER & LAUNCH TRAINING ==="

    # 1. Download base checkpoint
    info "Checking base checkpoint …"
    if [[ ! -f "${MODEL_DIR}/kokoro-v1_1-zh.pth" ]]; then
        info "Downloading Kokoro-82M-v1.1-zh checkpoint …"
        curl -L --retry 5 --retry-delay 3 \
            -o "${MODEL_DIR}/kokoro-v1_1-zh.pth" \
            "${BASE_CHECKPOINT_URL}"
    else
        ok "Base checkpoint already present."
    fi

    if [[ ! -f "${PROJECT_DIR}/config.json" ]]; then
        info "Downloading base config.json …"
        curl -L --retry 3 \
            -o "${PROJECT_DIR}/config.json" \
            "${CONFIG_URL}"
    else
        ok "Base config.json already present."
    fi

    # 2. Copy scripts into project
    info "Copying training scripts …"
    cp "${SCRIPT_DIR}/run_train.py"    "${SCRIPTS_DIR}/"
    cp "${SCRIPT_DIR}/extend_vocab.py" "${SCRIPTS_DIR}/"

    # 3. Generate Dockerfile
    info "Writing Dockerfile …"
    cat > "${PROJECT_DIR}/Dockerfile" << 'DOCKERFILE'
# ── KokoroTTS Vietnamese — ROCm Training Image ──────────────────────────────
# Target: Ryzen AI MAX 395+ (Strix Halo RDNA4, gfx1150)
FROM rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0

# AMD iGPU unified memory and ROCm tuning
ENV HSA_OVERRIDE_GFX_VERSION=11.5.0 \
    HSA_ENABLE_SDMA=0 \
    PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
    GPU_MAX_ALLOC_PERCENT=100 \
    GPU_MAX_HEAP_SIZE=100 \
    ROCR_VISIBLE_DEVICES=0 \
    HIP_VISIBLE_DEVICES=0 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# System deps
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
        git curl ffmpeg sox libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Clone StyleTTS2 (the actual model code)
RUN git clone --depth 1 https://github.com/yl4579/StyleTTS2.git /opt/StyleTTS2

# Python dependencies
COPY requirements_train.txt /tmp/requirements_train.txt
RUN pip install --no-cache-dir -r /tmp/requirements_train.txt

# Copy workspace scripts
COPY scripts/ /workspace/scripts/
COPY config_vi.json /workspace/config_vi.json
COPY checkpoints/kokoro-vi-north-extended.pth /workspace/checkpoints/

CMD ["python3", "/workspace/scripts/run_train.py"]
DOCKERFILE

    # 4. Write requirements_train.txt for the Docker image
    cat > "${PROJECT_DIR}/requirements_train.txt" << 'REQEOF'
soundfile
librosa
pyloudnorm
vinorm
underthesea
viphoneme
grapheme
regex
transformers>=4.40.0
accelerate
tqdm
wandb
PyYAML
onnxruntime
datasets
huggingface_hub
fsspec
pyarrow
REQEOF

    # 5. Vocabulary embedding surgery (run on host inside venv)
    info "Running vocabulary embedding surgery …"
    activate_venv
    pushd "${PROJECT_DIR}" > /dev/null
    python3 scripts/extend_vocab.py \
        --project-dir . \
        --checkpoint  "checkpoints/kokoro-v1_1-zh.pth" \
        --config      "config.json" \
        --manifest    "data/train_manifest.csv" \
        --out-checkpoint "checkpoints/kokoro-vi-north-extended.pth" \
        --out-config  "config_vi.json"
    popd > /dev/null
    deactivate
    ok "Embedding surgery complete."

    # 6. Build Docker image
    info "Building ROCm Docker image: ${DOCKER_IMAGE} …"
    sudo docker build -t "${DOCKER_IMAGE}" "${PROJECT_DIR}"
    ok "Docker image built."

    # 7. Smoke test (fast data pipeline + model forward pass check)
    if [[ "${SMOKE_TEST}" == "true" ]]; then
        info "Running smoke test inside container …"
        sudo docker run --rm \
            --ipc=host \
            --shm-size=4g \
            -v "${PROJECT_DIR}:/workspace" \
            "${DOCKER_IMAGE}" \
            python3 /workspace/scripts/run_train.py \
                --smoke-test \
                --batch-size 4 \
                --max-steps  30
        ok "Smoke test passed."
        exit 0
    fi

    # 8. Build resume flag
    RESUME_FLAG=""
    if [[ "${RESUME}" == "true" ]]; then
        LATEST_CKPT=$(ls -t "${MODEL_DIR}"/step_*.pth 2>/dev/null | head -1 || true)
        if [[ -n "${LATEST_CKPT}" ]]; then
            RESUME_FLAG="--resume /workspace/checkpoints/$(basename "${LATEST_CKPT}")"
            info "Resuming from: ${LATEST_CKPT}"
        else
            warn "--resume specified but no step_*.pth checkpoint found — starting fresh."
        fi
    fi

    # 9. Launch Stage 1 training
    info "Launching Stage 1 training (200k steps, batch 32, grad-accum 2 = effective 64) …"
    info "With 128 GB unified VRAM this should take ~18–24 hours."
    sudo docker run --rm -it \
        --device=/dev/kfd \
        --device=/dev/dri \
        --group-add video \
        --ipc=host \
        --shm-size=32g \
        -e HSA_OVERRIDE_GFX_VERSION=11.5.0 \
        -e HSA_ENABLE_SDMA=0 \
        -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
        -e GPU_MAX_ALLOC_PERCENT=100 \
        -e GPU_MAX_HEAP_SIZE=100 \
        -v "${PROJECT_DIR}:/workspace" \
        "${DOCKER_IMAGE}" \
        python3 /workspace/scripts/run_train.py \
            --stage         1 \
            --batch-size    32 \
            --grad-accum    2 \
            --max-steps     200000 \
            --warmup-steps  5000 \
            --cosine-t0     50000 \
            --lr-encoder    5e-6 \
            --lr-style      1e-5 \
            --lr-decoder    2e-5 \
            --lr-new        1e-4 \
            --save-every    5000 \
            --val-every     5000 \
            --log-every     100 \
            ${RESUME_FLAG}
}

# ==============================================================================
# Dispatcher
# ==============================================================================
case "${STAGE}" in
    setup)  run_setup  ;;
    fetch)  run_fetch  ;;
    filter) run_filter ;;
    train)  run_train  ;;
    all)
        run_setup
        run_fetch
        run_filter
        if [[ "${DATA_ONLY}" == "true" ]]; then
            info "Data-only mode: stopping before training. Re-run without --data-only to train."
        else
            run_train
        fi
        ;;
    *)
        die "Unknown stage '${STAGE}'. Valid stages: setup, fetch, filter, train, all"
        ;;
esac

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Northern Vietnamese Fine-Tuning Stage Complete!"
echo "  Stage: ${STAGE}"
echo "════════════════════════════════════════════════════════"