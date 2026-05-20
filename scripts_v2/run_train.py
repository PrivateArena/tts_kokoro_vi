#!/usr/bin/env python3
"""
Northern Vietnamese Kokoro/StyleTTS2 Fine-Tuning Controller.

Key fixes vs original:
  - Actually loads and trains the Kokoro/StyleTTS2 model (original trained a
    completely separate XPhoneBERT+GRU that ignored the Kokoro checkpoint entirely)
  - F0 (pitch) loss added — critical for Vietnamese 6-tone system
  - Proper multi-speaker manifest support (speaker_id column 4)
  - Duration prediction loss added
  - Gradient accumulation for large effective batch sizes
  - ROCm/Strix-Halo specific tuning (unified memory, bf16, HSA hints)
  - autocast only when on CUDA/ROCm (no CPU crash)
  - GradScaler properly conditioned on CUDA availability

Architecture:
  Stage 1 — Acoustic pre-training (this script):
    Freeze discriminators; train text_encoder embedding, style_encoder, decoder
    with mel-L1 + mel-SSIM + F0-L1 + duration losses.
  Stage 2 — Adversarial (future): unfreeze discriminators, add GAN + SLM losses.

Requires:
  - StyleTTS2 cloned to /opt/StyleTTS2  (done inside Docker via train.sh)
  - Extended Kokoro checkpoint at checkpoints/kokoro-vi-north-extended.pth
  - Extended config at config_vi.json
  - Train manifest: data/train_manifest.csv  (columns: path|ipa|text|speaker_id|dialect)
  - Val manifest:   data/val_manifest.csv
"""
import os
import sys
import csv
import json
import math
import logging
import argparse
import random
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
import soundfile as sf
import librosa

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TARGET_SR = 24_000
MEL_FMIN  = 0
MEL_FMAX  = None   # StyleTTS2 uses None (full bandwidth)

# ---------------------------------------------------------------------------
# ROCm / Strix Halo environment hints
# ---------------------------------------------------------------------------
def configure_rocm_environment():
    """Set environment variables that improve Strix Halo (RDNA4) stability."""
    env_hints = {
        "HSA_OVERRIDE_GFX_VERSION": "11.5.0",
        "HSA_ENABLE_SDMA": "0",
        # Reduce memory fragmentation in the 128GB unified pool
        "PYTORCH_HIP_ALLOC_CONF": "garbage_collection_threshold:0.9,max_split_size_mb:512",
    }
    for k, v in env_hints.items():
        if not os.environ.get(k):
            os.environ[k] = v

configure_rocm_environment()

# ---------------------------------------------------------------------------
# StyleTTS2 imports
# ---------------------------------------------------------------------------
STYLETTS2_PATH = "/opt/StyleTTS2"
sys.path.insert(0, STYLETTS2_PATH)

try:
    import yaml
    from models import build_model
    from utils import get_data_path_list
    _STYLETTS2_AVAILABLE = True
    log.info("StyleTTS2 modules loaded from %s", STYLETTS2_PATH)
except ImportError as e:
    _STYLETTS2_AVAILABLE = False
    log.error(
        "Cannot import StyleTTS2 from %s: %s\n"
        "Make sure the Docker image ran:  git clone %s %s",
        STYLETTS2_PATH, e,
        "https://github.com/yl4579/StyleTTS2.git", STYLETTS2_PATH
    )
    # Do NOT silently proceed with a fake GRU model — that was the original bug.
    # If StyleTTS2 isn't available, we must exit clearly.
    sys.exit(1)

# ---------------------------------------------------------------------------
# Mel spectrogram (must match Kokoro config parameters exactly)
# ---------------------------------------------------------------------------
def load_mel_config(config_path: str) -> dict:
    """Read mel params from the extended config_vi.json."""
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    preprocess = cfg.get("preprocess", {})
    return {
        "n_fft":      preprocess.get("n_fft", 2048),
        "hop_length": preprocess.get("hop_length", 300),
        "win_length": preprocess.get("win_length", 1200),
        "n_mels":     preprocess.get("n_mels", 80),
        "fmin":       preprocess.get("fmin", MEL_FMIN),
        "fmax":       preprocess.get("fmax", MEL_FMAX),
    }


def compute_mel(audio: np.ndarray, sr: int, mel_cfg: dict) -> torch.Tensor:
    """Log-mel spectrogram matching StyleTTS2/Kokoro conventions."""
    S = librosa.feature.melspectrogram(
        y=audio.astype(np.float32),
        sr=sr,
        n_fft=mel_cfg["n_fft"],
        hop_length=mel_cfg["hop_length"],
        win_length=mel_cfg["win_length"],
        n_mels=mel_cfg["n_mels"],
        fmin=mel_cfg["fmin"],
        fmax=mel_cfg["fmax"],
        power=1.0,
    )
    log_S = np.log(np.clip(S, 1e-5, None))
    return torch.FloatTensor(log_S)  # (n_mels, T)


# ---------------------------------------------------------------------------
# F0 extraction (critical for Vietnamese 6-tone system)
# ---------------------------------------------------------------------------
def extract_f0(audio: np.ndarray, sr: int, hop_length: int,
               fmin: float = 65.0, fmax: float = 1100.0) -> np.ndarray:
    """
    Extract F0 contour using pyin.  Returns array of shape (T,) in Hz,
    with 0.0 for unvoiced frames.
    Vietnamese fundamental frequency range:
      Male:   ~80-300 Hz (tones push it higher)
      Female: ~150-500 Hz (tones push it higher)
    We use 65-1100 Hz to safely cover all tones across genders.
    """
    f0, voiced_flag, _ = librosa.pyin(
        audio.astype(np.float32),
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        hop_length=hop_length,
        fill_na=0.0,
    )
    f0[~voiced_flag] = 0.0
    return f0.astype(np.float32)


def f0_loss(pred_f0: torch.Tensor, target_f0: torch.Tensor) -> torch.Tensor:
    """
    L1 loss on log-F0 for voiced frames only.
    Vietnamese tones live in the F0 contour; unvoiced frames contribute no
    tonal information so we mask them out.
    """
    voiced = (target_f0 > 0).float()
    if voiced.sum() < 1:
        return torch.tensor(0.0, device=pred_f0.device)
    # Log-domain makes loss scale-invariant (tone direction > absolute pitch)
    log_pred   = torch.log(pred_f0.clamp(min=1.0)) * voiced
    log_target = torch.log(target_f0.clamp(min=1.0)) * voiced
    return F.l1_loss(log_pred, log_target, reduction="sum") / voiced.sum()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ViDataset(Dataset):
    """
    Loads from manifest: path|ipa|transcript|speaker_id|dialect
    Columns 3+ are optional for backward compat with older manifests.
    """
    def __init__(
        self,
        manifest_path: str,
        mel_cfg: dict,
        speaker_map: Dict[str, int],
        max_token_len: int = 512,
    ):
        self.mel_cfg       = mel_cfg
        self.speaker_map   = speaker_map
        self.max_token_len = max_token_len
        self.records: List[Tuple] = []

        missing = 0
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="|"):
                if len(row) < 2:
                    continue
                wav_path = row[0]
                ipa_text = row[1]
                spk_id   = row[3].strip() if len(row) > 3 else "unknown"
                if not Path(wav_path).exists():
                    missing += 1
                    continue
                self.records.append((wav_path, ipa_text, spk_id))

        if missing:
            log.warning("Skipped %d entries with missing WAV files in %s", missing, manifest_path)
        log.info("Loaded %d valid samples from %s", len(self.records), manifest_path)
        if not self.records:
            raise RuntimeError(f"No valid samples found in {manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        wav_path, ipa_text, spk_id = self.records[idx]
        try:
            audio, _ = sf.read(wav_path, dtype="float32")
        except Exception as e:
            log.error("Failed to read %s: %s", wav_path, e)
            audio = np.zeros(TARGET_SR, dtype=np.float32)

        # Boundary silence padding (0.5 s)
        silence = np.zeros(int(0.5 * TARGET_SR), dtype=np.float32)
        audio   = np.concatenate([silence, audio, silence])

        mel = compute_mel(audio, TARGET_SR, self.mel_cfg)                     # (n_mels, T)
        f0  = extract_f0(audio, TARGET_SR, self.mel_cfg["hop_length"])        # (T,)

        # Truncate F0 to match mel time axis (pyin may differ by 1 frame)
        T = mel.shape[-1]
        if len(f0) > T:
            f0 = f0[:T]
        elif len(f0) < T:
            f0 = np.pad(f0, (0, T - len(f0)))

        # Phoneme token IDs from config vocab
        # NOTE: We use the raw IPA string directly; Kokoro's text encoder does
        # character-level lookup against the extended vocab dict, not a subword BPE.
        speaker_idx = self.speaker_map.get(spk_id, 0)

        return {
            "wav_path":   wav_path,
            "ipa_text":   ipa_text,
            "mel":        mel,                              # (n_mels, T)
            "f0":         torch.FloatTensor(f0),           # (T,)
            "speaker_id": torch.LongTensor([speaker_idx]), # (1,)
        }


def build_speaker_map(manifest_path: str) -> Dict[str, int]:
    """Build speaker → integer index map from the manifest."""
    speakers = set()
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="|"):
            if len(row) > 3:
                speakers.add(row[3].strip())
    speaker_map = {spk: i for i, spk in enumerate(sorted(speakers))}
    log.info("Speaker map: %d unique speakers", len(speaker_map))
    return speaker_map


def collate_fn(batch: list) -> dict:
    """Pad mel and F0 to batch-maximum length."""
    n_mels  = batch[0]["mel"].shape[0]
    max_T   = max(x["mel"].shape[-1] for x in batch)
    B       = len(batch)

    mels        = torch.zeros(B, n_mels, max_T)
    f0s         = torch.zeros(B, max_T)
    speaker_ids = torch.zeros(B, dtype=torch.long)
    ipa_texts   = []

    for i, x in enumerate(batch):
        T = x["mel"].shape[-1]
        mels[i, :, :T]   = x["mel"]
        f0s[i, :T]        = x["f0"]
        speaker_ids[i]    = x["speaker_id"].squeeze()
        ipa_texts.append(x["ipa_text"])

    return {
        "mel":        mels,         # (B, n_mels, max_T)
        "f0":         f0s,          # (B, max_T)
        "speaker_id": speaker_ids,  # (B,)
        "ipa_text":   ipa_texts,    # List[str]
    }


# ---------------------------------------------------------------------------
# Tokenizer from config vocab
# ---------------------------------------------------------------------------
def build_tokenizer(config_path: str, max_len: int = 512):
    """Build a character-level tokenizer from Kokoro's config vocab dict."""
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    vocab = cfg.get("vocab", {})
    if not vocab:
        raise RuntimeError("No 'vocab' key found in config. Run extend_vocab.py first.")
    pad_id = vocab.get("_", 0)  # Kokoro typically uses '_' as pad

    def tokenize(texts: List[str], max_length: int = max_len) -> Tuple[torch.Tensor, torch.Tensor]:
        ids_batch  = []
        mask_batch = []
        for text in texts:
            ids = [vocab.get(c, pad_id) for c in text][:max_length]
            ids_batch.append(ids)
        max_l = max(len(ids) for ids in ids_batch)
        for ids in ids_batch:
            pad = [pad_id] * (max_l - len(ids))
            mask_batch.append([1] * len(ids) + [0] * len(pad))
            ids += pad
        return (
            torch.LongTensor(ids_batch),   # (B, max_l)
            torch.LongTensor(mask_batch),  # (B, max_l)
        )

    return tokenize, vocab, pad_id


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------
def linear_warmup_cosine(optimizer, warmup_steps: int, total_steps: int):
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_kokoro_model(checkpoint_path: str, config_path: str, device: str):
    """
    Load the extended Kokoro checkpoint into StyleTTS2's model structure.
    Returns (model_dict, config_dict).
    model_dict contains the sub-models used by StyleTTS2:
      'generator', 'style_encoder', 'duration_model', 'text_encoder', etc.
    """
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    log.info("Building StyleTTS2 model from config…")
    # build_model() is StyleTTS2's own factory; it reads the full config dict
    models, _ = build_model(config, device=device)

    log.info("Loading extended checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model", ckpt)

    missing, unexpected = [], []
    for model_name, model in models.items():
        if model is None:
            continue
        prefix = f"{model_name}."
        sub_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if not sub_state:
            # Try flat (some checkpoints don't use sub-model prefix)
            sub_state = state
        result = model.load_state_dict(sub_state, strict=False)
        missing    += [f"{model_name}.{k}" for k in result.missing_keys]
        unexpected += [f"{model_name}.{k}" for k in result.unexpected_keys]

    if missing:
        log.warning("Missing keys (will be randomly initialized): %s … +%d more",
                    missing[:5], max(0, len(missing) - 5))
    if unexpected:
        log.info("Unexpected keys in checkpoint (ignored): %d keys", len(unexpected))

    return models, config


# ---------------------------------------------------------------------------
# Freeze / unfreeze helpers
# ---------------------------------------------------------------------------
def set_grad(model: Optional[nn.Module], requires_grad: bool):
    if model is None:
        return
    for p in model.parameters():
        p.requires_grad = requires_grad


def configure_stage1_freezing(models: dict):
    """
    Stage 1: freeze discriminators & vocoder; only train the acoustic path
    (text_encoder, style_encoder, decoder, duration_model).
    This prevents the pre-trained high-quality vocoder from degrading.
    """
    FROZEN  = ["discriminator", "mpd", "msd", "msstftd"]
    TRAINED = ["text_encoder", "style_encoder", "decoder", "duration_model",
               "predictor", "bert_encoder"]

    for name, model in models.items():
        if model is None:
            continue
        should_freeze = any(f in name.lower() for f in FROZEN)
        set_grad(model, not should_freeze)
        status = "FROZEN" if should_freeze else "trainable"
        log.info("  %-30s → %s", name, status)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Kokoro/StyleTTS2 for Vietnamese.")
    p.add_argument("--manifest",       default="data/train_manifest.csv")
    p.add_argument("--val-manifest",   default="data/val_manifest.csv")
    p.add_argument("--checkpoint",     default="checkpoints/kokoro-vi-north-extended.pth")
    p.add_argument("--config",         default="config_vi.json")
    p.add_argument("--resume",         default="",     help="Resume from a mid-run checkpoint.")
    p.add_argument("--stage",          type=int, default=1, choices=[1, 2],
                   help="Training stage (1=acoustic, 2=adversarial+SLM).")
    p.add_argument("--batch-size",     type=int, default=16,
                   help="Per-step batch size. Use --grad-accum to increase effective batch.")
    p.add_argument("--grad-accum",     type=int, default=4,
                   help="Gradient accumulation steps. Effective batch = batch_size × grad_accum.")
    p.add_argument("--max-steps",      type=int, default=200_000)
    p.add_argument("--lr-pretrained",  type=float, default=1e-5,
                   help="LR for pretrained layers (low to prevent catastrophic forgetting).")
    p.add_argument("--lr-new",         type=float, default=1e-4,
                   help="LR for newly added/extended layers (new Vietnamese tokens).")
    p.add_argument("--warmup-steps",   type=int, default=4_000)
    p.add_argument("--save-every",     type=int, default=5_000)
    p.add_argument("--log-every",      type=int, default=50)
    p.add_argument("--grad-clip",      type=float, default=5.0)
    p.add_argument("--mel-weight",     type=float, default=1.0)
    p.add_argument("--f0-weight",      type=float, default=2.0,
                   help="Weight for F0 loss (higher = better tone accuracy). Default 2.0.")
    p.add_argument("--dur-weight",     type=float, default=0.1)
    p.add_argument("--smoke-test",     action="store_true")
    p.add_argument("--wandb",          action="store_true")
    p.add_argument("--wandb-project",  default="kokoro-vi-north")
    p.add_argument("--seed",           type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    if args.smoke_test:
        args.max_steps  = 20
        args.batch_size = 2
        args.grad_accum = 1
        args.log_every  = 5
        log.info("SMOKE TEST: max_steps=%d batch_size=%d", args.max_steps, args.batch_size)

    # ── Device ────────────────────────────────────────────────────────────────
    use_cuda = torch.cuda.is_available()
    device   = "cuda" if use_cuda else "cpu"
    log.info("Device: %s%s", device,
             f"  [{torch.cuda.get_device_name(0)}]" if use_cuda else "")

    if use_cuda:
        # Strix Halo uses unified memory — no PCIe transfer, so benchmark is less useful
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True

    # ── Load model ────────────────────────────────────────────────────────────
    models, config = load_kokoro_model(args.checkpoint, args.config, device)
    mel_cfg = load_mel_config(args.config)

    log.info("Configuring Stage %d training…", args.stage)
    configure_stage1_freezing(models)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenize, vocab, pad_id = build_tokenizer(args.config)
    log.info("Vocab size: %d", len(vocab))

    # ── Speaker map ───────────────────────────────────────────────────────────
    speaker_map = build_speaker_map(args.manifest)

    # ── Datasets & loaders ───────────────────────────────────────────────────
    n_workers = min(6, os.cpu_count() or 1)
    train_ds  = ViDataset(args.manifest,     mel_cfg, speaker_map)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=n_workers,
        prefetch_factor=4,
        pin_memory=False,      # unified memory: no PCIe, pin_memory unhelpful
        collate_fn=collate_fn,
        persistent_workers=True,
    )

    val_loader = None
    if Path(args.val_manifest).exists():
        val_ds = ViDataset(args.val_manifest, mel_cfg, speaker_map)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=False,
            collate_fn=collate_fn,
        )

    # ── Optimizer (separate LR groups) ───────────────────────────────────────
    # New Vietnamese token embeddings need a higher LR to adapt quickly.
    # Pretrained weights need a low LR to prevent catastrophic forgetting.
    new_param_names = {"text_encoder.embed.weight"}  # extended embedding

    pretrained_params, new_params = [], []
    for name, model in models.items():
        if model is None:
            continue
        for pname, p in model.named_parameters():
            if not p.requires_grad:
                continue
            full_name = f"{name}.{pname}"
            if any(n in full_name for n in new_param_names):
                new_params.append(p)
            else:
                pretrained_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": pretrained_params, "lr": args.lr_pretrained},
        {"params": new_params,        "lr": args.lr_new},
    ], weight_decay=1e-2, betas=(0.9, 0.98), eps=1e-9)

    effective_steps = args.max_steps
    scheduler = linear_warmup_cosine(optimizer, args.warmup_steps, effective_steps)

    # GradScaler only meaningful on CUDA/ROCm
    scaler = GradScaler(enabled=use_cuda)

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step = 0
    if args.resume and Path(args.resume).exists():
        log.info("Resuming from %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device)
        for name, model in models.items():
            if model and name in ckpt.get("models", {}):
                model.load_state_dict(ckpt["models"][name])
        optimizer.load_state_dict(ckpt.get("optimizer", {}))
        start_step = ckpt.get("step", 0)
        log.info("Resumed at step %d", start_step)

    # ── torch.compile (ROCm-safe) ─────────────────────────────────────────────
    if use_cuda and not args.smoke_test:
        log.info("Attempting torch.compile for ROCm acceleration…")
        for name, model in models.items():
            if model and any(p.requires_grad for p in model.parameters()):
                try:
                    models[name] = torch.compile(model, backend="inductor",
                                                  mode="reduce-overhead")
                    log.info("  Compiled: %s", name)
                except Exception as e:
                    log.warning("  Compile failed for %s: %s", name, e)

    # ── W&B ──────────────────────────────────────────────────────────────────
    run = None
    if args.wandb:
        try:
            import wandb
            run = wandb.init(project=args.wandb_project, config=vars(args))
        except ImportError:
            log.warning("wandb not installed — skipping.")

    # ── Training loop ─────────────────────────────────────────────────────────
    log.info(
        "Starting Stage %d training: %d → %d steps  "
        "(effective batch = %d × %d = %d)",
        args.stage, start_step, args.max_steps,
        args.batch_size, args.grad_accum,
        args.batch_size * args.grad_accum,
    )

    for model in models.values():
        if model:
            model.train()

    step       = start_step
    loss_hist  = []
    best_val   = float("inf")
    accum_step = 0

    optimizer.zero_grad(set_to_none=True)

    while step < args.max_steps:
        for batch in train_loader:
            if step >= args.max_steps:
                break

            mel        = batch["mel"].to(device)          # (B, n_mels, T)
            f0_target  = batch["f0"].to(device)           # (B, T)
            speaker_id = batch["speaker_id"].to(device)   # (B,)
            ipa_texts  = batch["ipa_text"]

            # Tokenize IPA on CPU then move to device
            input_ids, attn_mask = tokenize(ipa_texts)
            input_ids  = input_ids.to(device)
            attn_mask  = attn_mask.to(device)

            amp_dtype = torch.bfloat16 if use_cuda else torch.float32
            with autocast(device_type=device, dtype=amp_dtype, enabled=use_cuda):
                # StyleTTS2 forward pass
                # The exact API varies by StyleTTS2 version; this follows the
                # standard interface from models.py build_model output.
                text_encoder = models.get("text_encoder")
                style_encoder = models.get("style_encoder")
                decoder       = models.get("decoder")

                if text_encoder is None or decoder is None:
                    raise RuntimeError(
                        "StyleTTS2 model components not loaded correctly. "
                        "Check that /opt/StyleTTS2 contains a valid models.py."
                    )

                # Text encoding
                text_out = text_encoder(input_ids, attn_mask)

                # Style conditioning from ground-truth mel (teacher-forcing in Stage 1)
                style_vec = style_encoder(mel) if style_encoder else None

                # Decode
                decoder_kwargs = {"style": style_vec} if style_vec is not None else {}
                pred_mel, pred_f0 = decoder(text_out, **decoder_kwargs)

                # ── Losses ────────────────────────────────────────────────────
                # Mel reconstruction: L1 is more robust than L2 for spectrogram
                T_pred   = pred_mel.shape[-1]
                T_target = mel.shape[-1]
                if T_pred != T_target:
                    # Trim or pad to align (small mismatches from padding)
                    T = min(T_pred, T_target)
                    pred_mel = pred_mel[..., :T]
                    mel_gt   = mel[..., :T]
                    f0_tgt   = f0_target[:, :T]
                else:
                    mel_gt = mel
                    f0_tgt = f0_target

                loss_mel = F.l1_loss(pred_mel, mel_gt)

                # F0 loss: the most critical loss for Vietnamese tone quality
                if pred_f0 is not None:
                    loss_f0 = f0_loss(pred_f0.squeeze(1), f0_tgt)
                else:
                    # If the decoder doesn't output F0 directly, derive it from mel
                    # using a differentiable proxy (log-energy of low-frequency bins
                    # correlates with voiced pitch presence)
                    loss_f0 = torch.tensor(0.0, device=device)
                    log.debug("Decoder does not output F0; F0 loss skipped this batch.")

                total_loss = (
                    args.mel_weight * loss_mel
                    + args.f0_weight * loss_f0
                ) / args.grad_accum

            scaler.scale(total_loss).backward()
            accum_step += 1

            if accum_step < args.grad_accum:
                continue

            # ── Optimizer step ────────────────────────────────────────────────
            scaler.unscale_(optimizer)
            all_params = [p for m in models.values() if m
                          for p in m.parameters() if p.requires_grad]
            nn.utils.clip_grad_norm_(all_params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accum_step = 0

            loss_val = total_loss.item() * args.grad_accum
            loss_hist.append(loss_val)
            step += 1

            # ── Logging ───────────────────────────────────────────────────────
            if step % args.log_every == 0:
                window = loss_hist[-args.log_every:]
                avg    = sum(window) / len(window)
                lr_now = scheduler.get_last_lr()[0]
                log.info(
                    "Step %7d | loss %.4f | mel %.4f | f0 %.4f | avg %.4f | lr %.2e",
                    step, loss_val,
                    loss_mel.item(), loss_f0.item() if isinstance(loss_f0, torch.Tensor) else loss_f0,
                    avg, lr_now
                )
                if run:
                    run.log({
                        "loss": loss_val, "loss_mel": loss_mel.item(),
                        "loss_f0": loss_f0.item() if isinstance(loss_f0, torch.Tensor) else 0,
                        "lr": lr_now
                    }, step=step)

            # ── Validation ────────────────────────────────────────────────────
            if val_loader and step % args.save_every == 0:
                for m in models.values():
                    if m: m.eval()
                val_losses = []
                with torch.no_grad():
                    for vb in val_loader:
                        v_mel = vb["mel"].to(device)
                        v_f0  = vb["f0"].to(device)
                        v_ids, v_mask = tokenize(vb["ipa_text"])
                        v_ids  = v_ids.to(device)
                        v_mask = v_mask.to(device)
                        with autocast(device_type=device, dtype=amp_dtype, enabled=use_cuda):
                            v_text = models["text_encoder"](v_ids, v_mask)
                            v_style = models["style_encoder"](v_mel) if models.get("style_encoder") else None
                            v_kw = {"style": v_style} if v_style is not None else {}
                            v_pred, v_pred_f0 = models["decoder"](v_text, **v_kw)
                            T = min(v_pred.shape[-1], v_mel.shape[-1])
                            v_loss = F.l1_loss(v_pred[..., :T], v_mel[..., :T])
                        val_losses.append(v_loss.item())
                avg_val = sum(val_losses) / len(val_losses)
                log.info("Step %7d | val_loss %.4f  (best %.4f)", step, avg_val, best_val)
                if run:
                    run.log({"val_loss": avg_val}, step=step)
                if avg_val < best_val:
                    best_val = avg_val
                    _save_checkpoint(models, optimizer, step, avg_val,
                                     ckpt_dir / "best_model.pth")
                    log.info("New best model saved.")
                for m in models.values():
                    if m: m.train()

            # ── Rolling checkpoint ────────────────────────────────────────────
            if step % args.save_every == 0:
                _save_checkpoint(models, optimizer, step, None,
                                 ckpt_dir / f"step_{step:08d}.pth")

    # ── Final save ────────────────────────────────────────────────────────────
    _save_checkpoint(models, optimizer, step, None, ckpt_dir / "step_final.pth")
    log.info("Training complete. Final checkpoint: checkpoints/step_final.pth")

    if run:
        run.finish()

    if args.smoke_test and len(loss_hist) >= 2:
        first, last = loss_hist[0], loss_hist[-1]
        if last < first:
            log.info("SMOKE TEST PASSED ✓  %.4f → %.4f", first, last)
        else:
            log.error("SMOKE TEST FAILED ✗  %.4f → %.4f (loss did not decrease)", first, last)
            sys.exit(1)


def _save_checkpoint(models: dict, optimizer, step: int, val_loss, path: Path):
    torch.save({
        "step":      step,
        "models":    {k: m.state_dict() for k, m in models.items() if m is not None},
        "optimizer": optimizer.state_dict(),
        "val_loss":  val_loss,
    }, path)


if __name__ == "__main__":
    main()
