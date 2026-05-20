#!/usr/bin/env python3
"""
Northern Vietnamese KokoroTTS — StyleTTS2 Training Controller
Acoustic pre-training & diffusion fine-tuning loop.
"""
import os
import sys
import csv
import logging
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import soundfile as sf
from transformers import AutoModel, AutoTokenizer
from torch.amp import autocast, GradScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train Northern-Vietnamese KokoroTTS head.")
    p.add_argument("--manifest",    default="data/train_manifest.csv")
    p.add_argument("--checkpoint",  default="checkpoints/kokoro-vi-north-extended.pth",
                   help="Path to base Kokoro checkpoint to load (optional).")
    p.add_argument("--resume",      default="",
                   help="Resume from a mid-run checkpoint (.pth with step/model/optimizer).")
    p.add_argument("--stage",       type=int, default=1, choices=[1, 2, 3],
                   help="Training stage (1=acoustic pre-train, 2=diffusion, 3=joint).")
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--max-steps",   type=int, default=100_000)
    p.add_argument("--lr",          type=float, default=2e-5)
    p.add_argument("--warmup-steps",type=int, default=2_000,
                   help="Linear LR warm-up steps.")
    p.add_argument("--save-every",  type=int, default=5_000,
                   help="Save a checkpoint every N steps.")
    p.add_argument("--log-every",   type=int, default=100)
    p.add_argument("--grad-clip",   type=float, default=1.0)
    p.add_argument("--smoke-test",  action="store_true",
                   help="Quick 20-step sanity check.")
    p.add_argument("--wandb",       action="store_true",
                   help="Log to Weights & Biases.")
    p.add_argument("--wandb-project", default="kokoro-vi-north")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def collate_fn(batch):
    """Pad mel sequences to the longest in the batch."""
    input_ids = torch.stack([x["input_ids"] for x in batch])
    attn_mask = torch.stack([x["attention_mask"] for x in batch])
    # mel: (1, T) → pad to max T in batch
    max_len = max(x["mel"].shape[-1] for x in batch)
    mels = torch.zeros(len(batch), 1, max_len)
    for i, x in enumerate(batch):
        t = x["mel"].shape[-1]
        mels[i, :, :t] = x["mel"]
    return {"input_ids": input_ids, "attention_mask": attn_mask, "mel": mels}


class ViNorthDataset(Dataset):
    def __init__(self, manifest_path: str, tokenizer, max_token_len: int = 256):
        self.tokenizer     = tokenizer
        self.max_token_len = max_token_len
        self.records: list[tuple[str, str]] = []

        missing = 0
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="|"):
                if len(row) < 2:
                    continue
                wav_path, ipa_text = row[0], row[1]
                if not Path(wav_path).exists():
                    missing += 1
                    continue
                self.records.append((wav_path, ipa_text))

        if missing:
            log.warning("Skipped %d manifest entries with missing WAV files.", missing)
        log.info("Dataset: %d valid samples loaded.", len(self.records))
        if not self.records:
            raise RuntimeError(f"No valid samples found in manifest: {manifest_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        wav_path, ipa_text = self.records[idx]
        try:
            audio, _ = sf.read(wav_path, dtype="float32")
        except Exception as e:
            log.error("Failed to read %s: %s", wav_path, e)
            # Return a dummy zero sample — collate_fn handles variable lengths
            audio = np.zeros(TARGET_SR, dtype=np.float32)

        tokens = self.tokenizer(
            ipa_text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_token_len,
            truncation=True,
        )
        # Store as raw waveform; mel spectrogram computed in the model forward
        mel = torch.FloatTensor(audio).unsqueeze(0)  # (1, T)
        return {
            "input_ids":      tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "mel":            mel,
        }


TARGET_SR = 24_000  # must match prepare_dataset.py


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ViNorthHead(nn.Module):
    """
    Lightweight projection head on top of XPhoneBERT for acoustic pre-training.
    CLS token → linear → mel-bin prediction.

    NOTE: This is a scaffold for Stage 1. Stages 2/3 would swap in a full
    StyleTTS2 decoder; mel_bins=80 here matches 80-band mel spectrograms.
    """
    def __init__(self, bert_hidden: int = 768, mel_bins: int = 80):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(bert_hidden, bert_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(bert_hidden, mel_bins),
        )

    def forward(self, bert_out: torch.Tensor, mel_target: torch.Tensor) -> torch.Tensor:
        # bert_out.last_hidden_state: (B, seq, hidden) → CLS token
        cls   = bert_out.last_hidden_state[:, 0, :]          # (B, hidden)
        pred  = self.proj(cls)                                # (B, mel_bins)
        # Collapse time axis of target to match prediction shape
        target = mel_target.mean(dim=-1)                      # (B, 1) → squeeze
        if target.dim() > 1:
            target = target.squeeze(1)                        # (B,) or (B, mel_bins)
        # If audio is 1-channel raw waveform mean is a scalar per sample; broadcast
        loss = nn.functional.mse_loss(pred, target.expand_as(pred))
        return loss


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------
def linear_warmup_cosine(optimizer, warmup_steps: int, total_steps: int):
    from torch.optim.lr_scheduler import LambdaLR
    import math
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(path: Path, step: int, model, optimizer, scheduler, scaler):
    torch.save({
        "step":      step,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler":    scaler.state_dict(),
    }, path)
    log.info("Checkpoint saved: %s", path)


def load_checkpoint(path: Path, model, optimizer, scheduler, scaler, device):
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return state.get("step", 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    if args.smoke_test:
        args.max_steps  = 20
        args.batch_size = 4
        args.log_every  = 5
        log.info("SMOKE TEST mode — max_steps=%d, batch_size=%d", args.max_steps, args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s  |  Stage: %d", device, args.stage)

    # ── W&B (optional) ────────────────────────────────────────────────────
    run = None
    if args.wandb:
        try:
            import wandb
            run = wandb.init(project=args.wandb_project, config=vars(args))
            log.info("W&B run: %s", run.name)
        except ImportError:
            log.warning("wandb not installed — skipping W&B logging.")

    # ── Tokenizer + encoder ───────────────────────────────────────────────
    log.info("Loading XPhoneBERT…")
    tokenizer  = AutoTokenizer.from_pretrained("vinai/xphonebert-base")
    xphonebert = AutoModel.from_pretrained("vinai/xphonebert-base").to(device)

    # Compile XPhoneBERT encoder for huge RDNA 3.5 speedups (AOTriton/FlashAttention)
    # Skip compilation in smoke test to avoid compile latency exceeding run duration
    if device == "cuda" and not args.smoke_test:
        log.info("Compiling XPhoneBERT encoder for optimized ROCm RDNA 3.5 execution…")
        try:
            xphonebert = torch.compile(xphonebert, backend="inductor", mode="reduce-overhead")
            log.info("XPhoneBERT compilation successfully initialized.")
        except Exception as e:
            log.warning("Could not compile XPhoneBERT: %s. Proceeding with eager mode.", e)

    # ── Dataset + loader ──────────────────────────────────────────────────
    dataset = ViNorthDataset(args.manifest, tokenizer)
    # Unified memory (APU) shares memory space, making pin_memory unnecessary/inefficient.
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(8, os.cpu_count() or 1),
        prefetch_factor=4,
        pin_memory=False,
        collate_fn=collate_fn,
        persistent_workers=True,   # avoid worker restart overhead each epoch
    )

    # ── Model, optimizer, scheduler, scaler ───────────────────────────────
    model     = ViNorthHead().to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(xphonebert.parameters()),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.98),   # common for transformer fine-tuning
    )
    scheduler = linear_warmup_cosine(optimizer, args.warmup_steps, args.max_steps)
    scaler    = GradScaler(enabled=(device == "cuda"))

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    # ── Resume ────────────────────────────────────────────────────────────
    start_step = 0
    if args.resume and Path(args.resume).exists():
        start_step = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler, scaler, device
        )
        log.info("Resumed from step %d", start_step)
    elif args.checkpoint and Path(args.checkpoint).exists():
        # Load weights only (no optimizer state) — transfer learning
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(
            state.get("model", state), strict=False
        )
        log.info(
            "Loaded base checkpoint %s (missing=%d, unexpected=%d)",
            args.checkpoint, len(missing), len(unexpected),
        )

    # ── Training loop ─────────────────────────────────────────────────────
    log.info("Training: steps %d → %d", start_step, args.max_steps)
    step      = start_step
    loss_hist = []

    model.train()
    xphonebert.train()

    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break

            ids  = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            mel  = batch["mel"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device, dtype=torch.bfloat16):
                bert_out = xphonebert(input_ids=ids, attention_mask=mask)
                loss     = model(bert_out, mel)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(xphonebert.parameters()),
                args.grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_val = loss.item()
            loss_hist.append(loss_val)
            step += 1

            if step % args.log_every == 0:
                window = loss_hist[-args.log_every:]
                avg    = sum(window) / len(window)
                lr_now = scheduler.get_last_lr()[0]
                log.info(
                    "Step %6d | loss %.4f | avg(%d) %.4f | lr %.2e",
                    step, loss_val, args.log_every, avg, lr_now,
                )
                if run:
                    run.log({"loss": loss_val, "loss_avg": avg, "lr": lr_now}, step=step)

            if step % args.save_every == 0:
                save_checkpoint(
                    ckpt_dir / f"step_{step:08d}.pth",
                    step, model, optimizer, scheduler, scaler,
                )

    # ── Final checkpoint ──────────────────────────────────────────────────
    save_checkpoint(
        ckpt_dir / f"step_{step:08d}_final.pth",
        step, model, optimizer, scheduler, scaler,
    )

    if run:
        run.finish()

    # ── Smoke test verdict ─────────────────────────────────────────────────
    if args.smoke_test and len(loss_hist) >= 2:
        first, last = loss_hist[0], loss_hist[-1]
        if last < first:
            log.info("SMOKE TEST PASSED ✓  loss %.4f → %.4f", first, last)
        else:
            log.error("SMOKE TEST FAILED ✗  loss did not decrease: %.4f → %.4f", first, last)
            sys.exit(1)

    log.info("Training complete.")


if __name__ == "__main__":
    main()