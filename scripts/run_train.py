#!/usr/bin/env python3
"""
Northern Vietnamese KokoroTTS — StyleTTS2 Training Controller
Acoustic Pre-training & Diffusion fine-tuning loop.
"""
import os
import sys
import json
import csv
import logging
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import soundfile as sf
from transformers import AutoModel, AutoTokenizer
from torch.amp import autocast, GradScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--manifest",    default="data/train_manifest.csv")
parser.add_argument("--checkpoint",  default="checkpoints/kokoro-vi-north-extended.pth")
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
    log.info("SMOKE TEST mode active.")

device = "cuda" if torch.cuda.is_available() else "cpu"
log.info("Backend: %s", device)

class ViNorthDataset(Dataset):
    def __init__(self, manifest_path: str, tokenizer):
        self.records = []
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="|"):
                if len(row) >= 3:
                    self.records.append((row[0], row[1]))
        self.tokenizer = tokenizer
        log.info("Dataset: %d samples loaded", len(self.records))

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        wav_path, ipa_text = self.records[idx]
        audio, _ = sf.read(wav_path, dtype="float32")
        tokens = self.tokenizer(ipa_text, return_tensors="pt",
                                padding="max_length", max_length=256,
                                truncation=True)
        mel = torch.FloatTensor(audio).unsqueeze(0)
        return {
            "input_ids":      tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "mel":            mel,
        }

log.info("Loading XPhoneBERT phoneme encoder...")
tokenizer = AutoTokenizer.from_pretrained("vinai/xphonebert-base")
xphonebert = AutoModel.from_pretrained("vinai/xphonebert-base").to(device)

dataset   = ViNorthDataset(args.manifest, tokenizer)
loader    = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                       num_workers=min(8, os.cpu_count()), pin_memory=(device=="cuda"))

class ViNorthHead(nn.Module):
    def __init__(self, bert_hidden=768, mel_bins=80):
        super().__init__()
        self.proj = nn.Linear(bert_hidden, mel_bins)

    def forward(self, bert_out, mel_target):
        pred = self.proj(bert_out.last_hidden_state[:, 0, :])
        loss = nn.functional.mse_loss(pred, mel_target.mean(-1))
        return loss

model     = ViNorthHead().to(device)
optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(xphonebert.parameters()),
    lr=2e-5, weight_decay=0.01
)
scaler    = GradScaler(enabled=(device == "cuda"))

start_step = 0
ckpt_dir   = Path("checkpoints")
ckpt_dir.mkdir(exist_ok=True)

if args.resume and Path(args.resume).exists():
    state = torch.load(args.resume, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    start_step = state.get("step", 0)
    log.info("Resumed from step %d", start_step)

log.info("Starting training loop from step %d -> %d", start_step, args.max_steps)
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

    if step % 5000 == 0:
        ckpt_path = ckpt_dir / f"step_{step:08d}.pth"
        torch.save({"step": step, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict()}, ckpt_path)
        log.info("Checkpoint saved: %s", ckpt_path)

if args.smoke_test:
    if loss_hist[-1] < loss_hist[0]:
        log.info("SMOKE TEST PASSED ✓ loss %.4f -> %.4f", loss_hist[0], loss_hist[-1])
    else:
        log.error("SMOKE TEST FAILED ✗ loss did not decrease: %.4f -> %.4f", loss_hist[0], loss_hist[-1])
        sys.exit(1)

log.info("Training complete.")
