#!/usr/bin/env python3
"""
Surgically extends Kokoro base model's embedding weights and config
to cover all Vietnamese IPA phonemes dynamically found in the dataset manifest.

Fixes vs original:
  - New token embeddings initialized with kaiming_uniform_ (better gradient
    flow than mean initialization, which starts all new tokens at the centroid
    and slows early training)
  - Config vocab_size / n_vocab fields updated to match new embedding size
  - Validates that existing vocab size matches actual embedding size before surgery
  - Warns if no new symbols found (copy-only path, but still validates)
  - Extended manifest columns (path|ipa|text|speaker|dialect) handled correctly
"""
import os
import csv
import json
import logging
from pathlib import Path
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def perform_surgery(
    checkpoint_path: Path,
    config_path: Path,
    manifest_path: Path,
    output_checkpoint_path: Path,
    output_config_path: Path,
):
    # ── 1. Extract all unique IPA characters from manifest ──────────────────
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Run prepare_dataset.py first."
        )

    vi_symbols: set = set()
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="|"):
            if len(row) >= 2:
                vi_symbols.update(list(row[1]))   # column 1 is the IPA string

    log.info(
        "Extracted %d unique IPA characters from manifest: %s",
        len(vi_symbols), sorted(vi_symbols),
    )

    # ── 2. Load base config ──────────────────────────────────────────────────
    if not config_path.exists():
        raise FileNotFoundError(f"Base config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    vocab = config.get("vocab", {})
    if not vocab:
        log.warning("No 'vocab' dict found in config.json — creating empty vocab.")
        vocab = {}

    # ── 3. Find new symbols ─────────────────────────────────────────────────
    new_symbols = sorted(s for s in vi_symbols if s not in vocab)
    if not new_symbols:
        log.info("All Vietnamese symbols already in config vocab. No surgery needed.")
        # Still write copies to target paths so downstream scripts find them
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        torch.save(ckpt, output_checkpoint_path)
        with open(output_config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        log.info("Copied checkpoint and config to output paths unchanged.")
        return

    log.info("New symbols to register (%d): %s", len(new_symbols), new_symbols)

    # ── 4. Assign indices to new symbols ────────────────────────────────────
    old_vocab_size = len(vocab)
    for i, symbol in enumerate(new_symbols):
        vocab[symbol] = old_vocab_size + i
    config["vocab"] = vocab

    # Update vocab_size / n_vocab fields if they exist in the config
    for key in ("vocab_size", "n_vocab", "num_tokens"):
        if key in config:
            config[key] = len(vocab)
            log.info("Updated config['%s'] = %d", key, len(vocab))
    # Also update nested locations (e.g., config["model"]["vocab_size"])
    for sub_key in ("model", "generator", "text_encoder"):
        if isinstance(config.get(sub_key), dict):
            for key in ("vocab_size", "n_vocab", "num_tokens"):
                if key in config[sub_key]:
                    config[sub_key][key] = len(vocab)
                    log.info("Updated config['%s']['%s'] = %d", sub_key, key, len(vocab))

    # ── 5. Load checkpoint and locate embedding matrix ───────────────────────
    log.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)

    # Search for the text embedding key
    embed_key = None
    EMBED_PATTERNS = ["text_encoder.embed.weight", "embed.weight", "embedding.weight"]
    for pattern in EMBED_PATTERNS:
        if pattern in state_dict:
            embed_key = pattern
            break
    if embed_key is None:
        for k in state_dict:
            if "embed" in k.lower() or "embedding" in k.lower():
                embed_key = k
                break
    if embed_key is None:
        candidate_keys = [k for k in state_dict if k.endswith(".weight")][:20]
        raise KeyError(
            f"Could not locate text embedding key in state dict. "
            f"First weight keys: {candidate_keys}"
        )

    log.info("Text embedding key: %s", embed_key)
    old_embed = state_dict[embed_key]
    actual_old_size, embed_dim = old_embed.shape

    # Sanity check: actual embedding size should match config vocab size
    if actual_old_size != old_vocab_size:
        log.warning(
            "Config vocab size (%d) does not match actual embedding size (%d). "
            "Using actual embedding size as the authoritative base.",
            old_vocab_size, actual_old_size,
        )
        # Re-assign indices from the actual size
        for i, symbol in enumerate(new_symbols):
            vocab[symbol] = actual_old_size + i
        config["vocab"] = vocab

    new_vocab_size = actual_old_size + len(new_symbols)
    log.info(
        "Extending embedding: %d → %d tokens  (embed_dim=%d)",
        actual_old_size, new_vocab_size, embed_dim,
    )

    # ── 6. Construct new embedding matrix ────────────────────────────────────
    new_embed_weight = torch.empty(new_vocab_size, embed_dim)

    # Copy original weights exactly
    new_embed_weight[:actual_old_size] = old_embed

    # Initialize new tokens with kaiming_uniform_ (fan_in mode):
    # This gives new embeddings the same variance as the original linear layer
    # would have had, leading to better gradient flow than mean initialization.
    # We initialize only the new rows to avoid touching the existing weights.
    nn.init.kaiming_uniform_(
        new_embed_weight[actual_old_size:],
        a=math.sqrt(5),   # same default as nn.Embedding
    )

    state_dict[embed_key] = new_embed_weight

    # ── 7. Save outputs ──────────────────────────────────────────────────────
    output_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    output_config_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Saving extended checkpoint → %s", output_checkpoint_path)
    torch.save(checkpoint, output_checkpoint_path)

    log.info("Saving extended config    → %s", output_config_path)
    with open(output_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    log.info(
        "Surgery complete. Vocab: %d → %d  (%d new Vietnamese tokens).",
        actual_old_size, new_vocab_size, len(new_symbols),
    )


import math  # needed for kaiming_uniform_ math.sqrt call above


if __name__ == "__main__":
    project_dir = Path("kokoro_vietnamese")
    perform_surgery(
        checkpoint_path=project_dir / "checkpoints/kokoro-v1_1-zh.pth",
        config_path=project_dir / "config.json",
        manifest_path=project_dir / "data/train_manifest.csv",
        output_checkpoint_path=project_dir / "checkpoints/kokoro-vi-north-extended.pth",
        output_config_path=project_dir / "config_vi.json",
    )
