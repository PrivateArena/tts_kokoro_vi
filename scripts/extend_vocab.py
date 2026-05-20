#!/usr/bin/env python3
"""
Surgically extends Kokoro base model's embedding weights and config
to cover all Vietnamese IPA phonemes dynamically found in the dataset manifest.
"""
import os
import json
import csv
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
    output_config_path: Path
):
    # 1. Load the manifest and extract all unique characters used in the phoneme representations
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Please run dataset prep first.")
    
    vi_symbols = set()
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="|"):
            if len(row) >= 2:
                # Add all characters in the IPA phoneme string
                vi_symbols.update(list(row[1]))
    
    log.info("Extracted %d unique phonemes/characters from manifest: %s", len(vi_symbols), sorted(list(vi_symbols)))

    # 2. Load the base config.json
    if not config_path.exists():
        raise FileNotFoundError(f"Base config not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # Check if 'vocab' or 'symbols' is in the config
    vocab = config.get("vocab", {})
    if not vocab:
        log.warning("No 'vocab' dict found in config.json. Creating a new one or verifying keys.")
        # Default Kokoro symbol mapping is usually defined in model code, 
        # but let's check config style. In Kokoro v1.0+, config has 'vocab' mapping characters to IDs.
        vocab = {}
    
    # 3. Find which symbols are new
    new_symbols = sorted([s for s in vi_symbols if s not in vocab])
    if not new_symbols:
        log.info("All Vietnamese symbols are already registered in config.json. No embedding extension needed.")
        # Still copy checkpoints to target destinations
        torch.save(torch.load(checkpoint_path, map_location="cpu"), output_checkpoint_path)
        with open(output_config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return

    log.info("Found %d new symbols to register: %s", len(new_symbols), new_symbols)

    # 4. Map new symbols to new indices starting from len(vocab)
    old_vocab_size = len(vocab)
    for i, symbol in enumerate(new_symbols):
        vocab[symbol] = old_vocab_size + i
    
    config["vocab"] = vocab

    # 5. Load base weights and extend text encoder embedding matrix
    log.info("Loading base checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    
    # Locate the embedding key. Typically 'text_encoder.embed.weight' in Kokoro/StyleTTS2.
    embed_key = None
    for k in state_dict.keys():
        if "embed.weight" in k or "text_encoder.embed" in k:
            embed_key = k
            break
            
    if not embed_key:
        # Fallback search
        for k in state_dict.keys():
            if "embedding" in k.lower():
                embed_key = k
                break
                
    if not embed_key:
        raise KeyError(f"Could not identify the text embedding weight key in state_dict keys: {list(state_dict.keys())[:10]}")
    
    log.info("Found text embedding layer: %s", embed_key)
    old_embed_weight = state_dict[embed_key]
    actual_old_vocab_size, embed_dim = old_embed_weight.shape
    
    new_vocab_size = actual_old_vocab_size + len(new_symbols)
    log.info("Extending embedding layer from %d -> %d channels", actual_old_vocab_size, new_vocab_size)
    
    # Surgery: Construct new embedding matrix
    new_embed = nn.Embedding(new_vocab_size, embed_dim)
    # Copy old weights
    new_embed.weight.data[:actual_old_vocab_size] = old_embed_weight
    # Initialize new weights as the average of existing ones to prevent gradient explosions
    new_embed.weight.data[actual_old_vocab_size:] = old_embed_weight.mean(dim=0, keepdim=True).expand(len(new_symbols), -1)
    
    # Put the modified weights back
    state_dict[embed_key] = new_embed.weight.data
    
    # Save the updated model and config files
    log.info("Saving extended model checkpoint to: %s", output_checkpoint_path)
    torch.save(checkpoint, output_checkpoint_path)
    
    log.info("Saving extended config to: %s", output_config_path)
    with open(output_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        
    log.info("Embedding weight surgery successfully completed.")

if __name__ == "__main__":
    # Standard paths matching Stage 4
    project_dir = Path("kokoro_vietnamese")
    perform_surgery(
        checkpoint_path=project_dir / "checkpoints/kokoro-v1_1-zh.pth",
        config_path=project_dir / "config.json",
        manifest_path=project_dir / "data/train_manifest.csv",
        output_checkpoint_path=project_dir / "checkpoints/kokoro-vi-north-extended.pth",
        output_config_path=project_dir / "config_vi.json"
    )
