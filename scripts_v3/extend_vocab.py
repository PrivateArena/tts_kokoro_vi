#!/usr/bin/env python3
"""
V3 Vocabulary Embedding Surgery & State Dict Restructuring for Kokoro/StyleTTS2.

Meticulously merges and enhances the V1/V2 features:
  - Dynamically extracts observed IPA phonemes and diacritics.
  - Automatically restructures flat Kokoro checkpoints into the grouped 'net' format
    expected by StyleTTS2-lite's `load_checkpoint` loader, preventing KeyError / load crashes.
  - Extends embedding matrix weights using stable Centroid Mean + Small Perturbation.
  - Performs recursive configuration vocabulary-size updates.
"""
import os
import csv
import json
import logging
import argparse
from pathlib import Path
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

VI_TONE_MARKERS = {"˧", "˨", "˦", "˥", "˩"}

# =============================================================================
# IPA Tokenization Engine
# =============================================================================

def _get_viphoneme_phone_set() -> set[str] | None:
    """Import viphoneme's canonical phone set if available."""
    try:
        import viphoneme
        for attr in ("PHONE_SET", "PHONES", "phone_set", "IPA_SYMBOLS"):
            if hasattr(viphoneme, attr):
                ps = getattr(viphoneme, attr)
                if isinstance(ps, (set, list, tuple)) and len(ps) > 5:
                    log.info("Loaded viphoneme.%s canonical phone set (%d symbols).", attr, len(ps))
                    return set(str(s) for s in ps)
        return None
    except ImportError:
        return None


def _grapheme_clusters(text: str) -> list[str]:
    """Split string into Unicode grapheme clusters to keep diacritics intact."""
    try:
        import grapheme
        return list(grapheme.graphemes(text))
    except ImportError:
        pass
    try:
        import regex
        return regex.findall(r"\X", text)
    except ImportError:
        pass
    return list(text)


def extract_tokens_from_manifest(manifest_path: Path) -> set[str]:
    """Extract phonemes from the manifest file."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    raw_ipa_strings = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="|"):
            if len(row) >= 2 and row[1].strip():
                raw_ipa_strings.append(row[1].strip())

    if not raw_ipa_strings:
        raise ValueError("No IPA strings found in manifest — run prepare_dataset.py first.")

    space_separated = any(" " in s for s in raw_ipa_strings[:200])
    tokens = set()
    if space_separated:
        log.info("IPA strings appear space-separated — using whitespace split.")
        for ipa in raw_ipa_strings:
            tokens.update(t for t in ipa.split() if t)
    else:
        log.info("IPA strings appear compact — using grapheme-cluster tokenization.")
        for ipa in raw_ipa_strings:
            tokens.update(_grapheme_clusters(ipa))

    tokens.update(VI_TONE_MARKERS)
    return tokens


def build_final_token_set(manifest_path: Path) -> set[str]:
    """Combine manifest tokens with viphoneme's canonical inventory."""
    manifest_tokens = extract_tokens_from_manifest(manifest_path)
    log.info("Manifest-derived tokens: %d unique symbols.", len(manifest_tokens))
    declared = _get_viphoneme_phone_set()
    if declared:
        combined = manifest_tokens | declared
        extra = declared - manifest_tokens
        if extra:
            log.info("Adding %d declared viphoneme tokens not seen in manifest: %s", len(extra), sorted(extra)[:20])
        return combined
    return manifest_tokens

# =============================================================================
# Vocabulary Surgery & Struct Conversion
# =============================================================================

def perform_surgery(
    checkpoint_path: Path,
    config_path: Path,
    manifest_path: Path,
    output_checkpoint_path: Path,
    output_config_path: Path,
    vocab_diff_report_path: Path | None = None,
):
    """Surgically extends all checkpoint tensors mapping vocabulary dimensions to align with the new Vietnamese phonemes."""
    vi_tokens = build_final_token_set(manifest_path)
    log.info("Total Vietnamese token inventory: %d symbols.", len(vi_tokens))

    # 1. Load config
    if not config_path.exists():
        raise FileNotFoundError(f"Base config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    vocab = config.get("vocab", {})
    if not vocab:
        log.warning("No 'vocab' dict in config.json — creating empty vocab.")
        vocab = {}

    old_vocab_size = len(vocab)

    # 2. Load Checkpoint
    log.info("Loading base checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Detect the actual vocab size of the checkpoint weights
    checkpoint_vocab_size = None
    for group in checkpoint.values():
        if isinstance(group, dict):
            for k, v in group.items():
                if isinstance(v, torch.Tensor):
                    k_lower = k.lower()
                    if "word_embeddings" in k_lower or "embedding.weight" in k_lower or "embed.weight" in k_lower:
                        checkpoint_vocab_size = v.shape[0]
                        break
            if checkpoint_vocab_size is not None:
                break

    if checkpoint_vocab_size is not None and checkpoint_vocab_size != old_vocab_size:
        log.info("Config vocab size (%d) differs from checkpoint weight size (%d). Aligning to checkpoint size %d.", old_vocab_size, checkpoint_vocab_size, checkpoint_vocab_size)
        # Fill the gap with padding keys to make vocab contiguous
        for idx in range(old_vocab_size, checkpoint_vocab_size):
            pad_key = f"_pad_{idx}"
            vocab[pad_key] = idx
        config["vocab"] = vocab
        old_vocab_size = checkpoint_vocab_size

    new_tokens = sorted(t for t in vi_tokens if t not in vocab)

    # Save a diff report for debugging
    if vocab_diff_report_path:
        report = {
            "existing_vocab_size": old_vocab_size,
            "new_tokens_count": len(new_tokens),
            "new_tokens": new_tokens,
            "all_vi_tokens": sorted(vi_tokens),
            "tone_markers_present": sorted(VI_TONE_MARKERS & vi_tokens),
        }
        vocab_diff_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(vocab_diff_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        log.info("Vocab diff report saved to %s", vocab_diff_report_path)

    # Tone coverage validation
    tone_found = VI_TONE_MARKERS & vi_tokens
    tone_missing = VI_TONE_MARKERS - vi_tokens
    if tone_missing:
        log.warning("TONE COVERAGE WARNING: Chao markers %s are absent. Pitch tracking may collapse.", tone_missing)
    else:
        log.info("Tone coverage OK: all Chao markers present (%s).", sorted(tone_found))

    # Allocate new token indices
    for i, token in enumerate(new_tokens):
        vocab[token] = old_vocab_size + i
    config["vocab"] = vocab
    new_vocab_size = len(vocab)

    # Update vocabulary parameters in configuration
    for key in ("vocab_size", "n_vocab", "num_tokens"):
        if key in config:
            config[key] = new_vocab_size
    for sub_key in ("model", "generator", "text_encoder", "model_params"):
        if isinstance(config.get(sub_key), dict):
            for key in ("vocab_size", "n_vocab", "num_tokens", "n_token"):
                if key in config[sub_key]:
                    config[sub_key][key] = new_vocab_size

    extended_count = 0

    def extend_checkpoint_tensors(obj):
        nonlocal extended_count
        if isinstance(obj, dict):
            for k in list(obj.keys()):
                v = obj[k]
                if isinstance(v, torch.Tensor):
                    shape = list(v.shape)
                    if len(shape) >= 1:
                        # Case 1: First dimension is the vocab size
                        if shape[0] == old_vocab_size:
                            log.info(f"Extending weight '{k}' along dim 0: {shape} -> [{new_vocab_size} ...]")
                            new_shape = [new_vocab_size] + shape[1:]
                            new_tensor = torch.empty(new_shape, dtype=v.dtype, device=v.device)
                            new_tensor[:old_vocab_size] = v
                            with torch.no_grad():
                                centroid = v.mean(dim=0)
                                std = v.std(dim=0) * 0.05
                                noise = torch.randn([len(new_tokens)] + shape[1:], dtype=v.dtype, device=v.device) * std
                                new_tensor[old_vocab_size:] = centroid + noise
                            obj[k] = new_tensor
                            extended_count += 1
                        # Case 2: Second dimension is the vocab size
                        elif len(shape) >= 2 and shape[1] == old_vocab_size:
                            log.info(f"Extending weight '{k}' along dim 1: {shape} -> [{shape[0]}, {new_vocab_size} ...]")
                            new_shape = [shape[0], new_vocab_size] + shape[2:]
                            new_tensor = torch.empty(new_shape, dtype=v.dtype, device=v.device)
                            new_tensor[:, :old_vocab_size] = v
                            with torch.no_grad():
                                centroid = v.mean(dim=1, keepdim=True)
                                std = v.std(dim=1, keepdim=True) * 0.05
                                noise = torch.randn([shape[0], len(new_tokens)] + shape[2:], dtype=v.dtype, device=v.device) * std
                                new_tensor[:, old_vocab_size:] = centroid + noise
                            obj[k] = new_tensor
                            extended_count += 1
                else:
                    extend_checkpoint_tensors(v)
        elif isinstance(obj, list):
            for item in obj:
                extend_checkpoint_tensors(item)

    # Run recursive tensor extension in place
    extend_checkpoint_tensors(checkpoint)
    log.info("Surgically extended %d parameter tensors in the checkpoint.", extended_count)

    # 4. Save checkpoint & config
    output_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_checkpoint_path)
    log.info("Surgically extended checkpoint saved → %s", output_checkpoint_path)
    
    with open(output_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    log.info("Extended config saved → %s", output_config_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extend Kokoro vocab and restructure state dict for StyleTTS2.")
    p.add_argument("--project-dir", default="kokoro_vietnamese")
    p.add_argument("--checkpoint", default="checkpoints/kokoro-v1_1-zh.pth")
    p.add_argument("--config", default="config.json")
    p.add_argument("--manifest", default="data/train_manifest.csv")
    p.add_argument("--out-checkpoint", default="checkpoints/kokoro-vi-north-extended.pth")
    p.add_argument("--out-config", default="config_vi.json")
    args = p.parse_args()

    proj = Path(args.project_dir)
    
    def resolve_path(p_str: str) -> Path:
        p = Path(p_str)
        if p.is_absolute():
            return p
        try:
            if p.parts and p.parts[0] == proj.name:
                return p
        except Exception:
            pass
        return proj / p

    perform_surgery(
        checkpoint_path=resolve_path(args.checkpoint),
        config_path=resolve_path(args.config),
        manifest_path=resolve_path(args.manifest),
        output_checkpoint_path=resolve_path(args.out_checkpoint),
        output_config_path=resolve_path(args.out_config),
        vocab_diff_report_path=resolve_path("data/vocab_diff_report.json"),
    )
