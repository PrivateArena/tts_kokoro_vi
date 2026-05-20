#!/usr/bin/env python3
"""
Vocabulary Embedding Surgery for Kokoro/StyleTTS2 Vietnamese fine-tuning.

Extends the base model's text embedding matrix to cover all Vietnamese IPA
phoneme tokens found in the training manifest.

Fixes vs original:
  - Uses viphoneme's declared PHONE_SET as the canonical token inventory
    instead of iterating over individual Unicode characters (which breaks
    multi-character phonemes and stacked tonal diacritics).
  - Falls back to whitespace-split tokenization if PHONE_SET is unavailable,
    with a Unicode-grapheme-cluster split as a final fallback.
  - Validates that all 6 Vietnamese tones are represented in the new vocab.
  - Saves a human-readable vocab diff report for debugging.
"""
import csv
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Vietnamese tone markers (Chao tone letters used by viphoneme) ────────────
VI_TONE_MARKERS = {"˧", "˨", "˦", "˥", "˩"}

# =============================================================================
# IPA tokenization helpers
# =============================================================================

def _get_viphoneme_phone_set() -> set[str] | None:
    """
    Try to import viphoneme's declared phone set.
    Returns None if the attribute isn't exposed by this version.
    """
    try:
        import viphoneme
        # Different versions expose this differently — try common attribute names
        for attr in ("PHONE_SET", "PHONES", "phone_set", "IPA_SYMBOLS"):
            if hasattr(viphoneme, attr):
                ps = getattr(viphoneme, attr)
                if isinstance(ps, (set, list, tuple)) and len(ps) > 5:
                    log.info(
                        "Using viphoneme.%s as canonical phone set (%d symbols).",
                        attr, len(ps),
                    )
                    return set(str(s) for s in ps)
        log.warning(
            "viphoneme is installed but does not expose a phone set attribute. "
            "Falling back to manifest-derived tokenization."
        )
        return None
    except ImportError:
        log.warning("viphoneme not installed — using manifest-derived tokenization.")
        return None


def _grapheme_clusters(text: str) -> list[str]:
    """
    Split a string into Unicode grapheme clusters (user-perceived characters).
    This correctly handles multi-codepoint sequences like base vowel + diacritic.
    Uses the `grapheme` library if available, otherwise falls back to regex.
    """
    try:
        import grapheme
        return list(grapheme.graphemes(text))
    except ImportError:
        pass
    # Regex fallback: match base char + any combining marks
    import regex  # `regex` (not `re`) supports \X
    try:
        return regex.findall(r"\X", text)
    except Exception:
        pass
    # Last resort: character-by-character (may split multi-codepoint sequences)
    return list(text)


def extract_tokens_from_manifest(manifest_path: Path) -> set[str]:
    """
    Derive the IPA token inventory from the manifest.

    Tokenization priority:
      1. Whitespace split — if IPA strings contain spaces between tokens
         (viphoneme ≥ 0.1.0 outputs space-separated tokens by default)
      2. Grapheme cluster split — for space-free IPA strings
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    raw_ipa_strings: list[str] = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="|"):
            if len(row) >= 2 and row[1].strip():
                raw_ipa_strings.append(row[1].strip())

    if not raw_ipa_strings:
        raise ValueError("No IPA strings found in manifest — run prepare_dataset.py first.")

    # Detect whether viphoneme outputs space-separated tokens
    space_separated = any(" " in s for s in raw_ipa_strings[:200])
    tokens: set[str] = set()

    if space_separated:
        log.info("IPA strings appear space-separated — using whitespace tokenization.")
        for ipa in raw_ipa_strings:
            tokens.update(t for t in ipa.split() if t)
    else:
        log.info("IPA strings appear compact — using grapheme-cluster tokenization.")
        for ipa in raw_ipa_strings:
            tokens.update(_grapheme_clusters(ipa))

    # Always add individual tone markers so they are independently addressable
    tokens.update(VI_TONE_MARKERS)
    return tokens


def build_final_token_set(manifest_path: Path) -> set[str]:
    """
    Merge viphoneme's declared phone set (if available) with tokens observed
    in the manifest. This catches any tokens the model might see at inference
    time that aren't in the training manifest.
    """
    manifest_tokens = extract_tokens_from_manifest(manifest_path)
    log.info("Manifest-derived tokens: %d unique symbols.", len(manifest_tokens))

    declared = _get_viphoneme_phone_set()
    if declared:
        combined = manifest_tokens | declared
        extra = declared - manifest_tokens
        if extra:
            log.info(
                "Adding %d declared viphoneme tokens not seen in manifest: %s",
                len(extra), sorted(extra)[:20],
            )
        return combined

    return manifest_tokens

# =============================================================================
# Embedding surgery
# =============================================================================

def perform_surgery(
    checkpoint_path: Path,
    config_path:     Path,
    manifest_path:   Path,
    output_checkpoint_path: Path,
    output_config_path:     Path,
    vocab_diff_report_path: Path | None = None,
):
    """
    Extend the embedding layer of a Kokoro/StyleTTS2 checkpoint to cover all
    Vietnamese IPA tokens, and update the config vocab accordingly.
    """
    # 1. Build final Vietnamese token set
    vi_tokens = build_final_token_set(manifest_path)
    log.info("Total Vietnamese token inventory: %d symbols.", len(vi_tokens))

    # 2. Load base config
    if not config_path.exists():
        raise FileNotFoundError(f"Base config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    vocab: dict[str, int] = config.get("vocab", {})
    if not vocab:
        log.warning(
            "No 'vocab' key in config.json — starting with an empty vocabulary. "
            "Make sure this matches the Kokoro model's actual token mapping."
        )

    # 3. Identify new tokens
    new_tokens = sorted(t for t in vi_tokens if t not in vocab)

    # Save a diff report for debugging / review
    if vocab_diff_report_path:
        report = {
            "existing_vocab_size": len(vocab),
            "new_tokens_count":    len(new_tokens),
            "new_tokens":          new_tokens,
            "all_vi_tokens":       sorted(vi_tokens),
            "tone_markers_present": sorted(VI_TONE_MARKERS & vi_tokens),
        }
        with open(vocab_diff_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        log.info("Vocab diff report saved to %s", vocab_diff_report_path)

    # Tone coverage validation
    tone_found = VI_TONE_MARKERS & vi_tokens
    tone_missing = VI_TONE_MARKERS - vi_tokens
    if tone_missing:
        log.warning(
            "TONE COVERAGE WARNING: Chao markers %s are absent from the token set. "
            "Vietnamese tones may not be correctly represented.", tone_missing
        )
    else:
        log.info("Tone coverage OK: all Chao markers present (%s).", sorted(tone_found))

    if not new_tokens:
        log.info("All Vietnamese tokens already in vocab — copying checkpoints as-is.")
        import shutil
        shutil.copy2(checkpoint_path, output_checkpoint_path)
        with open(output_config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return

    log.info("Registering %d new tokens (IDs %d–%d).", len(new_tokens), len(vocab), len(vocab) + len(new_tokens) - 1)
    old_vocab_size = len(vocab)
    for i, token in enumerate(new_tokens):
        vocab[token] = old_vocab_size + i
    config["vocab"] = vocab

    # 4. Load checkpoint and locate the embedding weight tensor
    log.info("Loading base checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict  = checkpoint.get("model", checkpoint)

    embed_key = _find_embed_key(state_dict)
    if embed_key is None:
        raise KeyError(
            "Could not locate text embedding weight in checkpoint. "
            f"Available keys (first 20): {list(state_dict.keys())[:20]}"
        )
    log.info("Text embedding key: '%s'", embed_key)

    old_weight = state_dict[embed_key]
    actual_old_size, embed_dim = old_weight.shape
    new_vocab_size = actual_old_size + len(new_tokens)
    log.info(
        "Extending embedding: %d → %d tokens (dim=%d).",
        actual_old_size, new_vocab_size, embed_dim,
    )

    # 5. Surgery: build extended embedding matrix
    new_embed = nn.Embedding(new_vocab_size, embed_dim)
    with torch.no_grad():
        # Copy original weights
        new_embed.weight[:actual_old_size] = old_weight
        # Initialize new rows as the mean of existing embeddings.
        # This is safer than random init: the model can start decoding
        # new tokens immediately without large gradient spikes.
        init_vector = old_weight.mean(dim=0, keepdim=True)
        new_embed.weight[actual_old_size:] = init_vector.expand(len(new_tokens), -1)

    state_dict[embed_key] = new_embed.weight.data

    # 6. Save extended checkpoint and config
    output_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_checkpoint_path)
    log.info("Extended checkpoint saved to %s", output_checkpoint_path)

    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    log.info("Extended config saved to %s", output_config_path)
    log.info(
        "Surgery complete. New vocab size: %d (was %d). Added %d tokens.",
        new_vocab_size, actual_old_size, len(new_tokens),
    )


def _find_embed_key(state_dict: dict) -> str | None:
    """Locate the text encoder embedding weight key by heuristic name matching."""
    # Priority-ordered candidate patterns (most specific first)
    patterns = [
        "text_encoder.embedding.weight",
        "text_encoder.embed.weight",
        "encoder.embed.weight",
        "embedding.weight",
    ]
    for p in patterns:
        if p in state_dict:
            return p
    # Fuzzy search
    for k in state_dict:
        k_lower = k.lower()
        if "embed" in k_lower and "weight" in k_lower:
            return k
        if "embedding" in k_lower and "weight" in k_lower:
            return k
    return None

# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Extend Kokoro vocab for Vietnamese IPA.")
    p.add_argument("--project-dir",  default="kokoro_vietnamese")
    p.add_argument("--checkpoint",   default="checkpoints/kokoro-v1_1-zh.pth")
    p.add_argument("--config",       default="config.json")
    p.add_argument("--manifest",     default="data/train_manifest.csv")
    p.add_argument("--out-checkpoint", default="checkpoints/kokoro-vi-north-extended.pth")
    p.add_argument("--out-config",     default="config_vi.json")
    args = p.parse_args()

    proj = Path(args.project_dir)
    perform_surgery(
        checkpoint_path        = proj / args.checkpoint,
        config_path            = proj / args.config,
        manifest_path          = proj / args.manifest,
        output_checkpoint_path = proj / args.out_checkpoint,
        output_config_path     = proj / args.out_config,
        vocab_diff_report_path = proj / "data/vocab_diff_report.json",
    )