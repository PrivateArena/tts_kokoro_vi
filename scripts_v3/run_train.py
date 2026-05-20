#!/usr/bin/env python3
"""
V3 Training Coordinator for StyleTTS2 Vietnamese Fine-Tuning.

Meticulously merges the best features of V1 and V2:
  - Dynamically constructs a 100% correct `config_vi.yaml` from `config_vi.json`.
  - Mathematically aligns the YAML's `symbol` list with the extended vocabulary indices,
    preventing character mapping mismatches.
  - Generates exact data paths, preprocessing, model, and loss configurations.
  - Automatically handles the official StyleTTS2 training script (/opt/StyleTTS2/train.py)
    execution inside the GPU/APU-accelerated Docker environment.
"""
import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

def generate_yaml_config(
    config_json_path: Path,
    output_yaml_path: Path,
    batch_size: int,
    epochs: int,
    save_freq: int,
    log_interval: int,
    project_dir: Path,
):
    """Generate StyleTTS2-lite compatible YAML config from config_vi.json."""
    if not config_json_path.exists():
        raise FileNotFoundError(f"Config JSON not found: {config_json_path}")
    
    with open(config_json_path, encoding="utf-8") as f:
        config_vi = json.load(f)

    # ── Mathematical Symbol Alignment ────────────────────────────────────────
    # Sort the vocabulary by its index to ensure the created symbol list
    # matches the exact row order in the embedding matrix.
    vocab = config_vi.get("vocab", {})
    sorted_vocab = sorted(vocab.items(), key=lambda item: item[1])
    sorted_keys = [item[0] for item in sorted_vocab]

    if not sorted_keys:
        raise ValueError("Vocabulary in config_vi.json is empty!")

    # Locate the padding key (usually '$' at index 0)
    pad_symbol = sorted_keys[0]
    extend_symbols = "".join(sorted_keys[1:])

    # ── Construct Config Dictionary ──────────────────────────────────────────
    config_yaml = {
        "log_dir": str(project_dir / "checkpoints" / "Finetune"),
        "save_freq": save_freq,
        "log_interval": log_interval,
        "device": "cuda",
        "epochs": epochs,
        "batch_size": batch_size,
        "max_len": 500,
        "pretrained_model": str(project_dir / "checkpoints" / "kokoro-vi-north-extended.pth"),
        "second_stage_load_pretrained": True, # Required for Stage 2 fine-tuning
        "load_only_params": True,
        "debug": False,
        
        # Absolute paths pointing inside the container's /opt/StyleTTS2
        "F0_path": "/opt/StyleTTS2/Utils/JDC/bst.t7",
        "ASR_config": "/opt/StyleTTS2/Utils/ASR/config.yml",
        "ASR_path": "/opt/StyleTTS2/Utils/ASR/epoch_00080.pth",
        "PLBERT_dir": "/opt/StyleTTS2/Utils/PLBERT/",
        
        "data_params": {
            "train_data": str(project_dir / "data" / "train_list.txt"),
            "val_data": str(project_dir / "data" / "val_list.txt"),
            "root_path": str(project_dir) + "/",
            "OOD_data": "/opt/StyleTTS2/Data/OOD_texts.txt", # Container OOD path
            "min_length": 50
        },
        
        "symbol": {
            "pad": pad_symbol,
            "punctuation": "",
            "letters": "",
            "letters_ipa": "",
            "extend": extend_symbols
        },
        
        "preprocess_params": {
            "sr": 24000,
            "spect_params": {
                "n_fft": 2048,
                "win_length": 1200,
                "hop_length": 300
            }
        },
        
        "training_strats": {
            "freeze_modules": [""],
            "ignore_modules": [""]
        },
        
        "model_params": {
            "multispeaker": True,
            "dim_in": 64,
            "hidden_dim": 512,
            "max_conv_dim": 512,
            "n_layer": 3,
            "n_mels": 80,
            "n_token": len(vocab), # Crucial mathematical alignment of n_token!
            "max_dur": 50,
            "style_dim": 128,
            "dropout": 0.2,
            "ASR_params": {
                "input_dim": 80,
                "hidden_dim": 256,
                "n_layers": 6,
                "token_embedding_dim": 512
            },
            "JDC_params": {
                "num_class": 1,
                "seq_len": 192
            },
            "decoder": {
                "type": "hifigan",
                "resblock_kernel_sizes": [3, 7, 11],
                "upsample_rates": [10, 5, 3, 2],
                "upsample_initial_channel": 512,
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "upsample_kernel_sizes": [20, 10, 6, 4]
            },
            "slm": {
                "model": "microsoft/wavlm-base-plus",
                "sr": 16000,
                "hidden": 768,
                "nlayers": 13,
                "initial_channel": 64
            },
            "diffusion": {
                "embedding_mask_proba": 0.1,
                "transformer": {
                    "num_layers": 3,
                    "num_heads": 8,
                    "head_features": 64,
                    "multiplier": 2
                },
                "dist": {
                    "sigma_data": 0.2,
                    "estimate_sigma_data": True,
                    "mean": -3.0,
                    "std": 1.0
                }
            }
        },
        
        "loss_params": {
            "lambda_mel": 5.0,
            "lambda_gen": 1.0,
            "lambda_mono": 1.0,
            "lambda_s2s": 1.0,
            "lambda_F0": 1.0,
            "lambda_norm": 1.0,
            "lambda_dur": 1.0,
            "lambda_ce": 20.0,
            "lambda_slm": 1.0,  # SLM reconstruction loss
            "lambda_sty": 1.0,  # Style reconstruction loss
            "lambda_diff": 1.0, # Diffusion loss
            "diff_epoch": 10,
            "joint_epoch": 30
        },
        
        "optimizer_params": {
            "lr": 0.0001,
            "bert_lr": 0.00001,
            "ft_lr": 0.0001
        },
        
        "slmadv_params": {
            "min_len": 400,
            "max_len": 500,
            "batch_percentage": 0.5,
            "iter": 10,
            "thresh": 5,
            "scale": 0.01,
            "sig": 1.5
        }
    }

    # ── Write YAML file ──────────────────────────────────────────────────────
    import yaml
    output_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config_yaml, f, default_flow_style=False, allow_unicode=True)
    log.info("Dynamically generated config_vi.yaml written to %s", output_yaml_path)


def main():
    parser = argparse.ArgumentParser(description="V3 Training Controller for StyleTTS2 Vietnamese.")
    parser.add_argument("--project-dir", default="/workspace")
    parser.add_argument("--manifest", default="/workspace/data/train_manifest.csv")
    parser.add_argument("--val-manifest", default="/workspace/data/val_manifest.csv")
    parser.add_argument("--checkpoint", default="/workspace/checkpoints/kokoro-vi-north-extended.pth")
    parser.add_argument("--config", default="/workspace/config_vi.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    # Smoke test limits
    batch_size = 2 if args.smoke_test else args.batch_size
    epochs = 1 if args.smoke_test else args.epochs

    log.info("Starting V3 Training Controller...")

    project_dir = Path(args.project_dir).resolve()
    
    # Resolve paths relative to project-dir if they are default /workspace paths
    def resolve_p(val, rel_suffix):
        if val.startswith("/workspace"):
            return project_dir / rel_suffix
        return Path(val)

    config_vi_json = resolve_p(args.config, "config_vi.json")
    output_yaml = project_dir / "Configs" / "config_vi.yaml"

    # Generate the Yaml config
    generate_yaml_config(
        config_vi_json,
        output_yaml,
        batch_size=batch_size,
        epochs=epochs,
        save_freq=args.save_every,
        log_interval=args.log_every,
        project_dir=project_dir,
    )

    # ── Verify and Launch official train.py ──────────────────────────────────
    official_train_py = Path("/opt/StyleTTS2/train_finetune.py")
    if not official_train_py.exists():
        log.error("Official StyleTTS2 train_finetune.py not found at /opt/StyleTTS2/train_finetune.py!")
        sys.exit(1)

    log.info("Launching official StyleTTS2 training script inside Docker environment...")
    cmd = [
        "python3",
        str(official_train_py),
        "--config_path",
        str(output_yaml)
    ]
    
    # Run the official training script from its repository directory to resolve imports and asset paths smoothly
    try:
        subprocess.run(cmd, check=True, cwd="/opt/StyleTTS2")
    except subprocess.CalledProcessError as e:
        log.error("Training script failed with exit status %d", e.returncode)
        sys.exit(e.returncode)

    log.info("Training stage completed successfully!")


if __name__ == "__main__":
    main()
