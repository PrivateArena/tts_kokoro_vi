#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Colab Notebook Generator for Northern Vietnamese StyleTTS2 Pipeline.
Generates a structured, cell-by-cell .ipynb file in the project root.
"""

import json
import sys
from pathlib import Path

def create_markdown_cell(content_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in content_lines]
    }

def create_code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_lines]
    }

def main():
    notebook = {
        "cells": [],
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    # 1. Title cell
    notebook["cells"].append(create_markdown_cell([
        "# 🇻🇳 Northern Vietnamese StyleTTS2 Fine-Tuning Factory (V3)",
        "",
        "Welcome to the industrial-grade, cloud-based fine-tuning orchestrator for Northern Vietnamese speech synthesis. This notebook pivots the StyleTTS2 training suite from the local development system to Google Colab, leveraging GPU-accelerated computing (T4, L4, or A100 instances) to execute large-scale, high-fidelity modeling.",
        "",
        "### Key Orchestration Highlights:",
        "1. **Google Drive Integration**: Checkpoints, logs, and preprocessed audio caches are streamed directly to Google Drive, ensuring that your training progress is persistent and **100% resumable** across Colab session disconnections.",
        "2. **Resilient Sidecar Progress Caching**: Preprocessing scans hashes. If interrupted, the dataset preparing cell skips processed files immediately, bypassing audio gating (DNSMOS, LUFS) and G2P.",
        "3. **Zero-Hurdle Compatibility**: Auto-detects dependencies and dynamically patches `vinorm`'s deprecated `imp` import to prevent failures.",
        "4. **Exact Mathematical Token-Alignment**: Automatically matches the model's text-encoder embedding layer to your dialect-expanded vocabulary size, preventing runtime dimension crashes."
    ]))

    # 2. Mount G-Drive cell
    notebook["cells"].append(create_markdown_cell([
        "## Step 1: Connect Google Drive",
        "We mount Google Drive to persist all processed audio, training configs, checkpoints, and cached metadata securely."
    ]))
    notebook["cells"].append(create_code_cell([
        "from google.colab import drive",
        "drive.mount('/content/drive')",
        "",
        "# Define persistent directory in your Google Drive",
        "import os",
        "DRIVE_DIR = '/content/drive/MyDrive/tts_kokoro_vi'",
        "os.makedirs(DRIVE_DIR, exist_ok=True)",
        "print(f'Persistent workspace established at: {DRIVE_DIR}')"
    ]))

    # 3. Clone repository cell
    notebook["cells"].append(create_markdown_cell([
        "## Step 2: Clone the Workspace Repository",
        "We clone the public workspace repository `PrivateArena/tts_kokoro_vi` into our active Colab environment."
    ]))
    notebook["cells"].append(create_code_cell([
        "!git clone https://github.com/PrivateArena/tts_kokoro_vi.git /content/tts_kokoro_vi",
        "%cd /content/tts_kokoro_vi",
        "",
        "# Verify repository files are present",
        "!ls -la"
    ]))

    # 4. Dependency cell
    notebook["cells"].append(create_markdown_cell([
        "## Step 3: Install System & Python Dependencies",
        "We install standard Linux audio processors (`ffmpeg`, `sox`, `libsndfile1`) and install the exact Python library ecosystem.",
        "Crucially, we **programmatically patch `vinorm`** on-the-fly to replace the deprecated standard library `imp` module, ensuring compatibility regardless of Colab's active Python version."
    ]))
    notebook["cells"].append(create_code_cell([
        "# 1. Install system audio tools",
        "!apt-get update -qq && apt-get install -y -qq ffmpeg sox libsndfile1 build-essential espeak-ng",
        "",
        "# 2. Install Python core libraries and custom NLP packages",
        "!pip install --no-cache-dir \\",
        "    datasets \\",
        "    soundfile \\",
        "    librosa \\",
        "    pyloudnorm \\",
        "    vinorm \\",
        "    underthesea \\",
        "    viphoneme \\",
        "    phonemizer \\",
        "    grapheme \\",
        "    regex \\",
        "    eng-to-ipa \\",
        "    transformers>=4.40.0 \\",
        "    accelerate \\",
        "    tqdm \\",
        "    wandb \\",
        "    PyYAML \\",
        "    onnxruntime \\",
        "    scipy \\",
        "    matplotlib \\",
        "    munch \\",
        "    einops \\",
        "    einops-exts \\",
        "    pydub \\",
        "    nltk",
        "",
        "# 3. Programmatic vinorm/viphoneme compatibility patch",
        "import os",
        "import importlib.util",
        "",
        "# Locate vinorm without importing it first (which fails on Python 3.12+ due to imp)",
        "spec = importlib.util.find_spec('vinorm')",
        "if spec is not None and spec.origin is not None:",
        "    init_file = spec.origin",
        "    with open(init_file, 'r', encoding='utf-8') as f:",
        "        content = f.read()",
        "    if 'import imp' in content:",
        "        print('Patching vinorm for Python 3.12+ (imp module removal) compatibility ...')",
        "        content = content.replace('import imp', '')",
        "        content = content.replace(\"A=imp.find_module('vinorm')[1]\", \"A=os.path.dirname(os.path.abspath(__file__))\")",
        "        with open(init_file, 'w', encoding='utf-8') as f:",
        "            f.write(content)",
        "        print('Patch applied successfully!')",
        "    else:",
        "        print('vinorm is already compatible or patched.')",
        "else:",
        "    print('Error: Could not locate vinorm package in sys.path.')",
        "",
        "# Test G2P & Normalization on import",
        "import vinorm",
        "import viphoneme",
        "from scripts_v3.prepare_dataset import to_ipa",
        "print('Normalization & Phonemizer test: [xin chào] ->', to_ipa('xin chào'))"
    ]))

    # 5. Hugging Face Login cell
    notebook["cells"].append(create_markdown_cell([
        "## Step 4: Hugging Face Authentication",
        "The PhoAudioBook dataset is gated. Run this cell to authenticate with Hugging Face so the streaming preprocessor can fetch the repository."
    ]))
    notebook["cells"].append(create_code_cell([
        "from huggingface_hub import notebook_login",
        "notebook_login()"
    ]))

    # 6. Download Microsoft DNSMOS ONNX cell
    notebook["cells"].append(create_markdown_cell([
        "## Step 5: High-Quality Audio Filter (DNSMOS ONNX Model)",
        "For maximum perceptual speech synthesis quality, we download the official Microsoft DNSMOS ONNX model. If not present, the preprocessing pipeline will fallback to a coarser RMS-proxy SNR, which is less ideal."
    ]))
    notebook["cells"].append(create_code_cell([
        "import urllib.request",
        "os.makedirs('kokoro_vietnamese/models', exist_ok=True)",
        "dnsmos_path = 'kokoro_vietnamese/models/dnsmos_p835.onnx'",
        "if not os.path.exists(dnsmos_path):",
        "    print('Downloading Microsoft DNSMOS ONNX model ...')",
        "    url = 'https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx'",
        "    # Using backup location or direct mirror if needed, or downloading custom model",
        "    # For simplicity, we download standard p835 reference model",
        "    try:",
        "        urllib.request.urlretrieve(url, dnsmos_path)",
        "        print('DNSMOS model placed at:', dnsmos_path)",
        "    except Exception as e:",
        "        print('Direct github download failed. Preprocessor will fall back to SNR filtering safely.')",
        "else:",
        "    print('DNSMOS model already exists.')"
    ]))

    # 8. Step 6: Preprocessing cell
    notebook["cells"].append(create_markdown_cell([
        "## Step 6: Stream & Filter the Dataset (PhoAudioBook)",
        "We execute the optimized dataset preprocessing pipeline (`prepare_dataset.py`). This generates the audio WAV files and constructs `train_manifest.csv` containing all IPA phonemic transcriptions.",
        "",
        "### Drive Integration Mechanism:",
        "To protect your progress, we **symlink your Google Drive workspace** to `kokoro_vietnamese/data`. All downloaded parquets, output WAV files, train/val lists, and `processed_records.json` progress markers will be written directly to your Drive.",
        "If Colab crashes or the runtime disconnects, simply re-run Step 1 and Step 6. The preprocessor will **instantly resume** exactly where it left off, avoiding redundant downloads and processing time!"
    ]))
    notebook["cells"].append(create_code_cell([
        "# 1. Symlink Colab data directory to Google Drive for persistent checkpoints",
        "import os",
        "from pathlib import Path",
        "",
        "local_data_path = Path('/content/tts_kokoro_vi/kokoro_vietnamese/data')",
        "drive_data_path = Path(DRIVE_DIR) / 'data'",
        "os.makedirs(drive_data_path, exist_ok=True)",
        "",
        "# Backup existing local template verified northern speakers list if exists",
        "if os.path.exists(local_data_path / 'verified_northern_speakers.txt') and not os.path.exists(drive_data_path / 'verified_northern_speakers.txt'):",
        "    import shutil",
        "    shutil.copy(local_data_path / 'verified_northern_speakers.txt', drive_data_path / 'verified_northern_speakers.txt')",
        "",
        "# Remove local data dir and symlink to Google Drive",
        "if local_data_path.exists() and not local_data_path.is_symlink():",
        "    !rm -rf {local_data_path}",
        "if not local_data_path.exists():",
        "    !ln -s {drive_data_path} {local_data_path}",
        "    print('Symlink to Google Drive active!')",
        "",
        "# 2. Run dataset preprocessing with smoke-test option",
        "# To run the full dataset pre-processing, omit the --smoke-test flag.",
        "smoke_test = True #@param {type:\"boolean\"}",
        "smoke_flag = \"--smoke-test\" if smoke_test else \"\"",
        "",
        "# Launch preprocessor",
        "os.environ['DATA_ROOT'] = 'kokoro_vietnamese/data'",
        "!python3 scripts_v3/prepare_dataset.py --data-root kokoro_vietnamese/data {smoke_flag}"
    ]))

    # 7. Step 7: Vocabulary Surgery cell
    notebook["cells"].append(create_markdown_cell([
        "## Step 7: Vocabulary Surgery & Base Checkpoint Fetching",
        "Now that `train_manifest.csv` has been generated, we download the multilingual Chinese/English Kokoro baseline and run surgical embedding adaptations, expanding the model's text encoder to align perfectly with all Northern Vietnamese IPA symbols and tones found in the dataset."
    ]))
    notebook["cells"].append(create_code_cell([
        "# 1. Download base pth & config from HuggingFace",
        "os.makedirs('kokoro_vietnamese/checkpoints', exist_ok=True)",
        "if not os.path.exists('kokoro_vietnamese/checkpoints/kokoro-v1_1-zh.pth'):",
        "    !curl -L -o kokoro_vietnamese/checkpoints/kokoro-v1_1-zh.pth https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/kokoro-v1_1-zh.pth",
        "if not os.path.exists('kokoro_vietnamese/config.json'):",
        "    !curl -L -o kokoro_vietnamese/config.json https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/config.json",
        "",
        "# 2. Extract unique speakers list if not already present",
        "if not os.path.exists('kokoro_vietnamese/data/unique_speakers.txt'):",
        "    print('Extracting speaker metadata ...')",
        "    !python3 scripts_v3/get_unique_speakers.py",
        "",
        "# Setup Northern Vietnamese speakers template if missing",
        "verified_file = 'kokoro_vietnamese/data/verified_northern_speakers.txt'",
        "if not os.path.exists(verified_file):",
        "    with open(verified_file, 'w', encoding='utf-8') as f:",
        "        f.write('\\n'.join([",
        "            '# Verified Northern Vietnamese Speakers Whitelist',",
        "            'Nguyễn_Văn_Khỏa',",
        "            'Lê_Đức_Quân',",
        "            'Diễm_Hân',",
        "            'Mai_Anh',",
        "            'Đông_Quân',",
        "            'Thoan_Đây',",
        "            'Ngọc_Diễm',",
        "            'Thanh_Thủy',",
        "            'Vy_Vy',",
        "            'Tuấn_Anh'",
        "        ]))",
        "",
        "# 3. Execute surgical embedding surgery",
        "import subprocess",
        "subprocess.run(['python3', 'scripts_v3/extend_vocab.py', '--project-dir', 'kokoro_vietnamese'], check=True)",
        "print('Vocabulary surgery completed successfully!')"
    ]))

    # 9. Step 8: Clone & Compile StyleTTS2 cell
    notebook["cells"].append(create_markdown_cell([
        "## Step 8: Download & Compile StyleTTS2 Architecture",
        "Fine-tuning requires StyleTTS2's core architecture and their Cython `monotonic_align` timeline library. We clone StyleTTS2 to `/opt/StyleTTS2` and compile the Cython code in-place."
    ]))
    notebook["cells"].append(create_code_cell([
        "# 1. Clean up legacy or incomplete StyleTTS2 folders",
        "if os.path.exists('/content/StyleTTS2') and not os.path.exists('/content/StyleTTS2/train_finetune.py'):",
        "    print('Cleaning up incomplete StyleTTS2 clone ...')",
        "    !rm -rf /content/StyleTTS2",
        "",
        "# 2. Clone StyleTTS2 if missing",
        "if not os.path.exists('/content/StyleTTS2'):",
        "    print('Cloning StyleTTS2 repository ...')",
        "    !git clone --depth 1 https://github.com/yl4579/StyleTTS2.git /content/StyleTTS2",
        "",
        "# 3. Clean up any local monotonic_align folder inside StyleTTS2 to prevent package import collision",
        "if os.path.exists('/content/StyleTTS2/monotonic_align'):",
        "    print('Cleaning up local monotonic_align folder to use global package ...')",
        "    !rm -rf /content/StyleTTS2/monotonic_align",
        "",
        "# 4. Install and compile monotonic_align globally via pip",
        "print('Installing monotonic_align globally ...')",
        "!pip install git+https://github.com/resemble-ai/monotonic_align.git",
        "",
        "# 5. Create symlink at /opt/StyleTTS2 for run_train.py compatibility",
        "# Remove any legacy real directory to prevent collision",
        "if os.path.exists('/opt/StyleTTS2') and not os.path.islink('/opt/StyleTTS2'):",
        "    print('Cleaning up legacy real /opt/StyleTTS2 directory to prevent collision ...')",
        "    !rm -rf /opt/StyleTTS2",
        "if not os.path.exists('/opt/StyleTTS2'):",
        "    !ln -s /content/StyleTTS2 /opt/StyleTTS2",
        "    print('Created symlink /opt/StyleTTS2 -> /content/StyleTTS2')",
        "",
        "# 6. Programmatically patch StyleTTS2's train_finetune.py to disable weights_only for PyTorch 2.6+ compatibility",
        "train_file = '/content/StyleTTS2/train_finetune.py'",
        "if os.path.exists(train_file):",
        "    with open(train_file, 'r', encoding='utf-8') as f:",
        "        code = f.read()",
        "    if 'weights_only=False' not in code and 'torch.load =' not in code:",
        "        print('Patching train_finetune.py for PyTorch 2.6+ compatibility ...')",
        "        patch = (",
        "            'import torch\\n'",
        "            'try:\\n'",
        "            '    _orig_load = torch.load\\n'",
        "            '    torch.load = lambda *args, **kwargs: _orig_load(*args, **{**kwargs, \\'weights_only\\': False})\\n'",
        "            'except Exception:\\n'",
        "            '    pass\\n\\n'",
        "        )",
        "        with open(train_file, 'w', encoding='utf-8') as f:",
        "            f.write(patch + code)",
        "        print('PyTorch 2.6+ compatibility patch applied successfully!')",
        "",
        "# 7. Programmatically patch StyleTTS2's models.py to fallback if 'net' key is missing (allowing loading of standard state_dicts)",
        "models_file = '/content/StyleTTS2/models.py'",
        "if os.path.exists(models_file):",
        "    with open(models_file, 'r', encoding='utf-8') as f:",
        "        code = f.read()",
        "    if \"params = state.get('net', state)\" not in code:",
        "        print(\"Patching models.py to support loading raw state_dicts...\")",
        "        code = code.replace(\"params = state['net']\", \"params = state.get('net', state)\")",
        "        code = code.replace(\"optimizer.load_state_dict(state['optimizer'])\", \"if 'optimizer' in state: optimizer.load_state_dict(state['optimizer'])\")",
        "        code = code.replace(\"epoch = state['epoch']\", \"epoch = state.get('epoch', 0)\")",
        "        code = code.replace(\"iters = state['iters']\", \"iters = state.get('iters', 0)\")",
        "        with open(models_file, 'w', encoding='utf-8') as f:",
        "            f.write(code)",
        "        print('models.py standard state_dict fallback patch applied successfully!')",
        "",
        "# 8. Programmatically update PLBERT config's vocab_size to match our extended vocabulary size",
        "plbert_config_file = '/content/StyleTTS2/Utils/PLBERT/config.yml'",
        "config_json_file = '/content/tts_kokoro_vi/kokoro_vietnamese/config_vi.json'",
        "if not os.path.exists(config_json_file):",
        "    config_json_file = '/content/tts_kokoro_vi/config_vi.json'",
        "if os.path.exists(plbert_config_file) and os.path.exists(config_json_file):",
        "    with open(config_json_file, 'r', encoding='utf-8') as f:",
        "        config_vi = json.load(f)",
        "    new_vocab_size = len(config_vi.get('vocab', {}))",
        "    with open(plbert_config_file, 'r', encoding='utf-8') as f:",
        "        plbert_code = f.read()",
        "    import yaml",
        "    plbert_data = yaml.safe_load(plbert_code)",
        "    if plbert_data and 'model_params' in plbert_data:",
        "        plbert_data['model_params']['vocab_size'] = new_vocab_size",
        "        with open(plbert_config_file, 'w', encoding='utf-8') as f:",
        "            yaml.dump(plbert_data, f, default_flow_style=False)",
        "        print(f'Successfully updated PLBERT vocab_size to {new_vocab_size} in config.yml!')",
        "",
        "# 9. Programmatically patch StyleTTS2's Utils/PLBERT/util.py to filter out shape-mismatched parameters before load_state_dict",
        "plbert_util_file = '/content/StyleTTS2/Utils/PLBERT/util.py'",
        "if os.path.exists(plbert_util_file):",
        "    with open(plbert_util_file, 'r', encoding='utf-8') as f:",
        "        code = f.read()",
        "    if 'shape-mismatch' not in code and 'bert.state_dict()' not in code:",
        "        print('Patching PLBERT util.py to skip shape-mismatched parameters ...')",
        "        # We insert a shape check filter right before bert.load_state_dict(new_state_dict, strict=False)",
        "        target = 'bert.load_state_dict(new_state_dict, strict=False)'",
        "        replacement = (",
        "            '# Shape-mismatch filter to prevent RuntimeError on extended vocab\\n'",
        "            '    for k in list(new_state_dict.keys()):\\n'",
        "            '        if k in bert.state_dict():\\n'",
        "            '            if new_state_dict[k].shape != bert.state_dict()[k].shape:\\n'",
        "            '                print(f\"[PLBERT Patch] Skipping parameter {k} due to shape mismatch: {new_state_dict[k].shape} vs {bert.state_dict()[k].shape}\")\\n'",
        "            '                del new_state_dict[k]\\n'",
        "            '    ' + target",
        "        )",
        "        code = code.replace(target, replacement)",
        "        with open(plbert_util_file, 'w', encoding='utf-8') as f:",
        "            f.write(code)",
        "        print('PLBERT shape mismatch self-healing patch applied successfully!')"
    ]))

    # 10. Step 9: Launch training cell
    notebook["cells"].append(create_markdown_cell([
        "## Step 9: Launch Fine-Tuning Training Loop",
        "We execute `run_train.py`. The coordinator script will:",
        "* Automatically construct the full `config_vi.yaml` matching all adversarial parameters.",
        "* Bind Colab-absolute container directories to StyleTTS2's dependencies.",
        "* Run embedding surgery token validation.",
        "* Initiate the multi-task loss adversarial optimization loop.",
        "",
        "Checkpoints are saved inside `/content/drive/MyDrive/tts_kokoro_vi/checkpoints` persistently, enabling resumption if the cell gets disconnected!"
    ]))
    notebook["cells"].append(create_code_cell([
        "# Self-healing check: Move datasets from root to project subfolder if they were preprocessed there",
        "import shutil",
        "if os.path.exists('data') and not os.path.exists('kokoro_vietnamese/data'):",
        "    print('Self-healing: Moving data directory from root to kokoro_vietnamese/data ...')",
        "    !mv data kokoro_vietnamese/",
        "",
        "# Symlink checkpoints folder to Google Drive so checkpoints persist across restarts",
        "local_ckpt_path = Path('/content/tts_kokoro_vi/kokoro_vietnamese/checkpoints')",
        "drive_ckpt_path = Path(DRIVE_DIR) / 'checkpoints'",
        "os.makedirs(drive_ckpt_path, exist_ok=True)",
        "",
        "if local_ckpt_path.exists() and not local_ckpt_path.is_symlink():",
        "    # Copy existing files to drive if any",
        "    !cp -r kokoro_vietnamese/checkpoints/* {drive_ckpt_path}/ 2>/dev/null || true",
        "    !rm -rf {local_ckpt_path}",
        "if not local_ckpt_path.exists():",
        "    !ln -s {drive_ckpt_path} {local_ckpt_path}",
        "    print('Symlinked checkpoints to Google Drive!')",
        "",
        "# Copy training execution scripts to the workspace expected paths",
        "!mkdir -p kokoro_vietnamese/scripts",
        "!cp scripts_v3/run_train.py kokoro_vietnamese/scripts/run_train.py",
        "!cp scripts_v3/extend_vocab.py kokoro_vietnamese/scripts/extend_vocab.py",
        "",
        "# Copy OOD texts if missing",
        "!mkdir -p /opt/StyleTTS2/Data",
        "!cp kokoro_vietnamese/data/val_list.txt /opt/StyleTTS2/Data/OOD_texts.txt 2>/dev/null || echo 'No val list yet'",
        "",
        "# Enable Weights & Biases tracking (Optional)",
        "use_wandb = False #@param {type:\"boolean\"}",
        "wandb_flag = \"\" if use_wandb else \"--bypass-wandb\"",
        "",
        "# Launch training via V3 Orchestrator",
        "# Runs a fast smoke-test or launches standard 50-epoch training",
        "smoke_test_train = True #@param {type:\"boolean\"}",
        "train_flag = \"--smoke-test --batch-size 2\" if smoke_test_train else \"--batch-size 16 --epochs 50\"",
        "",
        "# Execute",
        "import os",
        "import subprocess",
        "os.environ['PYTHONPATH'] = '/opt/StyleTTS2:/opt/StyleTTS2/monotonic_align:' + os.environ.get('PYTHONPATH', '')",
        "cmd = [",
        "    'python3', 'kokoro_vietnamese/scripts/run_train.py',",
        "    '--save-every', '1',",
        "    '--log-every', '10',",
        "    '--project-dir', 'kokoro_vietnamese'",
        "]",
        "cmd.extend(train_flag.split())",
        "print('Launching command:', ' '.join(cmd))",
        "try:",
        "    result = subprocess.run(cmd, capture_output=True, text=True, check=True)",
        "    print(result.stdout)",
        "except subprocess.CalledProcessError as e:",
        "    print('--- STDOUT ---')",
        "    print(e.stdout)",
        "    print('--- STDERR ---')",
        "    print(e.stderr)",
        "    raise e"
    ]))

    # Write notebook file
    out_path = Path("vietnamese_styletts2_colab.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"Jupyter Notebook successfully created at: {out_path.resolve()}")

if __name__ == "__main__":
    main()
