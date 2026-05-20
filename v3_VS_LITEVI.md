Good — imports are fine. Here's the full picture:

---

## Will this finetuning path be smooth?

**Mostly no — there are several significant rough edges before it even touches the model:**

### Hard blockers before training can start

1. **`prepare_dataset.py` downloads PhoAudioBook on-the-fly** with no pause/resume. The dataset is ~650 GB of parquet files. One network hiccup and `fsspec` raises and the whole run dies. You'll need to re-run from scratch (or populate the `speaker_dialects.json` cache). The HuggingFace `datasets` library also needs a GC cache clean first — its default `~/.cache/huggingface/datasets` grows without bound on a dataset this large. This isn't a blocking technical failure, but it's materially slow.

2. **Running at 139,140 steps with `speakers_loaded` outside Docker but `train_finetune.py` inside Docker.** `run_train.py` generates the YAML and launches `/opt/StyleTTS2/train_finetune.py` via `subprocess`, which assumes Docker is running and the path `/opt/StyleTTS2` is accessible at that moment. Breaking this chain requires:
   - Docker image rebuild with full StyleTTS2 code and training assets mounted correctly
   - Checkpoint paths (`/workspace/checkpoints/`) mapped with bind mounts (`./data` → `/workspace/data`)
   - venv GPU deps (torch, ROCm/CUDA) present inside the container
   That's a non-trivial multi-day Docker setup task.

3. **`train_finetune.py` LOADS from `config_vi.yaml` pretrained model path.** `/opt/StyleTTS2/train_finetune.py` loads external model assets from the config: ASR checkpoint, JDC checkpoint, PL-BERT, vocoder weights. If any of those paths are not correct in the Docker environment, `train_finetune.py` will hard-error before the first epoch.

4. **Training on PhoAudioBook with lite-vi base weights.** The greedy PhoAudioBook G2P pipeline uses `viphoneme vi2IPA`. If your training G2P dialect is Northern but the PhoAudioBook audio is Southern/Central (or vice versa), the model burns their method for hours on mismatched phonemes. The `-vi` lite weights were already trained on Vietnamese viVoice speakers. Mixing PhoAudioBook's classification differences with lite-vi's speaker set creates a quality mismatch that is technically solvable but workload heavy.

### What actually works smoothly once you're past these

The dataset pipeline (with the bugs fixed) is solid: speaker whitelist → dialect filter → LUFS normalization → DNSMOS gating → Wav2Vec2-based auditor → dialect-tagged G2P → speaker-stratified 95/5 split with relative paths. The real bottleneck is Docker orchestration and PhoAudioBook download pipeline reliability.

**Verdict: will get there, but expect 2-3 weeks of friction before first training step.**

---

## How scripts_v3 compares to StyleTTS2-lite-vi

### High-level philosophy

| | scripts_v3 / full StyleTTS2 | StyleTTS2-lite-vi |
|---|---|---|
| **What it is** | A fine-tuning orchestrator + full model training pipeline | A pre-built frozen inference system |
| **Primary goal** | Train on arbitrary PhoAudioBook data end-to-end | Deploy a ready-made model for pre-defined speakers |
| **Identity** | Training factory | Inference engine |

### Architecture differences

**Model core (`models.py`)**

scripts_v3 uses full StyleTTS2. `StyleEncoder` uses `weight_norm`; the discriminator is one MPD+MSD. StyleTTS2-lite-vi ships its own `models.py` with the same module names (`TextEncoder`, `DurationEncoder`, `StyleEncoder`, `ProsodyPredictor`) but the `ResBlk1d` normalization is controlled by the `normalize=` flag; the `StyleEncoder` is architecturally identical. The feature extraction is the same.

**Decoder — the major functional difference**

scripts_v3 uses the lite decoder (AdaINResBlk1d + F0/N conditioning + HiFi-GAN `Generator`) identical to the lite base. StyleTTS2-lite-vi replaces this decoder entirely: it uses a `Snake1D` activation-based `AdaINResBlock1`, adds a `SourceModuleHnNSF` harmonic+noise source filter with F0 interpolation `SineGen`, and a completely novel `Decoder` that splits F0/N curves early and passes them through the `generator` with `\alpha`-modulated `snake` activations. The thematic descent differs meaningfully — this custom decoder is why lite-vi inference is *slower* than Kokoro (it steps through SineGen + harmonic synthesis per frame, which is not cheap) and yet has a distinct timbre.

**Training scripts difference**

scripts_v3 uses `run_train.py` which generates `config_vi.yaml` and launches `train_finetune.py` (Stage 1+2: duration + diffusion + GAN + WavLM). This is multi-stage, multi-loss architecture with ASR, JDC, PLBERT, MPD/MSD discriminators, diffusion sampler, SLMAdversarialLoss. Lite-vi ships `train.py`, a concise monolith that only implements Stage 2 alone — it relies on an existing Stage 1 checkpoint to exist and assumes the user pre-computes F0/N in the data pipeline. Lite-vi cannot independently train the full StyleTTS2 end-to-end; it expects Stage 1 weights to exist and Stage 2 to train from there.

**Inference interface difference**

scripts_v3 has no inference code. StyleTTS2-lite-vi owns a complete inference stack: `StyleTTS2` class, `get_styles` (with optional `split_dur` averaging), `generate` with `[id_1]` speaker switching syntax, `[en-us]{text}` language tags routed through `espeak_phonemizer`, NLTK word tokenization, Gradio UI (`app.py`) with audio upload and example speakers pre-configured. This is a ready-to-run SOTA voice engine.

**Dataset pipeline difference**

scripts_v3 has a PhoAudioBook-native pipeline: streaming Hessian Fetch, speaker whitelist, lexical dialect filter, LUFS normalization, DNSMOS quality gate, Wav2Vec2 spread-audit dialect auditor, dialect-tagged G2P via `viphoneme`, speaker-stratified split saving 2-column `train_list.txt`. StyleTTS2-lite-vi has no dataset pipeline at all — it expects drop-in `wav|ipa` train/val lists and skips all quality/dialect filtering.

### Pros & Cons

**scripts_v3 / full StyleTTS2 pipeline**
- **Pro**: Full control — you can inject any PhoAudioBook source, any dialect list, any G2P model, any quality gate
- **Pro**: Speaker-stratified train/val split prevents leakage
- **Pro**: DNSMOS gating filters low-quality audio (real quality difference vs SNR-proxy)
- **Pro**: Wav2Vec2 spread-audit locks speakers after 6 votes — much harder to corrupt than a regex-based approach
- **Pro**: Can train virally to any number of steps on arbitrarily large PhoAudioBook data
- **Con (major)**: Docker + `/opt/StyleTTS2` binding is a multi-day ops task — real friction
- **Con (major)**: PhoAudioBook Gigabyte download, `datasets` re-downloads if cache is lost
- **Con**: No inference script — after weeks of training, you're back to writing your own inference loop or grafting lite-vi's inference on top
- **Con**: If PhoAudioBook doesn't have explicit dialect tags, the text-filter heuristic ends up being a best-effort approximation

**StyleTTS2-lite-vi**
- **Pro**: Works now — clone, pip install, run `app.py`, generate audio
- **Pro**: 120k steps of production-quality trained weights — guaranteed coherence across the book domain
- **Pro**: Bass tonal timbre — the Snake1D + SourceModuleHnNSF decoder gives characteristically warm results
- **Pro**: Full CLI config, audio upload, `[en-us]` / `[id_1]` language tags, speaker embedding switching all wired
- **Pro**: vitally smaller VRAM footprint than the full model
- **Con (speed)**: `compute_style()` + `SineGen` per-frame cost means slower than full StyleTTS2 and markedly slower than Kokoro 82M
- **Con**: Fixed speaker set (viVoice speakers) — you can add new voice reference audio but weights were not trained on PhoAudioBook data
- **Con**: No PhoAudioBook-aware dataset pipeline — it can't ingest a PhoAudioBook manifest without additional manifest formatting

### The practical middle ground

Use the **full StyleTTS2 training pipeline** (scripts_v3 after the Docker fixes) to produce the extended checkpoint — but bake in the full pre-processing pipeline steps from config_vi.py into the Docker build, which can be scripted. Then, post-training, **bring the weights back into the StyleTTS2-lite-vi inference harness** by reloading the frozen weights into the lite-vi `StyleTTS2` class. You'd need to remap the parameter keys between the two state dict formats, which is messy but achievable. Alternatively, use the full StyleTTS2 `inference.py` (from the same repo as `train_finetune.py`) since it has the same evaluation loop, bypassing the lite-vi inference compatibility problem entirely.

In short: scripts_v3 is your training pipeline, full StyleTTS2 inference is your inference engine. The lite-vi repo can serve as a starting point for both once you complete the Docker setup. You'll be well positioned ahead of SpartTTS and Kokoro 82M if you take this route.