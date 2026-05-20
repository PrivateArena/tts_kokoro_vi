Let me read all the files carefully before writing improved versions.Now I have everything. Let me write all 5 improved files.Here's what changed in each file, in order of impact:

**`run_train.py` — completely restructured (was the fatal bug)**
The old script trained a throwaway GRU and never touched the Kokoro checkpoint. The new one loads `/opt/StyleTTS2` from the Docker image and calls `build_model()` + `load_state_dict()` on your vocabulary-extended checkpoint. It falls back to a lightweight acoustic head only if the StyleTTS2 repo isn't cloned — so smoke tests work without the full repo. Other changes: mel params now match StyleTTS2 exactly (`n_fft=2048, hop=300`), speaker IDs from column 4 feed the style encoder, gradient accumulation support (batch 32 × accum 2 = effective 64), bfloat16 AMP (more stable than float16 on ROCm), cosine restart scheduler instead of one-shot cosine, and ROCm env vars set at import time.

**`prepare_dataset.py` — dialect filter and speaker whitelist re-enabled**
Both were silently returning `True`/`None`. Now `passes_dialect_filter()` actually runs the Southern/Central blocklists, and `load_verified_speakers()` actually reads the whitelist file. Other changes: LUFS loudness normalization replaces peak normalization (matches Kokoro's -23 LUFS training data), DNSMOS perceptual quality scoring replaces the RMS-proxy SNR (drops noisy audiobook recordings), the manifest now includes a `speaker_id` column (col 4), train/val split is speaker-stratified so each speaker appears in both splits proportionally, and tone coverage is validated after manifest generation.

**`extend_vocab.py` — phoneme tokenization fixed**
`list(ipa_string)` was splitting multi-codepoint IPA phonemes into broken fragments. The new code tries `viphoneme.PHONE_SET` first (the library's declared inventory), falls back to whitespace-split (which works if viphoneme outputs space-separated tokens), and finally uses Unicode grapheme clusters as a last resort. It also saves a `vocab_diff_report.json` so you can audit exactly what tokens were added.

**`get_unique_speakers.py` — cleaner progress tracking**
Uses a `Progress` dataclass instead of raw dicts, separates error files from completed files (so retries are accurate), falls back to full column scan when row-group stats are missing, and the progress line now overwrites in-place instead of flooding the terminal.

**`train.sh` — Dockerfile inlined + correct training hyperparameters**
The Dockerfile now clones StyleTTS2 from GitHub into `/opt/StyleTTS2` inside the image. Training flags updated to match the new `run_train.py` arguments (batch 32, grad-accum 2, 200k steps, per-component LRs). GTT-tuning failure is now a warning instead of a silent skip.