## Research analysis.

It's another "success" attempt to finetune Kokoro-English to Vietnamese: https://huggingface.co/dangtr0408/StyleTTS2-lite-vi
- It uses viVoice
- It uses Kokoro English Base model - We use Kokoro ZH Chinese Base Model because both Vietnamese and Chinese are tonal language
- It uses both Northern and Southern audio
file:///media/jang/home/Deve/tts_kokoro_vi/failure_point_comparison_dangtr0408_vs_ours.svg

---

## Critical failure points

### 1. The training objective is not TTS training

This is the most fundamental gap. Their `train.py` runs a complete **StyleTTS2 adversarial loop** — mel spectrogram loss, F0 (pitch) loss, duration loss, CE alignment loss, monotonic alignment loss, speech-to-speech loss, and a full GAN discriminator pair (MPD + MSD). It is genuinely fine-tuning an acoustic decoder that learns to produce speech.

Our `run_train.py` trains a `Linear → GELU → Dropout → Linear` head projecting XPhoneBERT's CLS token to the *mean* of a raw waveform, using MSE loss. This does not train any Kokoro decoder. It will produce a loss that goes down — but it has no path to actually generating speech. Essentially, our training loop is a classifier stub, not a TTS fine-tuner.

**What needs to happen instead:** The actual Kokoro/StyleTTS2 acoustic decoder, predictor, and discriminators need to be loaded and trained end-to-end with the multi-component loss stack they use.

### 2. Token set was never extended or audited

They explicitly expanded the token set to 189 symbols to ensure full Vietnamese IPA compatibility, and provide a dedicated `extend.ipynb` that surgically widens the text encoder's embedding matrix before any training begins. The process requires running the extension notebook first, finding the extended weights, replacing the original weights, and adding new symbols to the config — critically, the instruction warns not to add symbols to the config before running extend.ipynb, as this causes misalignment between the model and the symbol set.

We never do this. Our pipeline generates viphoneme IPA strings and assumes Kokoro ZH's embedding matrix already covers Vietnamese phonemes. It almost certainly doesn't — Chinese IPA and Vietnamese IPA share some glyphs but Vietnamese tonal diacritics, final consonants, and specific vowels (ơ, ư, â, ê, ô) map to characters that likely aren't in Kokoro ZH's vocabulary. Any unseen token would be either skipped silently or crash the model.

### 3. No validation loop, no best-model gate

After each training epoch, they evaluate on a validation set, track mel + duration + F0 validation losses, and save the best model gated by validation loss improvement. They also save a `current_model.pth` every 2,000 iterations as a rolling checkpoint.

Our script has no val split, no eval loop, and no best-model selection. The smoke test only checks whether loss went down between step 1 and step 20 — it cannot catch overfitting, divergence after warmup, or a plateau at a poor local minimum.

### 4. Dataset strategy — their "North + South mixed" is a feature, not a bug

You noted they use both dialects. Looking at viVoice's composition, it's sourced from 186 YouTube channels with 887,772 samples across 1,016 hours at 24kHz, with all audio cleaned from noise and music, and clean cuts made at sentence boundaries. Mixing Northern and Southern data actually gives the style encoder more speaker diversity to learn from, making it more robust at inference time when someone presents a Northern reference audio. A model that only ever heard Northern speech during training is worse at generalizing the style encoder.

Our aggressive dialectal blocklist, combined with the SNR gate and speaker whitelist, may leave us with a very small retained corpus — potentially a few hundred hours or less. That's not enough to properly train the style encoder and decoder.

### 5. The `BatchSampler` duration-binning trick

Their `build_dataloader` uses parallel processing to determine sample lengths, then creates a `BatchSampler` that groups samples by duration for efficient training, placing all samples into time bins so each batch contains clips of similar length. This is important for two reasons: it avoids the padding waste of mixing 2-second clips with 14-second clips in the same batch, and it prevents the decoder from training on mostly-padded mel frames.

Our `collate_fn` does zero-padding to the batch maximum, which is correct but wasteful without binning. On a unified-memory device like the 8060S this is especially costly since every padded frame still occupies unified RAM bandwidth.

### 6. Half-second silence padding in the data loader

A subtle but meaningful detail: when loading each audio file, their dataset prepends and appends `np.zeros([12000])` — half a second of silence at 24kHz — to every clip. This trains the model to handle utterance boundaries cleanly, preventing the decoder from learning to abruptly cut off phonemes at clip edges. We don't do this.

### 7. Per-module learning rate differentiation

They use a lower fine-tuning learning rate (`ft_lr`) specifically for the decoder and style encoder, and tune the beta parameters differently for acoustic components: betas=(0.0, 0.99) rather than the standard (0.9, 0.99). This prevents catastrophic forgetting in the heavily pretrained acoustic decoder while allowing the alignment components to adapt faster. Our AdamW uses a single LR and standard betas across all parameters.

---

## The base model question — Kokoro ZH vs LibriTTS English

Their choice of English LibriTTS as the base is actually defensible and possibly *better* for general Vietnamese because:

- English LibriTTS has an extremely well-trained text aligner and prosody predictor with clean attention paths
- Vietnamese IPA contains phonemes closer to English in some respects (e.g., /f/, /v/, final stops) than to Mandarin
- The style encoder trained on English speech generalizes better to the speaker diversity in viVoice's mixed Northern/Southern corpus

Your choice of Kokoro ZH is theoretically sound for tone — Mandarin's 4 tones vs Vietnamese's 6 tones is the closest match in any available base — but this only matters if the F0/pitch contour predictor actually gets trained. Since our current `run_train.py` has no F0 loss, the tonal advantage of the ZH base is completely unutilized.

**Bottom line:** the base model choice is the second-order problem. The first-order problem is that the training loop doesn't train a TTS model at all.