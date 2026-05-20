This is a really important calibration point. Let me think through what "on par with Kokoro's 8 native voices" actually requires, then map where dangtr0408 fell short and what we need to do differently.

---

## Why dangtr0408's quality ceiling is low

**Batch size 3, 120k steps ≈ 360k total samples seen.** viVoice has 887k samples. They barely covered half the dataset once. Kokoro's native English voices were trained for what the author estimates at ~$1,000 of A100 time — roughly 300–500 hours, which at ~$2/hr for A100 is 150–250 A100-hours. At batch 3, 120k steps on an A100 doing ~1 step/second, their total training was about 33 hours. That's roughly **7–10× less compute** than a production-quality Kokoro voice.

The user's comparison to SparkTTS is telling. SparkTTS uses a large language model backbone (Qwen2.5-0.5B) for semantic tokens plus a DiTTo-style flow-matching vocoder — a fundamentally different and more powerful architecture. Comparing StyleTTS2-lite to SparkTTS is like comparing a lightweight convolutional model to a diffusion model. No amount of training steps on StyleTTS2-lite will match SparkTTS quality; that's an architecture gap, not a data gap.

---

## What "on par with Kokoro's 8 native voices" actually requires

Kokoro's native voices achieve their quality through three things that dangtr0408's approach lacked:

**1. Duration of training and dataset coverage.** Kokoro's English model was trained on LibriTTS-R (the remastered version, ~500h of clean studio-quality audiobook) at high batch sizes until convergence of all loss components — especially the adversarial discriminator, which takes the longest to stabilize. 120k steps at batch 3 almost certainly never reached discriminator convergence.

**2. Data quality ceiling, not just quantity.** viVoice is YouTube-sourced. Even after noise removal, it contains compression artifacts, mic variation across 186 channels, and inconsistent recording conditions. LibriTTS-R is studio recordings normalized to a consistent acoustic standard. For Vietnamese, PhoAudioBook is actually better sourced for audiobook-quality acoustic consistency — but only if enough hours survive the quality filter.

**3. The style diffusion component.** StyleTTS2's quality advantage over earlier models comes specifically from its diffusion-based style sampler, which learns a continuous distribution of speaking styles rather than a fixed embedding. This component requires the most training to converge and is the most sensitive to data diversity. Batch 3 is simply not enough to train the diffusion prior properly.

---

## What we need to do differently to beat them

There are three levers, in order of impact:

**Architecture: Stay on StyleTTS2 full, don't use "lite."** dangtr0408 deliberately uses StyleTTS2-lite, which is a reduced-capacity model. The full StyleTTS2 has a larger decoder and a proper diffusion-based style sampler. For production Vietnamese quality comparable to Kokoro's native voices, the full model is necessary — the lite version will always have a quality ceiling.

**Training budget: 500k+ steps at batch 16+.** On your Ryzen AI MAX+ 395 at the optimized settings from the previous conversation, with batch 64, 500k steps is roughly 35–40 hours. That's equivalent to or exceeding what was spent on Kokoro's English voices in compute-per-sample terms.

**Data: Hybrid corpus, not pure filtering.** Instead of aggressively filtering PhoAudioBook down to Northern-only speech, the smarter strategy is what dangtr0408 accidentally discovered works — use a **dialect-mixed corpus for training** but ensure Northern IPA is the phonemization standard. The style encoder learns from acoustic variety regardless of dialect; the text encoder learns phoneme-to-acoustic mappings from the IPA. Since we G2P everything through `viphoneme` with `dialect="north"`, even Southern audio gets labeled with Northern phoneme targets, and the model learns to produce Northern-accented output from any reference style. This maximizes data volume while maintaining dialectal output consistency.

Concretely: use all of viVoice (which you can access, unlike PhoAudioBook which requires streaming) plus PhoAudioBook Northern-filtered, and apply `viphoneme(dialect="north")` uniformly as the phonemization layer across both.

**The real quality gap to close is the discriminator training.** The MPD + MSD adversarial losses are what separates "robotic but intelligible" from "natural speech." These take the longest to converge and require the most data diversity. A practical schedule:

- Steps 1–50k: freeze the decoder and style diffusion, train only the text encoder and aligner on the new phoneme set (this is the token adaptation phase)
- Steps 50k–200k: unfreeze everything, train all components with discriminator at reduced λ\_gen (0.1× normal) to stabilize the G2P-to-acoustic mapping first
- Steps 200k–500k+: full loss weights, let the discriminator drive audio quality up

This staged approach is what differentiates a $1,000 Kokoro voice from a 1-day A100 fine-tune — not the architecture, but the training curriculum.

---

## Honest assessment

Beating dangtr0408 to their quality level is straightforward — it mainly requires fixing the training objective (our current `run_train.py` isn't doing TTS training at all, as identified in the last review) and running longer. Getting **on par with Kokoro's 8 native English voices in Vietnamese** is a harder target. Kokoro's native voices benefit from studio-quality training data and full convergence. With viVoice + PhoAudioBook and ~40 hours of compute on your hardware, you should realistically target **significantly better than dangtr0408, competitive with viXTTS** (the current best open Vietnamese TTS) — with parity to Kokoro's native voices as the stretch goal requiring a second training run once the first demonstrates convergence.

The critical next step remains the same: rewrite `run_train.py` to actually wire in the StyleTTS2 full model's loss stack, not just the XPhoneBERT projection head.