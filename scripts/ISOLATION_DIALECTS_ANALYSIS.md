For **Kokoro-82M** (which is structurally based on *StyleTTS2*), the correct engineering choice is highly specific to how this particular model architecture operates.

You should **train both dialects in a single model**, but you must introduce **explicit language/dialect conditioning tags** at the phoneme or token level.

Splitting them into two standalone models is an intuitive brute-force approach, but doing so heavily undermines Kokoro's core architectural advantages.

---

## Why You Should Train Both in One Model

### 1. Latent Style Space Sharing

Kokoro uses a style encoder and a latent vector space ($S$-space) to map speaker identity, emotion, and prosody.

* If you train separate models, each model has to learn human speech physics, emotional intonation, breathing artifacts, and rhythm entirely from scratch using only half the available Vietnamese data.
* If you train both in one model, a Southern voice can borrow the "reading style," "emotional weight," or "studio acoustics" learned from a Northern dataset. The underlying voice physics cross-pollinate, leading to significantly higher naturalness for both dialects.

### 2. Cross-Dialect Voice Swapping (The Cool Feature)

If you train both dialects in a single model, you gain the ability to pass a **Southern style embedding** to text processed via a **Northern phonemizer**, or vice versa. This allows you to generate a Saigon native trying to speak with a Hanoi accent, or let a single speaker voiceprint seamlessly toggle between both regions. You lose this capability entirely with two separate models.

---

## The Catch: How to Prevent Dialect "Bleeding"

If you just blindly throw Northern and Southern audio files into a single training bucket, the model will output a jarring, corrupted hybrid accent. To make a single model work, you must isolate the acoustic spaces through the frontend pipeline.

### Step 1: Strict Dialect Tagging via the Phonemizer

Kokoro depends heavily on **espeak-ng** and the International Phonetic Alphabet (IPA) to convert text into phoneme tokens before feeding them to the StyleTTS2 backbone.

* You **cannot** use the same language code or the same phonemizer rules for both regions.
* You need to treat Northern Vietnamese and Southern Vietnamese as two separate "languages" at the pipeline level (e.g., assigning token identifiers like `vi-vn-north` and `vi-vn-south`).

When processing the text *vui*:

* The Northern pipeline must emit phonemes mapping to `/zui/`.
* The Southern pipeline must emit phonemes mapping to `/jui/`.

By strictly separating the input tokens, the underlying neural network understands that `/z/` and `/j/` are completely different acoustic targets, preventing the model from getting confused.

### Step 2: Explicit Speaker & Style Conditioning

During fine-tuning, make sure your training metadata maps speakers explicitly to their regional identities. If Speaker A is from Saigon, ensure all their text is strictly tokenized using the Southern phonemizer rules. The model's projection layers will naturally cluster Southern audio features with Southern phoneme streams.

---

## Strategy Summary

| Strategy | Pros | Cons | Verdict |
| --- | --- | --- | --- |
| **Separate Models** | Zero risk of regional accent mixing or bleeding. | Double the VRAM/storage footprint; lower overall voice quality because data is cut in half; no cross-dialect voice synthesis. | **Not Recommended** |
| **Single Joint Model (Properly Tagged)** | Maximum data density; shared prosody and emotional weights; enables cross-dialect voice blending. | Requires careful, strict data curation and two distinct G2P (Grapheme-to-Phoneme) frontend configurations. | **Highly Recommended** |

If you have at least 10–20 hours of clean audio for each region, training a single multilingual/multidialect model using distinct regional phoneme pipelines will yield a much more robust and expressive result for Kokoro.