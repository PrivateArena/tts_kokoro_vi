# PhoAudiobook Dataset Reference (`thivux/phoaudiobook`)

This document outlines the schema, structure, and unique speakers of the **PhoAudiobook** dataset, a large-scale high-quality Vietnamese speech corpus optimized for zero-shot text-to-speech (TTS) synthesis.

---

## 1. Dataset Overview

* **Dataset ID:** `thivux/phoaudiobook` (Gated)
* **Total Audio Duration:** ~1,494 hours (940 hours of long-form audio $>10$s, 554 hours of augmented short audio)
* **Total Utterances (Train):** 1,042,919
* **Original Source:** Curated audiobook tracks, background-noise-removed using `demucs`, silent periods trimmed, and volumes normalized.
* **Total Dataset Size:** 167.16 GB (download size) / 401.51 GB (uncompressed dataset size)

---

## 2. Schema and Columns

In the Hugging Face `datasets` abstraction layer, the schema features are:
* `audio`: `Audio(sampling_rate=None, decode=True)`
* `text`: `Value('string')` (Transcription)
* `speaker`: `Value('string')` (Explicit speaker name/ID)

Inside the underlying physical **Parquet files** (`data/train-*.parquet`), the schema consists of the following columns:

| Column | Type | Description |
| :--- | :--- | :--- |
| **`bytes`** | Binary | Raw audio sample bytes (decoded as waveform array in `datasets`). |
| **`path`** | String | Original/relative path of the audio source clip. |
| **`text`** | String | Normalized transcription text (spoken form). |
| **`speaker`** | String | Speaker name/ID (e.g. `Nguyễn_Văn_Khỏa`, `Lê_Đức_Quân`). |

---

## 3. Data Splits

| Split | Number of Examples | Compressed Size (Parquet) |
| :--- | :--- | :--- |
| **`train`** | 1,042,919 | 167.16 GB (Split across 803 Parquet files) |
| **`validation`** | 141 | 52.68 MB |
| **`test`** | 383 | 142.91 MB |

---

## 4. Unique Speakers List (670 Speakers)

A total of **670 unique speakers** were successfully extracted from the dataset train split metadata. The extracted speaker list is saved locally in:
* Plain text format: [data/unique_speakers.txt](file:///media/jang/home/Deve/tts_kokoro_vi/data/unique_speakers.txt)
* JSON format: [data/unique_speakers.json](file:///media/jang/home/Deve/tts_kokoro_vi/data/unique_speakers.json)

### Top Extracted Speakers (Alphabetical order - Sample):
```text
Diễm_Hân
Lê_Minh_Thương
Lê_Vương
Lê_Đức_Quân
Mai_Anh
Nguyễn_Hiến_Lê
Nguyễn_Phước_Hải
Nguyễn_Văn_Khỏa
Phúc_An
Phúc_Lâm
Thanh_Thủy
Trần_Trung_Đạo
Trần_Vân
Voiz
Văn_Hoàng
Vũ_Liêm
Đông_Quân
Đạt_Phi
Đức_Uy
...
```
*(Refer to `data/unique_speakers.txt` for the full alphabetical list of all 670 speakers).*
