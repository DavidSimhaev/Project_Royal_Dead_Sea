# Inference — Kokoro-82M Hebrew (fine-tuned)

A Hebrew text-to-speech voice, fine-tuned from
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (StyleTTS2 architecture)
on the **SASPEECH gold-standard** dataset (speaker: Shaul Amsterdamski). Single
speaker, 24 kHz output.

## Contents
```
infer.py                    standalone inference script
download_model.py           fetch + convert the weights from Hugging Face
hebrew_g2p.py               Hebrew -> IPA (phonikud) -> Kokoro vocab
kokoro_symbols.py           178-token vocab (required by hebrew_g2p)
config.json                 Kokoro model config
hebrew_phoneme_contract.json  reference text->phoneme pairs (G2P contract)
kokoro/                     bundled kokoro library (Apache-2.0 — see kokoro/LICENSE)
```
The model weights are **not** committed to git. `download_model.py` fetches them
from Hugging Face.

> **Note — the bundled `kokoro/` matters.** This model was trained against a
> specific kokoro fork; upstream pip `kokoro` (0.9.x) produces garbled audio from
> these weights. The bundle ships the compatible `kokoro/` package, and `infer.py`
> puts this folder first on `sys.path` so the local copy wins.

## Install
```bash
pip install kokoro phonikud soundfile torch huggingface_hub safetensors
```
You still `pip install kokoro` to pull shared dependencies (misaki, transformers,
torch, …); the bundled `kokoro/` code is what actually runs.

## Get the weights
The weights are on Hugging Face, **gated and non-commercial**
([`avris/kokoro-hebrew-saspeech`](https://huggingface.co/avris/kokoro-hebrew-saspeech)):

1. Open the model page and accept the terms (SASPEECH is non-commercial — see
   [`../DATASET.md`](../DATASET.md)).
2. `huggingface-cli login`
3. `python download_model.py` — downloads and writes `kokoro_v1_hebrew.pth` and
   `voices/he_shaul.pt` into this folder.

On Apple Silicon you can instead use the safetensors directly with
[`mlx-audio`](https://github.com/Blaizzy/mlx-audio); no conversion needed.

## Use
Input text must be **vocalized** (with niqqud), because Hebrew vowels are not
written and the model was trained on diacritized text:
```bash
python infer.py --text "שָׁלוֹם, מָה שְׁלוֹמְךָ הַיּוֹם?" --out hello.wav
# add --device cuda if you have an NVIDIA GPU
```
To vocalize plain Hebrew first, use `phonikud-onnx` (`add_diacritics`) or
[Dicta Nakdan](https://nakdan.dicta.org.il/).

Programmatic use:
```python
import torch, soundfile as sf, numpy as np
from kokoro import KModel
from hebrew_g2p import phonemize_hebrew

km = KModel(repo_id="hexgrad/Kokoro-82M", config="config.json", model="kokoro_v1_hebrew.pth").eval()
voice = torch.load("voices/he_shaul.pt", weights_only=True)
ps, _ = phonemize_hebrew("תּוֹדָה רַבָּה!")
audio = km(ps, voice[len(ps)-1], 1.0, return_output=True).audio.numpy()
sf.write("out.wav", audio, 24000)
```

## Quality & limitations
- **Intelligibility:** ~3% word error rate (Whisper large-v3 round-trip) on the
  best checkpoint — clear, understandable Hebrew.
- Trained on ~4 h (SASPEECH gold) + StyleTTS2 Stage 2 (optionally seeded from the
  larger ~26 h automatic subset). Naturalness/prosody improves with more Stage-2
  epochs.
- The G2P (`hebrew_g2p.phonemize_hebrew`) **must** be used for inference — Kokoro/
  misaki have no built-in Hebrew, so the model consumes phonemes produced by
  phonikud.

## Licensing
- This inference code (excluding `kokoro/`): **MIT** (repo root `LICENSE`).
- Bundled `kokoro/`: **Apache-2.0** (`kokoro/LICENSE`).
- **phonikud** G2P: **CC BY 4.0** — attribution required (see repo `NOTICE`).
- **Weights / training data (SASPEECH)**: **non-commercial** — see
  [`../DATASET.md`](../DATASET.md). The weights inherit these terms.
