#!/usr/bin/env python3
"""
download_model.py — fetch the Hebrew Kokoro weights and prepare them for the
pytorch inference package in this folder.

The published weights live on Hugging Face in mlx-audio / safetensors form:

    https://huggingface.co/avris/kokoro-hebrew-saspeech   (GATED, non-commercial)

The bundled pytorch `kokoro/` library in this folder loads the classic
`.pth` / `.pt` format, so this script downloads the safetensors and converts
them in place to:

    kokoro_v1_hebrew.pth     (model weights, fp32)
    voices/he_shaul.pt       (voicepack tensor)

The conversion is the exact inverse of how the safetensors were exported: the
weight_norm parameters are renamed from the classic `weight_g`/`weight_v` names
back to the parametrizations API (`parametrizations.weight.original0/1`) that
this fork of kokoro expects. Shapes are identical; only names and dtype change.

Prerequisites (one-time):
    1. Open the model page above and click "Agree and access repository"
       (the data is non-commercial — see DATASET.md).
    2. `pip install huggingface_hub safetensors torch`
    3. `huggingface-cli login`   (so the gated download is authorized)

Usage:
    python download_model.py
"""
from __future__ import annotations
from pathlib import Path

REPO_ID = "avris/kokoro-hebrew-saspeech"
HERE = Path(__file__).resolve().parent
# top-level KModel components, used to split "comp.rest" safetensors keys
COMPONENTS = ("bert", "bert_encoder", "predictor", "text_encoder", "decoder")


def _rename_key(sub: str) -> str:
    """classic weight_norm name -> parametrizations API name."""
    if sub.endswith(".weight_g"):
        return sub[: -len(".weight_g")] + ".parametrizations.weight.original0"
    if sub.endswith(".weight_v"):
        return sub[: -len(".weight_v")] + ".parametrizations.weight.original1"
    return sub


def main() -> None:
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    print(f"Downloading weights from {REPO_ID} (gated) ...")
    model_st = hf_hub_download(repo_id=REPO_ID, filename="kokoro-v1_0.safetensors")
    voice_st = hf_hub_download(repo_id=REPO_ID, filename="voices/he_shaul.safetensors")

    # --- model: flat safetensors -> {component: state_dict} for KModel.load ---
    flat = load_file(model_st)
    state: dict[str, dict] = {c: {} for c in COMPONENTS}
    for key, tensor in flat.items():
        comp, _, sub = key.partition(".")
        if comp not in state:
            raise ValueError(f"Unexpected top-level component in weights: {comp!r}")
        state[comp][_rename_key(sub)] = tensor.float()
    out_model = HERE / "kokoro_v1_hebrew.pth"
    torch.save(state, out_model)
    print(f"  wrote {out_model.name}  ({sum(len(v) for v in state.values())} tensors)")

    # --- voicepack: single 'voice' tensor -> plain .pt ---
    voice = load_file(voice_st)["voice"].float()
    (HERE / "voices").mkdir(exist_ok=True)
    out_voice = HERE / "voices" / "he_shaul.pt"
    torch.save(voice, out_voice)
    print(f"  wrote voices/{out_voice.name}  shape={tuple(voice.shape)}")

    print("\nDone. Now run, e.g.:")
    print('  python infer.py --text "שָׁלוֹם, מָה שְׁלוֹמְךָ הַיּוֹם?" --out hello.wav')


if __name__ == "__main__":
    main()
