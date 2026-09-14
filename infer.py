#!/usr/bin/env python3
"""
infer.py — Hebrew Text-to-Speech with the fine-tuned Kokoro-82M model.

Setup:
    pip install kokoro phonikud soundfile torch

Usage:
    python infer.py --text "שָׁלוֹם, מָה שְׁלוֹמְךָ הַיּוֹם?" --out hello.wav

Input text must be VOCALIZED (with niqqud). To vocalize plain Hebrew first:
    pip install phonikud-onnx   # then use its add_diacritics(), or Dicta Nakdan.
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SPLIT = re.compile(r"(?<=[.!?])\s+")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="VOCALIZED (niqqud) Hebrew text")
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--model", default=str(HERE / "kokoro_v1_hebrew.pth"))
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--voice", default=str(HERE / "voices" / "he_shaul.pt"))
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    import torch, numpy as np, soundfile as sf
    from kokoro import KModel
    from hebrew_g2p import phonemize_hebrew, has_niqqud

    dev = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    km = KModel(repo_id="hexgrad/Kokoro-82M", config=args.config, model=args.model).to(dev).eval()
    voice = torch.load(args.voice, map_location=dev, weights_only=True)

    if not has_niqqud(args.text):
        print("WARNING: input has no niqqud — vowels will likely be wrong.")

    pieces = []
    for chunk in SPLIT.split(args.text.strip()):
        if not chunk:
            continue
        ps, dropped = phonemize_hebrew(chunk)
        if dropped:
            print("dropped (out-of-vocab):", dict(dropped))
        if not ps:
            continue
        n = min(len(ps), voice.shape[0]) - 1
        with torch.no_grad():
            out = km(ps, voice[n], args.speed, return_output=True)
        pieces.append(out.audio.cpu().numpy())

    if not pieces:
        sys.exit("No audio produced.")
    sf.write(args.out, np.concatenate(pieces), 24000)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
