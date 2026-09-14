#!/usr/bin/env python3
"""
hebrew_g2p.py — Hebrew grapheme-to-phoneme for this Kokoro-Hebrew model.
Portable (self-contained) version: imports kokoro_symbols.py from this folder.

Backend: phonikud (vocalized Hebrew -> IPA), normalized into Kokoro's 178-token
vocab. MUST be used identically at train and inference time.
    pip install phonikud
"""
from __future__ import annotations
import re, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kokoro_symbols import dicts as KOKORO_DICTS          # bundled alongside this file
VALID_SYMBOLS = set(KOKORO_DICTS.keys())

_NIQQUD_RE = re.compile(r"[֑-ׇ]")
_STRIP_CHARS = {"͡", "͜", "​", "‌", "‍", "﻿", "_"}
_FIXUPS = {"g": "ɡ", "ɫ": "l", "ˑ": "ː"}
_MULTISPACE_RE = re.compile(r"[ ]{2,}")
_NUM_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")


def has_niqqud(text: str) -> bool:
    return bool(_NIQQUD_RE.search(text))


def preprocess_text(text: str) -> str:
    return _NUM_COMMA_RE.sub("", text)


def normalize_to_kokoro(ph: str) -> tuple[str, Counter]:
    out, dropped = [], Counter()
    for ch in ph:
        if ch in _STRIP_CHARS:
            continue
        if ch in _FIXUPS:
            if _FIXUPS[ch]:
                out.append(_FIXUPS[ch])
            continue
        if ch in VALID_SYMBOLS:
            out.append(ch)
        else:
            dropped[ch] += 1
    return _MULTISPACE_RE.sub(" ", "".join(out)).strip(), dropped


_PHONEMIZE = None


def phonemize_hebrew(text: str) -> tuple[str, Counter]:
    global _PHONEMIZE
    if _PHONEMIZE is None:
        from phonikud import phonemize as _p
        _PHONEMIZE = _p
    return normalize_to_kokoro(_PHONEMIZE(preprocess_text(text)))
