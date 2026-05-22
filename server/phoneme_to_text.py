"""
phoneme_to_text.py — ARPAbet phoneme string → English text.

Strategy (in order):
  1. CMU Pronouncing Dictionary lookup, ranked by Brown corpus word frequency
  2. Pollinations LLM fallback for groups with no cmudict match
"""

import os
import re
import requests

import nltk

# ─── Download required NLTK data once ───────────────────────────────────────

for _pkg in ("cmudict", "brown", "words"):
    try:
        nltk.data.find(f"corpora/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

from nltk.corpus import cmudict as _cmudict, brown as _brown, words as _nltk_words

# ─── Word frequency from Brown corpus (higher = more common) ─────────────────

_FREQ: dict[str, int] = {}
for _w in _brown.words():
    _FREQ[_w.lower()] = _FREQ.get(_w.lower(), 0) + 1

# Also build a set of known English words for filtering
_KNOWN_WORDS = set(w.lower() for w in _nltk_words.words())

# ─── Build stress-stripped cmudict index ──────────────────────────────────────

_RAW_DICT = _cmudict.dict()
_STRESS_RE = re.compile(r"\d")

def _strip(ph: str) -> str:
    return _STRESS_RE.sub("", ph)


def _word_score(word: str) -> int:
    """
    Higher is better.
    Known common words beat rare/archaic ones; frequency breaks remaining ties.
    """
    base = 1 if word in _KNOWN_WORDS else 0
    return _FREQ.get(word, 0) + base * 10_000


def _build_index(raw: dict) -> dict:
    """
    tuple(stripped_phonemes) → best English word, ranked by corpus frequency.
    """
    index: dict[tuple, str] = {}
    for word, pron_list in raw.items():
        if not word.isalpha():          # skip abbreviations / contractions
            continue
        for pron in pron_list:
            key = tuple(_strip(p) for p in pron)
            existing = index.get(key)
            if existing is None or _word_score(word) > _word_score(existing):
                index[key] = word
    return index


_INDEX = _build_index(_RAW_DICT)


def _lookup(tokens: list) -> str | None:
    """Exact stress-stripped cmudict lookup."""
    key = tuple(t.upper() for t in tokens)
    return _INDEX.get(key)


# ─── Pollinations LLM (fallback for truly unknown phoneme combos) ─────────────

POLLINATION_API_URL = "https://text.pollinations.ai/openai"

_WORD_PROMPT = (
    "You are an ARPAbet phoneme decoder. "
    "Given ARPAbet phonemes for ONE word, output ONLY that single English word. "
    "No punctuation, no explanation — just the word."
)


def _llm_word(phoneme_group: list, api_key=None) -> str:
    ph_str = " ".join(phoneme_group)
    key = api_key or os.environ.get("POLLINATION_API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    payload = {
        "model": "openai",
        "messages": [
            {"role": "system", "content": _WORD_PROMPT},
            {"role": "user",   "content": ph_str},
        ],
        "temperature": 0.0,
        "max_tokens": 16,
    }
    resp = requests.post(POLLINATION_API_URL, json=payload,
                         headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip().lower()


# ─── Public API ───────────────────────────────────────────────────────────────

def phonemes_to_text(phoneme_string: str, api_key=None) -> dict:
    """
    Convert an ARPAbet phoneme string to English text.

    Args:
        phoneme_string: e.g.  "W AO T ER | B AA T AH L"
        api_key:        Pollinations API key (optional)

    Returns:
        { 'text': str, 'raw_phonemes': str }
    """
    groups = [g.strip().split() for g in phoneme_string.split("|")]
    groups = [g for g in groups if g]

    words = []
    for group in groups:
        word = _lookup(group)
        if word is None:
            try:
                word = _llm_word(group, api_key)
            except Exception:
                word = " ".join(group)   # last resort: echo phonemes
        words.append(word)

    sentence = " ".join(words).capitalize()
    if sentence and sentence[-1] not in ".?!":
        sentence += "."

    return {"text": sentence, "raw_phonemes": phoneme_string}


def batch_phonemes_to_text(phoneme_list: list, api_key=None) -> list:
    results = []
    for phonemes in phoneme_list:
        try:
            results.append(phonemes_to_text(phonemes, api_key))
        except Exception as e:
            results.append({"text": f"[ERROR: {e}]", "raw_phonemes": phonemes})
    return results


# ─── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        "W AO T ER | B AA T AH L",
        "OW P AH N | DH AH | D AO R",
        "DH AH | K AE T | S AE T | AA N | DH AH | M AE T",
        "HH AW | AA R | Y UW | T AH D EY",
        "G UH D | M AO R N IH NG",
        "TH AE NG K | Y UW",
        "N UH R AH L | D IH K OW D",
        "HH AH L OW | W ER L D",
        "AY | W AA N T | W AO T ER",
        "T ER N | AO F | DH AH | L AY T",
    ]
    for ph in tests:
        r = phonemes_to_text(ph)
        print(f"IN : {ph}")
        print(f"OUT: {r['text']}\n")
