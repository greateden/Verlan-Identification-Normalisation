import re
from pathlib import Path

import pandas as pd
from unidecode import unidecode

def toks(s: str):
    """Tokenize a string using the same logic as the original script."""
    return re.findall(r"[a-z0-9]+(?:’[a-z0-9]+)?", unidecode(s.lower()))

def load_lexicon():
    """Load the lexicon of known verlan forms."""
    data_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "GazetteerEntries.xlsx"
    df = pd.read_excel(data_path)
    return set(df["verlan_form"].dropna().astype(str).str.lower())

def test_tokenization_and_lexicon_matches():
    lexicon = load_lexicon()
    cases = [
        ("je joue ma bite", ["je", "joue", "ma", "bite"], []),
        ("Je joue ma bite.", ["je", "joue", "ma", "bite"], []),
        ("Je joue ma tebi.", ["je", "joue", "ma", "tebi"], []),
        ("Je joue avec mon chat.", ["je", "joue", "avec", "mon", "chat"], []),
        ("Je parle avec ma meuf.", ["je", "parle", "avec", "ma", "meuf"], ["meuf"]),
    ]
    for sentence, expected_tokens, expected_hits in cases:
        tokens = toks(sentence)
        assert tokens == expected_tokens
        assert sorted(set(tokens) & lexicon) == expected_hits
