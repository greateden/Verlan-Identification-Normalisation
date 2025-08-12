import pandas as pd, re
from unidecode import unidecode

def toks(s): return re.findall(r"[a-z0-9]+(?:’[a-z0-9]+)?", unidecode(s.lower()))
v = set(pd.read_excel("GazetteerEntries.xlsx")["verlan_form"].dropna().astype(str).str.lower())
for s in ["je joue ma bite", "Je joue ma bite.", "Je joue ma tebi.", "Je joue avec mon chat."]:
    ts = toks(s)
    exact = sorted(set(ts) & v)
    print(s, "-> tokens:", ts, " | exact_hits:", exact)
