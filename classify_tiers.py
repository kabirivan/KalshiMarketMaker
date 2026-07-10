"""Classify each series in series_ranking.csv into tiers S/A/B/C/D/E/F.

Rules (from user's prior S/A/B assignments):
  S: STD < 0.5c, TOX < 0.2%, two_sided > 60%
  A: STD < 2c,   TOX < 1%,   two_sided > 95%, vol_24h > 1000, min(bid_sz, ask_sz) > 100
  B: STD < 4c,   TOX < 3%,   two_sided > 80%, vol_24h > 500
  C: STD < 6c,   TOX < 5%,   two_sided > 80%, vol_24h > 100
  D: STD < 10c,  TOX < 8%,   two_sided > 70%, vol_24h > 50
  E: STD < 20c,  TOX < 15%,  two_sided > 40%
  F: everything else (including rows with missing STD/TOX)
"""
from pathlib import Path
import math

import pandas as pd

ROOT = Path(__file__).parent
df = pd.read_csv(ROOT / "series_ranking.csv")


def tier(r) -> str:
    std = r.get("avg_std_mid_c")
    tox = r.get("toxicity_pct")
    ts = r.get("two_sided_pct") or 0
    vol = r.get("total_vol_24h") or 0
    bid_sz = r.get("med_bid_sz") or 0
    ask_sz = r.get("med_ask_sz") or 0

    if std is None or (isinstance(std, float) and math.isnan(std)):
        return "F"
    if tox is None or (isinstance(tox, float) and math.isnan(tox)):
        return "F"

    depth_min = min(bid_sz, ask_sz)

    if std < 0.5 and tox < 0.2 and ts > 60:
        return "S"
    if std < 2 and tox < 1 and ts > 95 and vol > 1000 and depth_min > 100:
        return "A"
    if std < 4 and tox < 3 and ts > 80 and vol > 500:
        return "B"
    if std < 6 and tox < 5 and ts > 80 and vol > 100:
        return "C"
    if std < 10 and tox < 8 and ts > 70 and vol > 50:
        return "D"
    if std < 20 and tox < 15 and ts > 40:
        return "E"
    return "F"


df["tier"] = df.apply(tier, axis=1)

order = {"S": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}
df = df.sort_values(["tier", "score"], key=lambda c: c.map(order) if c.name == "tier" else -c)

out = ROOT / "series_tiers.csv"
df.to_csv(out, index=False)

print(f"Saved {out}")
print()
counts = df["tier"].value_counts().reindex(list(order.keys())).fillna(0).astype(int)
print("Tier counts:")
for t, n in counts.items():
    print(f"  {t}: {n}")

for t in ["S", "A", "B", "C", "D", "E"]:
    sub = df[df["tier"] == t]
    if not len(sub):
        continue
    print()
    print(f"=== Tier {t} ({len(sub)} series) ===")
    cols = ["series", "avg_std_mid_c", "toxicity_pct", "two_sided_pct",
            "med_spread_c", "med_bid_sz", "med_ask_sz", "total_vol_24h", "score"]
    print(sub[cols].to_string(index=False))
