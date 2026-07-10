"""Rank Kalshi series by market-making viability from collected snapshots.

Reads market_snapshots.csv and computes per-series metrics.
Works with either a single snapshot (spatial metrics only) or many (adds time metrics).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path(__file__).parent / "market_snapshots.csv"

print(f"Loading {CSV} ...")
df = pd.read_csv(
    CSV,
    dtype={
        "ts": "int64",
        "ticker": "string",
        "series": "string",
        "status": "string",
    },
    low_memory=False,
)
print(f"Loaded {len(df):,} rows. Distinct snapshots: {df['ts'].nunique()}")

# Coerce numeric
for col in ["yes_bid","yes_ask","yes_bid_size","yes_ask_size","spread_cents","mid","last_price","volume_24h","volume_total","open_interest"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["has_bid"] = df["yes_bid"].notna()
df["has_ask"] = df["yes_ask"].notna()
df["two_sided"] = df["has_bid"] & df["has_ask"]

n_snapshots = df["ts"].nunique()
has_time = n_snapshots >= 5

per_series = []
for series, g in df.groupby("series", sort=False):
    n_tickers = g["ticker"].nunique()
    two_sided_pct = 100.0 * g["two_sided"].mean() if len(g) else 0.0

    ts_slice = g[g["two_sided"]]
    med_spread_c = float(np.nanmedian(ts_slice["spread_cents"])) if len(ts_slice) else np.nan
    med_bid_size = float(np.nanmedian(ts_slice["yes_bid_size"])) if len(ts_slice) else np.nan
    med_ask_size = float(np.nanmedian(ts_slice["yes_ask_size"])) if len(ts_slice) else np.nan

    # per-ticker mid stability (only if multi-snapshot)
    avg_std_mid_c = np.nan
    toxicity = np.nan
    if has_time:
        stds, tox_counts = [], []
        for _, tg in g.groupby("ticker"):
            m = tg.sort_values("ts")["mid"].dropna()
            if len(m) >= 5:
                stds.append(m.std() * 100.0)
                diffs_c = m.diff().dropna().abs() * 100.0
                tox_counts.append((diffs_c > 3.0).mean())
        if stds:
            avg_std_mid_c = float(np.mean(stds))
            toxicity = float(np.mean(tox_counts))

    # totals from most recent snapshot
    latest = g.sort_values("ts").drop_duplicates("ticker", keep="last")
    total_vol_24h = float(latest["volume_24h"].fillna(0).sum())
    total_oi = float(latest["open_interest"].fillna(0).sum())

    per_series.append({
        "series": series,
        "n_tickers": n_tickers,
        "n_snapshots": g["ts"].nunique(),
        "two_sided_pct": round(two_sided_pct, 1),
        "med_spread_c": round(med_spread_c, 2) if not np.isnan(med_spread_c) else None,
        "med_bid_sz": round(med_bid_size, 1) if not np.isnan(med_bid_size) else None,
        "med_ask_sz": round(med_ask_size, 1) if not np.isnan(med_ask_size) else None,
        "avg_std_mid_c": round(avg_std_mid_c, 2) if not np.isnan(avg_std_mid_c) else None,
        "toxicity_pct": round(100 * toxicity, 1) if not np.isnan(toxicity) else None,
        "total_vol_24h": int(total_vol_24h),
        "total_oi": int(total_oi),
    })

pr = pd.DataFrame(per_series)

# Score
def score(r):
    ts = (r.get("two_sided_pct") or 0) / 100.0
    sp = r.get("med_spread_c") or 0
    st = (r.get("avg_std_mid_c") or 1)  # if we have no time data, treat as 1
    tox = (r.get("toxicity_pct") or 0) / 100.0
    vol = r.get("total_vol_24h") or 0
    oi = r.get("total_oi") or 0
    # cap spread contribution (very wide = illiquid trap, not opportunity)
    sp_eff = 0 if sp < 2 else min(sp, 15)
    depth_factor = np.log1p(vol + oi)
    return (ts * sp_eff * depth_factor) / (max(st, 1.0) * (1 + 3 * tox))

pr["score"] = pr.apply(score, axis=1)

# meaningful sample: >=3 tickers
pr = pr[pr["n_tickers"] >= 3].copy()
pr = pr.sort_values("score", ascending=False)

hdr = f"{'SERIES':<25} {'#TIC':>5} {'#SNP':>5} {'2SIDE%':>7} {'SPRD_c':>7} {'BID_SZ':>7} {'ASK_SZ':>7}"
if has_time:
    hdr += f" {'STD_c':>7} {'TOX%':>6}"
hdr += f" {'VOL_24h':>12} {'OI':>10} {'SCORE':>10}"

print("\n" + "=" * len(hdr))
print(hdr)
print("=" * len(hdr))
for _, r in pr.head(30).iterrows():
    line = (f"{r['series']:<25} {r['n_tickers']:>5} {r['n_snapshots']:>5} "
            f"{r['two_sided_pct']:>7.1f} "
            f"{str(r['med_spread_c']):>7} "
            f"{str(r['med_bid_sz']):>7} "
            f"{str(r['med_ask_sz']):>7}")
    if has_time:
        line += f" {str(r['avg_std_mid_c']):>7} {str(r['toxicity_pct']):>6}"
    line += f" {r['total_vol_24h']:>12,} {r['total_oi']:>10,} {r['score']:>10.2f}"
    print(line)

out = Path(__file__).parent / "series_ranking.csv"
pr.to_csv(out, index=False)
print(f"\nFull ranking saved to {out}")
print(f"\nMode: {'time-aware (multi-snapshot)' if has_time else 'spatial only (1 snapshot)'}")
print("\nLegend:")
print("  2SIDE% = % rows with both bid AND ask")
print("  SPRD_c = median spread (cents) when two-sided")
print("  BID_SZ / ASK_SZ = median depth on each side")
if has_time:
    print("  STD_c = mean per-ticker std of mid, in cents (lower = stable)")
    print("  TOX%  = % consecutive mid moves > 3c (info flow proxy)")
print("  VOL_24h / OI = totals from latest snapshot")
print("  SCORE = 2SIDE * min(SPRD,15) * log(1+VOL+OI) / (STD * (1 + 3*TOX))")
