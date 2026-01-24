# src/patrol_planner.py
import numpy as np
import pandas as pd

def build_band(df: pd.DataFrame) -> pd.Series:
    """risk_decile varsa kullanır; yoksa risk_level."""
    if "risk_decile" in df.columns and df["risk_decile"].notna().any():
        d = pd.to_numeric(df["risk_decile"], errors="coerce")
        # 9-10 high, 6-8 med, 1-5 low
        out = pd.Series(index=df.index, dtype="object")
        out[(d >= 9)] = "HIGH"
        out[(d >= 6) & (d <= 8)] = "MED"
        out[(d >= 1) & (d <= 5)] = "LOW"
        out = out.fillna("NA")
        return out
    # fallback: risk_level
    rl = df.get("risk_level", pd.Series(["NA"]*len(df), index=df.index)).astype(str)
    out = pd.Series(index=df.index, dtype="object")
    out[rl.isin(["VERY_HIGH","HIGH"])] = "HIGH"
    out[rl.isin(["MED"])] = "MED"
    out[rl.isin(["LOW","VERY_LOW"])] = "LOW"
    out = out.fillna("NA")
    return out

def equal_distribution_plan(
    df_fc: pd.DataFrame,
    *,
    rank_col: str = "expected_harm",
    num_units: int = 10,
    cells_per_unit: int = 10,
    share_high: float = 0.50,
    share_med: float  = 0.30,
    share_low: float  = 0.20,
    max_cells_per_geoid: int = 3,
) -> pd.DataFrame:
    """
    Band kotasıyla hücre seçer ve round-robin ile ekiplere dağıtır.
    """
    df = df_fc.copy()
    if df.empty:
        return df

    # sanitize shares
    s = float(share_high + share_med + share_low)
    if s <= 0:
        share_high, share_med, share_low = 0.5, 0.3, 0.2
    else:
        share_high, share_med, share_low = share_high/s, share_med/s, share_low/s

    total_cells = int(num_units) * int(cells_per_unit)
    n_high = int(round(total_cells * share_high))
    n_med  = int(round(total_cells * share_med))
    n_low  = total_cells - n_high - n_med

    # band
    df["risk_band"] = build_band(df)

    # rank
    if rank_col not in df.columns:
        rank_col = "risk_score" if "risk_score" in df.columns else "expected_harm"
    df[rank_col] = pd.to_numeric(df[rank_col], errors="coerce")
    df = df.sort_values(rank_col, ascending=False)

    def pick(df_band: pd.DataFrame, n: int) -> pd.DataFrame:
        if n <= 0 or df_band.empty:
            return df_band.head(0)
        out = []
        geoid_counts = {}
        for _, r in df_band.iterrows():
            g = str(r.get("GEOID",""))
            geoid_counts[g] = geoid_counts.get(g, 0)
            if geoid_counts[g] >= int(max_cells_per_geoid):
                continue
            out.append(r)
            geoid_counts[g] += 1
            if len(out) >= n:
                break
        return pd.DataFrame(out)

    high = pick(df[df["risk_band"]=="HIGH"], n_high)
    med  = pick(df[df["risk_band"]=="MED"],  n_med)
    low  = pick(df[df["risk_band"]=="LOW"],  n_low)

    plan = pd.concat([high, med, low], ignore_index=True)
    plan = plan.sort_values(rank_col, ascending=False).reset_index(drop=True)

    # round-robin assign
    unit_ids = []
    for i in range(len(plan)):
        unit_ids.append((i % int(num_units)) + 1)
    plan["unit_id"] = unit_ids

    cols_front = [c for c in ["unit_id","risk_band","date","hour_range","GEOID","risk_level",rank_col,"expected_harm","p_event","expected_count","top1_category"] if c in plan.columns]
    plan = plan[cols_front + [c for c in plan.columns if c not in cols_front]]

    return plan
