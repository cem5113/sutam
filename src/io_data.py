# src/io_data.py
import os
import numpy as np
import pandas as pd

GEOID_PAD_TO_LEN = 11
APPLY_GEOID_PAD = True

def norm_geoid(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.replace(r"\D+", "", regex=True)
    if APPLY_GEOID_PAD:
        s = s.str.zfill(GEOID_PAD_TO_LEN)
    return s

def safe_num(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def load_parquet_or_csv(path: str) -> pd.DataFrame:
    if not path or (not os.path.exists(path)):
        return pd.DataFrame()
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def prepare_forecast(df_fc: pd.DataFrame, df_profile: pd.DataFrame | None = None) -> pd.DataFrame:
    """UI için forecast'i normalize eder, tipleri düzeltir, top1_category yoksa profilden üretir."""
    if df_fc is None or df_fc.empty:
        return pd.DataFrame()

    df = df_fc.copy()

    # required-ish columns
    for c in ["GEOID", "date", "hour_range"]:
        if c not in df.columns:
            raise ValueError(f"Forecast'te zorunlu kolon eksik: {c}")

    df["GEOID"] = norm_geoid(df["GEOID"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()

    for c in ["p_event","expected_count","expected_harm","risk_score","risk_decile"]:
        df = safe_num(df, c)

    # top1_category yoksa geoid_profile.top5_categories -> top1 üret
    if (df_profile is not None) and (not df_profile.empty) and ("top1_category" not in df.columns):
        gp = df_profile.copy()
        if "GEOID" in gp.columns:
            gp["GEOID"] = norm_geoid(gp["GEOID"])
        if "top5_categories" in gp.columns:
            def pick_first(tc):
                if tc is None or (isinstance(tc, float) and np.isnan(tc)):
                    return np.nan
                s = str(tc)
                if not s:
                    return np.nan
                return s.split(",")[0].split(":")[0]
            gp["top1_category"] = gp["top5_categories"].apply(pick_first)
            df = df.merge(gp[["GEOID","top1_category"]], on="GEOID", how="left")
        else:
            df["top1_category"] = np.nan

    return df

def prepare_profile(df_profile: pd.DataFrame) -> pd.DataFrame:
    if df_profile is None or df_profile.empty:
        return pd.DataFrame()
    gp = df_profile.copy()
    if "GEOID" in gp.columns:
        gp["GEOID"] = norm_geoid(gp["GEOID"])
    return gp

def make_cell_id(df: pd.DataFrame) -> pd.Series:
    """BLOK-9 join için stabil anahtar: GEOID|YYYY-MM-DD|hour_range"""
    d = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df["GEOID"].astype(str) + "|" + d.astype(str) + "|" + df["hour_range"].astype(str)
