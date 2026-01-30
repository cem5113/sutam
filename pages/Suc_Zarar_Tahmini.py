# pages/Suc_Zarar_Tahmini.py
from __future__ import annotations

import os
import pandas as pd
import numpy as np
import streamlit as st

from src.io_data import load_parquet_or_csv, prepare_forecast, prepare_profile

# ----------------------------
# Paths (yeni düzen)
# ----------------------------
DATA_DIR = os.getenv("DATA_DIR", "data")

FC_CANDIDATES = [
    os.path.join(DATA_DIR, "forecast_7d.parquet"),
    os.path.join("deploy", "full_fc.parquet"),
    os.path.join(DATA_DIR, "full_fc.parquet"),
    "full_fc.parquet",
]
GP_CANDIDATES = [
    os.path.join(DATA_DIR, "geoid_profile.parquet"),
    os.path.join("deploy", "geoid_profile.parquet"),
    os.path.join(DATA_DIR, "geoid_profile.csv"),
    "geoid_profile.parquet",
    "geoid_profile.csv",
]

def _first_existing(paths: list[str]) -> str | None:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def _pick(df: pd.DataFrame, *names: str) -> str | None:
    """Kolon adı toleranslı seçici."""
    if df is None or df.empty:
        return None
    cols = {c.lower(): c for c in df.columns}
    for n in names:
        if n in df.columns:
            return n
        if n.lower() in cols:
            return cols[n.lower()]
    return None

def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    c_ts = _pick(df, "timestamp", "ts", "datetime", "dt")
    c_date = _pick(df, "date", "day")
    c_hour = _pick(df, "hour", "hour_idx", "hour_of_day", "event_hour")
    c_hrange = _pick(df, "hour_range", "hour_range_3h", "hour_block")

    if c_ts:
        df["timestamp"] = pd.to_datetime(df[c_ts], errors="coerce")
        df["date"] = df["timestamp"].dt.floor("D")
        df["hour"] = df["timestamp"].dt.hour.astype("Int64")
    else:
        # date + hour varsa timestamp üret
        if c_date:
            df["date"] = pd.to_datetime(df[c_date], errors="coerce").dt.floor("D")
        else:
            df["date"] = pd.NaT

        if c_hour:
            df["hour"] = pd.to_numeric(df[c_hour], errors="coerce").astype("Int64").clip(0, 23)
        else:
            # hour_range varsa başlangıç saatini çek
            if c_hrange:
                def parse_start(x):
                    if pd.isna(x):
                        return np.nan
                    s = str(x).strip().replace("–", "-").replace("—", "-")
                    if "-" not in s:
                        return np.nan
                    a = s.split("-", 1)[0].strip()
                    try:
                        return int(a)
                    except Exception:
                        return np.nan
                df["hour"] = df[c_hrange].map(parse_start).astype("Int64")
            else:
                df["hour"] = pd.NA

        if "date" in df.columns:
            df["timestamp"] = df["date"] + pd.to_timedelta(df["hour"].fillna(0).astype(int), unit="h")

    # hour_range yoksa üret (3 saatlik gibi göstermek için opsiyonel)
    if _pick(df, "hour_range") is None:
        if "hour" in df.columns:
            h = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int)
            # 3 saatlik blok başlangıcı
            h0 = (h // 3) * 3
            h1 = (h0 + 3).clip(upper=24)
            df["hour_range"] = h0.astype(str).str.zfill(2) + "–" + h1.astype(str).str.zfill(2)

    return df

def _ensure_core_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # GEOID
    c_geoid = _pick(df, "GEOID", "geoid", "cell_id", "id")
    if c_geoid:
        df["GEOID"] = df[c_geoid].astype(str).str.replace(".0", "", regex=False)
    else:
        df["GEOID"] = "0"

    # p_event
    c_p = _pick(df, "p_event", "risk_prob", "prob", "probability", "p_stack", "risk_score", "score", "risk")
    if c_p:
        p = pd.to_numeric(df[c_p], errors="coerce")
        # 0–100 geldiyse 0–1'e çevir
        if pd.notna(p.max()) and p.max() > 1.0:
            p = p.clip(0, 100) / 100.0
        df["p_event"] = p.clip(0.0, 1.0)
    else:
        df["p_event"] = np.nan

    # expected_count
    c_ec = _pick(df, "expected_count", "expected_crimes", "expected", "lambda", "mu")
    if c_ec:
        df["expected_count"] = pd.to_numeric(df[c_ec], errors="coerce")
    else:
        df["expected_count"] = np.nan

    # harm_score
    c_hs = _pick(df, "harm_score", "o_chf_score", "harm", "harm_index")
    if c_hs:
        hs = pd.to_numeric(df[c_hs], errors="coerce")
        if pd.notna(hs.max()) and hs.max() > 1.0:
            hs = hs.clip(0, 100) / 100.0
        df["harm_score"] = hs.clip(0.0, 1.0)
    else:
        df["harm_score"] = np.nan

    # expected_harm (yoksa üret)
    c_eh = _pick(df, "expected_harm", "exp_harm", "harm_expected")
    if c_eh:
        df["expected_harm"] = pd.to_numeric(df[c_eh], errors="coerce")
    else:
        # harm_score yoksa bile en azından expected_count üzerinden boş kalmasın
        if df["expected_count"].notna().any() and df["harm_score"].notna().any():
            df["expected_harm"] = df["expected_count"] * df["harm_score"]
        else:
            df["expected_harm"] = np.nan

    # risk_level / risk_decile (opsiyonel)
    if _pick(df, "risk_level") is None and df["p_event"].notna().any():
        # basit 5 seviye
        q = df["p_event"].quantile([0.2, 0.4, 0.6, 0.8]).values
        def lvl(x):
            if pd.isna(x): return ""
            if x < q[0]: return "Çok Düşük"
            if x < q[1]: return "Düşük"
            if x < q[2]: return "Orta"
            if x < q[3]: return "Yüksek"
            return "Çok Yüksek"
        df["risk_level"] = df["p_event"].map(lvl)

    if _pick(df, "risk_decile") is None and df["p_event"].notna().any():
        try:
            df["risk_decile"] = pd.qcut(df["p_event"], 10, labels=False, duplicates="drop") + 1
        except Exception:
            df["risk_decile"] = pd.NA

    # top1_category (opsiyonel)
    c_top1 = _pick(df, "top1_category", "top_category", "category_top1")
    if c_top1:
        df["top1_category"] = df[c_top1].astype(str)
    else:
        df["top1_category"] = ""

    return df

@st.cache_data(show_spinner=False)
def _load() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    fc_path = _first_existing(FC_CANDIDATES)
    gp_path = _first_existing(GP_CANDIDATES)

    meta = {"fc_path": fc_path, "gp_path": gp_path}

    if not fc_path:
        return pd.DataFrame(), pd.DataFrame(), meta

    fc = load_parquet_or_csv(fc_path)
    gp = load_parquet_or_csv(gp_path) if gp_path else pd.DataFrame()

    # Hazır fonksiyonların varsa (senin repo)
    if not gp.empty:
        try:
            gp = prepare_profile(gp)
        except Exception:
            pass
    if not fc.empty:
        try:
            fc = prepare_forecast(fc, gp) if not gp.empty else fc
        except Exception:
            pass

    # Ek normalize (kolon farkları için)
    fc = _ensure_datetime(fc)
    fc = _ensure_core_cols(fc)

    return fc, gp, meta

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="🔮 Suç / Zarar Tahmini", layout="wide")

fc, gp, meta = _load()

st.title("🔮 Suç / Zarar Tahmini")
with st.expander("📌 Veri kaynağı (debug)", expanded=False):
    st.write(meta)

if fc.empty:
    st.error("Forecast verisi bulunamadı. (forecast_7d.parquet / deploy/full_fc.parquet kontrol et)")
    st.stop()

# Controls
rank_mode = st.radio(
    "Sıralama",
    ["Zarara göre sırala (expected_harm)", "Risk olasılığına göre sırala (p_event)"],
    index=0,
    horizontal=True,
)
rank_col = "expected_harm" if rank_mode.startswith("Zarara") else "p_event"

scope = st.radio("Kapsam", ["Tüm SF", "Tek GEOID"], index=0, horizontal=True)

geoids = sorted(fc["GEOID"].astype(str).unique().tolist())
sel_geoid = None
if scope == "Tek GEOID":
    sel_geoid = st.selectbox("GEOID", geoids, index=0)

# date filters
dates = sorted(pd.to_datetime(fc["date"], errors="coerce").dropna().dt.normalize().unique())
if not dates:
    st.warning("Tarih kolonu üretilemedi. Veride date/timestamp yok gibi görünüyor.")
    d_from = d_to = None
else:
    d_from, d_to = st.select_slider("Tarih aralığı", options=dates, value=(dates[0], dates[-1]))

# hour filters
hours = sorted(fc["hour_range"].astype(str).dropna().unique().tolist())
sel_hours = st.multiselect("Saat bandı", hours, default=hours)

K = st.slider("Top-K", 20, 5000, 200, 10)

# ----------------------------
# Filter + Rank
# ----------------------------
df = fc.copy()

if d_from is not None and d_to is not None:
    df = df[
        (df["date"].dt.normalize() >= pd.Timestamp(d_from))
        & (df["date"].dt.normalize() <= pd.Timestamp(d_to))
    ]

if sel_hours:
    df = df[df["hour_range"].astype(str).isin(sel_hours)]

if sel_geoid:
    df = df[df["GEOID"].astype(str) == str(sel_geoid)]

# sıralama kolonunu garanti et
if rank_col not in df.columns:
    st.warning(f"'{rank_col}' kolonu yok. p_event ile sıralıyorum.")
    rank_col = "p_event"

df = df.sort_values(rank_col, ascending=False).head(K)

cols = [c for c in [
    "date", "hour_range", "GEOID",
    "risk_level", "risk_decile",
    "p_event", "expected_count",
    "harm_score", "expected_harm",
    "top1_category",
] if c in df.columns]

st.dataframe(df[cols], use_container_width=True, height=560)

csv_bytes = df[cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Listeyi indir (CSV)",
    data=csv_bytes,
    file_name="forecast_ranked_topk.csv",
    mime="text/csv",
)
