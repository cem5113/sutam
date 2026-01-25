# app.py — FULL REVIZE (deploy/data uyumlu + fallback + cache-bust)
import os
import json
import streamlit as st
import pandas as pd

from src.io_data import load_parquet_or_csv, prepare_forecast, prepare_profile

st.set_page_config(page_title="SF Ops Crime Forecast", layout="wide")

# ============================================================
# PATHS (standart isimler)
# - DATA_DIR:
#   - local: "data" (default)
#   - url  : "https://raw.githubusercontent.com/cem5113/sutam/main/deploy"
# ============================================================
DATA_DIR = os.getenv("DATA_DIR", "data").rstrip("/")

# primary expected paths (standard)
FC_PATH = f"{DATA_DIR}/forecast_7d.parquet"
GP_PATH = f"{DATA_DIR}/geoid_profile.parquet"
OPS_DIR = f"{DATA_DIR}/ops"

# fallbacks (deploy naming)
FC_FALLBACKS = [
    f"{DATA_DIR}/full_fc.parquet",                # if DATA_DIR=deploy url or deploy folder
    "deploy/full_fc.parquet",                     # if DATA_DIR=local repo root
    "data/full_fc.parquet",                       # if someone saved as full_fc in data/
]

GP_FALLBACKS = [
    f"{DATA_DIR}/geoid_profile.parquet",
    f"{DATA_DIR}/geoid_profile.csv",
    "deploy/geoid_profile.parquet",
    "deploy/geoid_profile.csv",
    "data/geoid_profile.parquet",
    "data/geoid_profile.csv",
]

AUDIT_CANDIDATES = [
    f"{DATA_DIR}/deploy_audit.json",
    "deploy/deploy_audit.json",
    "data/deploy_audit.json",
]

# ============================================================
# UI — sidebar
# ============================================================
st.sidebar.title("⚙️ Veri Kaynakları")
st.sidebar.caption(f"DATA_DIR: {DATA_DIR}")

# ============================================================
# Helpers
# ============================================================
def _is_url(p: str) -> bool:
    return p.startswith("http://") or p.startswith("https://")

def _try_read_any(path: str) -> pd.DataFrame:
    """
    Robust read:
      - URL -> pandas directly (csv/parquet)
      - local -> use load_parquet_or_csv (your existing helper)
    """
    try:
        if _is_url(path):
            if path.lower().endswith(".parquet"):
                return pd.read_parquet(path)
            elif path.lower().endswith(".csv"):
                return pd.read_csv(path)
            else:
                # last resort: try parquet then csv
                try:
                    return pd.read_parquet(path)
                except Exception:
                    return pd.read_csv(path)
        else:
            # local
            if os.path.exists(path):
                return load_parquet_or_csv(path)
            return pd.DataFrame()
    except Exception as e:
        # return empty but keep info
        st.sidebar.warning(f"⚠️ Okuma hatası: {os.path.basename(path)} → {type(e).__name__}")
        return pd.DataFrame()

def _first_existing(candidates):
    """
    candidates: list[str]
    returns: (chosen_path or None)
    """
    for p in candidates:
        if _is_url(p):
            # URL existence check is expensive; skip and let reader try
            return p
        if os.path.exists(p):
            return p
    return None

def _load_audit_tag() -> str:
    """
    Cache bust key: deploy_audit.json içinden deploy_time_utc alınır.
    Bulunamazsa sabit 'no_audit'.
    """
    for p in AUDIT_CANDIDATES:
        try:
            if _is_url(p):
                # try read remote json
                j = pd.read_json(p, typ="series")
                # if returns Series-like
                if "deploy_time_utc" in j:
                    return str(j["deploy_time_utc"])
                return "audit_url"
            else:
                if os.path.exists(p):
                    with open(p, "r", encoding="utf-8") as f:
                        obj = json.load(f)
                    return str(obj.get("deploy_time_utc", "audit_local"))
        except Exception:
            continue
    return "no_audit"

def _resolve_forecast_path() -> str:
    # prefer standard
    if _is_url(FC_PATH):
        return FC_PATH  # let reader try
    if os.path.exists(FC_PATH):
        return FC_PATH
    # fallback list
    cand = [FC_PATH] + FC_FALLBACKS
    p = _first_existing(cand)
    return p or FC_PATH

def _resolve_profile_path() -> str:
    if _is_url(GP_PATH):
        return GP_PATH
    if os.path.exists(GP_PATH):
        return GP_PATH
    cand = [GP_PATH] + GP_FALLBACKS
    p = _first_existing(cand)
    return p or GP_PATH

# ============================================================
# LOAD (cache)
# - audit_tag cache bust için key’e girer
# ============================================================
@st.cache_data(show_spinner=False)
def _load_all(fc_path: str, gp_path: str, audit_tag: str):
    fc = _try_read_any(fc_path)
    gp = _try_read_any(gp_path)

    gp = prepare_profile(gp) if not gp.empty else gp
    fc = prepare_forecast(fc, gp) if not fc.empty else fc
    return fc, gp

audit_tag = _load_audit_tag()
resolved_fc = _resolve_forecast_path()
resolved_gp = _resolve_profile_path()

st.sidebar.caption(f"FC_PATH: {resolved_fc}")
st.sidebar.caption(f"GP_PATH: {resolved_gp}")
st.sidebar.caption(f"audit_tag: {audit_tag}")

fc, gp = _load_all(resolved_fc, resolved_gp, audit_tag)

# ============================================================
# Guards
# ============================================================
if fc.empty:
    st.error(
        "Forecast dosyası bulunamadı/boş.\n\n"
        f"- Aranan (standart): {FC_PATH}\n"
        f"- Seçilen: {resolved_fc}\n\n"
        "Çözüm:\n"
        "1) Repo içine `data/forecast_7d.parquet` koy (full_fc kopyası), veya\n"
        "2) DATA_DIR'i `.../deploy` raw URL yap, veya\n"
        "3) `full_fc.parquet` kullanımı için fallback açık (bu revizede var)."
    )
    st.stop()

# ============================================================
# GLOBAL INFO
# ============================================================
st.sidebar.success("✅ Forecast yüklendi")
st.sidebar.write("Satır:", f"{len(fc):,}")

if "date" in fc.columns:
    st.sidebar.write("Tarih aralığı:", f"{pd.to_datetime(fc['date']).min().date()} → {pd.to_datetime(fc['date']).max().date()}")
else:
    st.sidebar.warning("⚠️ fc içinde `date` kolonu yok (prepare_forecast kontrol)")

if gp.empty:
    st.sidebar.warning("⚠️ geoid_profile yok (kategori/profil kartı kısıtlı)")
else:
    st.sidebar.success("✅ GEOID profile yüklendi")
    st.sidebar.write("GEOID:", f"{gp['GEOID'].nunique():,}" if "GEOID" in gp.columns else f"{len(gp):,} satır")

st.sidebar.divider()
st.sidebar.caption("Sayfalar soldaki menüde: Anlık Tahmin / Suç-Zarar / Devriye / Raporlar")

# ============================================================
# HOME
# ============================================================
st.title("SF Crime Forecast — Kolluk Operasyon Paneli")
st.write("Sol menüden sayfa seçin. (Bu ana sayfa bilgilendirme amaçlı)")
st.info("Not: Model metrikleri tezde. Burada operasyonel çıktı (Top-K, gerekçe, aksiyon) odaklıyız.")
