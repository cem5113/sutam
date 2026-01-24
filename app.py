# app.py
import os
import streamlit as st
import pandas as pd

from src.io_data import load_parquet_or_csv, prepare_forecast, prepare_profile

st.set_page_config(page_title="SF Ops Crime Forecast", layout="wide")

# --- PATHS (standart isimler) ---
DATA_DIR = os.getenv("DATA_DIR", "data")
FC_PATH  = os.path.join(DATA_DIR, "forecast_7d.parquet")
GP_PATH  = os.path.join(DATA_DIR, "geoid_profile.parquet")
OPS_DIR  = os.path.join(DATA_DIR, "ops")  # BLOK-9 opsiyonel

st.sidebar.title("⚙️ Veri Kaynakları")
st.sidebar.caption(f"DATA_DIR: {DATA_DIR}")

# --- LOAD (cache) ---
@st.cache_data(show_spinner=False)
def _load_all(fc_path, gp_path):
    fc = load_parquet_or_csv(fc_path)
    gp = load_parquet_or_csv(gp_path)
    gp = prepare_profile(gp) if not gp.empty else gp
    fc = prepare_forecast(fc, gp) if not fc.empty else fc
    return fc, gp

fc, gp = _load_all(FC_PATH, GP_PATH)

if fc.empty:
    st.error(f"Forecast dosyası bulunamadı/boş: {FC_PATH}\nfull_fc'yi bu path'e kaydet.")
    st.stop()

# --- GLOBAL INFO ---
st.sidebar.success("✅ Forecast yüklendi")
st.sidebar.write("Satır:", f"{len(fc):,}")
st.sidebar.write("Tarih aralığı:", f"{fc['date'].min().date()} → {fc['date'].max().date()}")

if gp.empty:
    st.sidebar.warning("⚠️ geoid_profile yok (kategori/profil kartı kısıtlı)")

st.sidebar.divider()
st.sidebar.caption("Sayfalar soldaki menüde: Anlık Tahmin / Suç-Zarar / Devriye / Raporlar")

# --- HOME ---
st.title("SF Crime Forecast — Kolluk Operasyon Paneli")
st.write("Sol menüden sayfa seçin. (Bu ana sayfa bilgilendirme amaçlı)")
st.info("Not: Model metrikleri tezde. Burada operasyonel çıktı (Top-K, gerekçe, aksiyon) odaklıyız.")
