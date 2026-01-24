# pages/2_🔮_Suc_Zarar_Tahmini.py
import os
import streamlit as st
import pandas as pd

from src.io_data import load_parquet_or_csv, prepare_forecast, prepare_profile

DATA_DIR = os.getenv("DATA_DIR", "data")
FC_PATH  = os.path.join(DATA_DIR, "forecast_7d.parquet")
GP_PATH  = os.path.join(DATA_DIR, "geoid_profile.parquet")

@st.cache_data(show_spinner=False)
def _load():
    fc = load_parquet_or_csv(FC_PATH)
    gp = load_parquet_or_csv(GP_PATH)
    gp = prepare_profile(gp) if not gp.empty else gp
    fc = prepare_forecast(fc, gp) if not fc.empty else fc
    return fc, gp

fc, gp = _load()

st.title("🔮 Suç / Zarar Tahmini")

# controls
rank_mode = st.radio("Sıralama", ["Zarara göre sırala (expected_harm)", "Risk olasılığına göre sırala (p_event)"], index=0, horizontal=True)
rank_col = "expected_harm" if rank_mode.startswith("Zarara") else "p_event"

scope = st.radio("Kapsam", ["Tüm SF", "Tek GEOID"], index=0, horizontal=True)

geoids = sorted(fc["GEOID"].astype(str).unique())
sel_geoid = None
if scope == "Tek GEOID":
    sel_geoid = st.selectbox("GEOID", geoids, index=0)

# date/hour filters
dates = sorted(fc["date"].dt.normalize().unique())
d_from, d_to = st.select_slider("Tarih aralığı", options=dates, value=(dates[0], dates[-1]))
hours = sorted(fc["hour_range"].astype(str).unique())
sel_hours = st.multiselect("Saat bandı", hours, default=hours)

K = st.slider("Top-K", 20, 2000, 200, 10)

df = fc.copy()
df = df[(df["date"].dt.normalize() >= pd.Timestamp(d_from)) & (df["date"].dt.normalize() <= pd.Timestamp(d_to))]
if sel_hours:
    df = df[df["hour_range"].astype(str).isin(sel_hours)]
if sel_geoid:
    df = df[df["GEOID"].astype(str) == str(sel_geoid)]

df = df.sort_values(rank_col, ascending=False).head(K)

cols = [c for c in ["date","hour_range","GEOID","risk_level","expected_harm","p_event","expected_count","top1_category","risk_decile"] if c in df.columns]
st.dataframe(df[cols], use_container_width=True, height=560)

csv_bytes = df[cols].to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Listeyi indir (CSV)", data=csv_bytes, file_name="forecast_ranked_topk.csv", mime="text/csv")
