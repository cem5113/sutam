# pages/3_🚓_Devriye_Planlama.py
import os
import streamlit as st
import pandas as pd

from src.io_data import load_parquet_or_csv, prepare_forecast, prepare_profile
from src.patrol_planner import equal_distribution_plan, build_band

DATA_DIR = os.getenv("DATA_DIR", "data")
FC_PATH  = os.path.join(DATA_DIR, "forecast_7d.parquet")
GP_PATH  = os.path.join(DATA_DIR, "geoid_profile.parquet")

@st.cache_data(show_spinner=False)
def _load():
    fc = load_parquet_or_csv(FC_PATH)
    gp = load_parquet_or_csv(GP_PATH)
    gp = prepare_profile(gp) if not gp.empty else gp
    fc = prepare_forecast(fc, gp) if not fc.empty else fc
    return fc

fc = _load()

st.title("🚓 Devriye Planlama")

st.subheader("A) expected_harm tabanlı TOP-K hücreler")
K = st.slider("Top-K", 20, 3000, 200, 10)
rank_col = st.selectbox("Sıralama kolonu", ["expected_harm","p_event","expected_count"], index=0)

df_top = fc.copy()
if rank_col not in df_top.columns:
    rank_col = "expected_harm"
df_top = df_top.sort_values(rank_col, ascending=False).head(K)

cols = [c for c in ["date","hour_range","GEOID","risk_level","expected_harm","p_event","expected_count","top1_category","risk_decile"] if c in df_top.columns]
st.dataframe(df_top[cols], use_container_width=True, height=420)

st.subheader("B) Düşük / Orta / Yüksek risk alanları")
tmp = df_top.copy()
tmp["risk_band"] = build_band(tmp)

c1, c2, c3 = st.columns(3)
for band, col in [("HIGH", c1), ("MED", c2), ("LOW", c3)]:
    with col:
        st.markdown(f"**{band}**")
        t = tmp[tmp["risk_band"] == band].head(30)
        st.dataframe(t[cols], use_container_width=True, height=280)

st.subheader("C) Eşit dağılımlı patrol senaryosu")
c1, c2, c3 = st.columns(3)
with c1:
    num_units = st.number_input("Devriye ekibi sayısı", min_value=1, max_value=200, value=10, step=1)
with c2:
    cells_per_unit = st.number_input("Ekip başı hücre", min_value=1, max_value=100, value=10, step=1)
with c3:
    max_cells_per_geoid = st.number_input("Aynı GEOID max hücre", min_value=1, max_value=50, value=3, step=1)

st.caption("Band kotaları (toplam = 1.0 olacak şekilde normalize edilir)")
share_high = st.slider("HIGH payı", 0.0, 1.0, 0.5, 0.05)
share_med  = st.slider("MED payı",  0.0, 1.0, 0.3, 0.05)
share_low  = st.slider("LOW payı",  0.0, 1.0, 0.2, 0.05)

plan = equal_distribution_plan(
    fc,
    rank_col="expected_harm",
    num_units=int(num_units),
    cells_per_unit=int(cells_per_unit),
    share_high=float(share_high),
    share_med=float(share_med),
    share_low=float(share_low),
    max_cells_per_geoid=int(max_cells_per_geoid),
)

st.write(f"✅ Plan üretildi: {len(plan):,} hücre")
pcols = [c for c in ["unit_id","risk_band","date","hour_range","GEOID","risk_level","expected_harm","p_event","expected_count","top1_category"] if c in plan.columns]
st.dataframe(plan[pcols], use_container_width=True, height=520)

csv_bytes = plan[pcols].to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Devriye planını indir (CSV)", data=csv_bytes, file_name="patrol_plan_equal_distribution.csv", mime="text/csv")
