# pages/1_🧭_Anlik_Tahmin.py
import os
import streamlit as st
import pandas as pd
import numpy as np

from src.io_data import load_parquet_or_csv, prepare_forecast, prepare_profile, make_cell_id
from src.ops_rules import reasons_actions_fallback

DATA_DIR = os.getenv("DATA_DIR", "data")
FC_PATH  = os.path.join(DATA_DIR, "forecast_7d.parquet")
GP_PATH  = os.path.join(DATA_DIR, "geoid_profile.parquet")
OPS_DIR  = os.path.join(DATA_DIR, "ops")

@st.cache_data(show_spinner=False)
def _load():
    fc = load_parquet_or_csv(FC_PATH)
    gp = load_parquet_or_csv(GP_PATH)
    gp = prepare_profile(gp) if not gp.empty else gp
    fc = prepare_forecast(fc, gp) if not fc.empty else fc
    return fc, gp

fc, gp = _load()

st.title("🧭 Anlık Tahmin (Ana Sayfa)")

# selections
dates = sorted(fc["date"].dt.normalize().unique())
hours = sorted(fc["hour_range"].astype(str).unique())
geoids = sorted(fc["GEOID"].astype(str).unique())

c1, c2, c3 = st.columns(3)
with c1:
    sel_date = st.selectbox("Seçilen tarih", dates, index=len(dates)-1, format_func=lambda x: x.strftime("%Y-%m-%d"))
with c2:
    sel_hr = st.selectbox("Seçilen saat dilimi", hours, index=0)
with c3:
    sel_geoid = st.selectbox("Seçilen GEOID", geoids, index=0)

row_df = fc[(fc["date"].dt.normalize() == pd.Timestamp(sel_date)) &
            (fc["hour_range"].astype(str) == str(sel_hr)) &
            (fc["GEOID"].astype(str) == str(sel_geoid))].copy()

if row_df.empty:
    st.warning("Bu hücre için forecast kaydı bulunamadı.")
    st.stop()

row = row_df.iloc[0]

# profile row
profile_row = None
if not gp.empty:
    hit = gp[gp["GEOID"].astype(str) == str(sel_geoid)]
    if not hit.empty:
        profile_row = hit.iloc[0]

# BLOK-9 reasons/actions varsa oku (opsiyonel)
reasons = None
action = None
cell_id = f"{sel_geoid}|{pd.Timestamp(sel_date).strftime('%Y-%m-%d')}|{sel_hr}"

ops_reasons_path = os.path.join(OPS_DIR, "ops_reasons.csv")
ops_actions_path = os.path.join(OPS_DIR, "ops_actions.csv")

if os.path.exists(ops_reasons_path) and os.path.exists(ops_actions_path):
    rdf = pd.read_csv(ops_reasons_path)
    adf = pd.read_csv(ops_actions_path)
    # beklenen kolon: cell_id, reason_1..reason_k / action
    if "cell_id" in rdf.columns:
        rr = rdf[rdf["cell_id"].astype(str) == cell_id]
        if not rr.empty:
            rrow = rr.iloc[0].to_dict()
            reasons = [str(v) for k,v in rrow.items() if str(k).startswith("reason") and pd.notna(v) and str(v).strip()]
    if "cell_id" in adf.columns and "action" in adf.columns:
        aa = adf[adf["cell_id"].astype(str) == cell_id]
        if not aa.empty:
            action = str(aa.iloc[0]["action"])

if reasons is None or action is None:
    reasons2, action2 = reasons_actions_fallback(row, profile_row)
    reasons = reasons or reasons2
    action = action or action2

# KPI cards
k1, k2, k3, k4 = st.columns(4)
k1.metric("Risk Seviyesi", str(row.get("risk_level","NA")))
k2.metric("p_event", f"{float(row.get('p_event',np.nan)):.3f}" if pd.notna(row.get("p_event",np.nan)) else "NA")
k3.metric("expected_count", f"{float(row.get('expected_count',np.nan)):.2f}" if pd.notna(row.get("expected_count",np.nan)) else "NA")
k4.metric("expected_harm", f"{float(row.get('expected_harm',np.nan)):.2f}" if pd.notna(row.get("expected_harm",np.nan)) else "NA")

# dominant categories
st.subheader("Baskın suç kategorileri")
cat_show = None
if "top1_category" in row_df.columns and pd.notna(row.get("top1_category", np.nan)):
    cat_show = [str(row.get("top1_category"))]
elif profile_row is not None and pd.notna(profile_row.get("top5_categories", np.nan)):
    s = str(profile_row.get("top5_categories",""))
    # "A:10,B:5,..." -> ilk 3
    parts = s.split(",")[:3]
    cat_show = [p.split(":")[0] for p in parts if p]
else:
    cat_show = ["NA"]

st.write(" • " + " | ".join(cat_show))

# reasons/actions
st.subheader("Gerekçe (insan dili)")
for r in reasons[:4]:
    st.write("•", r)

st.subheader("Önerilen aksiyon")
st.success(action)

# audit
with st.expander("Audit / Teknik (opsiyonel)"):
    audit_cols = [c for c in ["run_id","model_version","forecast_generated_at","audit_tag","risk_decile","risk_score"] if c in row_df.columns]
    if audit_cols:
        st.json({c: (str(row.get(c)) if c not in ["risk_score","risk_decile"] else row.get(c)) for c in audit_cols})
    else:
        st.caption("Audit kolonları yok.")
