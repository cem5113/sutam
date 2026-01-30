# SUTAM — Suç & Suç Zarar Tahmini (Operasyon Paneli)
# Risk = mekân
# Zarar = etki
# Zaman = aksiyon

from __future__ import annotations

import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# -----------------------------------------------------------------------------
# SAFE IMPORT
# -----------------------------------------------------------------------------
try:
    from src.io_data import load_parquet_or_csv, prepare_forecast
except Exception as e:
    load_parquet_or_csv = None
    prepare_forecast = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


# =============================================================================
# PATHS / CONSTANTS
# =============================================================================
DATA_DIR = os.getenv("DATA_DIR", "data").rstrip("/")
FC_FILES = [
    f"{DATA_DIR}/forecast_7d.parquet",
    f"{DATA_DIR}/full_fc.parquet",
    "deploy/full_fc.parquet",
]
GEOJSON_PATH = os.getenv("GEOJSON_PATH", "data/sf_cells.geojson")
TARGET_TZ = "America/Los_Angeles"

LIKERT = {
    1: ("Çok Düşük",  [46, 204, 113]),
    2: ("Düşük",      [88, 214, 141]),
    3: ("Orta",       [241, 196, 15]),
    4: ("Yüksek",     [230, 126, 34]),
    5: ("Çok Yüksek", [192, 57, 43]),
}
DEFAULT_FILL = [220, 220, 220]


# =============================================================================
# CSS (tooltip küçük ve net)
# =============================================================================
def _css():
    st.markdown("""
    <style>
      .deckgl-tooltip{
        max-width:320px!important;
        padding:10px 12px!important;
        line-height:1.25;
        border-radius:12px;
      }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# HELPERS
# =============================================================================
def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def _digits11(x):
    s = "".join(c for c in str(x) if c.isdigit())
    return s.zfill(11) if s else ""

def _pick(df, names):
    cols = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in cols:
            return cols[n.lower()]
    return None

def _quintile(v):
    v = pd.to_numeric(v, errors="coerce")
    if v.notna().sum() < 10:
        return pd.Series([3]*len(v), index=v.index)
    return pd.qcut(v.rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)


# =============================================================================
# LOAD
# =============================================================================
@st.cache_data(show_spinner=False)
def load_fc():
    p = _first_existing(FC_FILES)
    if not p or load_parquet_or_csv is None:
        return pd.DataFrame()
    df = load_parquet_or_csv(p)
    if prepare_forecast:
        try:
            df = prepare_forecast(df, gp=None)
        except Exception:
            pass
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_geojson():
    if os.path.exists(GEOJSON_PATH):
        with open(GEOJSON_PATH,"r",encoding="utf-8") as f:
            return json.load(f)
    return {}


# =============================================================================
# SAHA NOTU (1–2 cümle)
# =============================================================================
def saha_notu(r):
    notes = []
    h = int(r.get("hour", -1))

    if r.get("bar_count",0)>0 and h>=18:
        notes.append("Akşam saatleri ve bar yoğunluğu mevcut.")

    if r.get("school_count",0)>0 and 14<=h<=17:
        notes.append("Okul çıkış saatlerine denk geliyor.")

    if r.get("neighbor_crime_7d",0)>0:
        notes.append("Komşu bölgelerde son günlerde artış var.")

    return " ".join(notes[:2])


# =============================================================================
# GEOJSON ENRICH
# =============================================================================
def enrich_geojson(gj, df):
    if not gj or df.empty:
        return gj

    d = df.copy()
    d["geoid"] = d[_pick(d,["geoid","GEOID"])].map(_digits11)

    risk_col = _pick(d,["risk_score","p_event","risk_prob"])
    harm_col = _pick(d,["expected_harm","harm"])

    d["risk"] = _quintile(d[risk_col]) if risk_col else 3
    d["harm"] = _quintile(d[harm_col]) if harm_col else 3
    d["color"] = d["risk"].map(lambda k: LIKERT[k][1])
    d["warn"] = d["harm"].eq(5)

    d["note"] = d.apply(saha_notu, axis=1)

    d = (
        d.sort_values(["harm","risk"], ascending=False)
         .drop_duplicates("geoid", keep="first")
         .set_index("geoid")
    )

    feats=[]
    for f in gj.get("features",[]):
        p = dict(f.get("properties") or {})
        gid = _digits11(p.get("geoid") or p.get("GEOID"))
        p.update({
            "fill_color": DEFAULT_FILL,
            "note": "",
            "harm_icon": "",
            "display_id": gid
        })
        if gid in d.index:
            r = d.loc[gid]
            p["fill_color"]=r["color"]
            p["note"]=r["note"]
            p["harm_icon"]="⚠️" if r["warn"] else ""
        feats.append({**f,"properties":p})

    return {**gj,"features":feats}


# =============================================================================
# KRİTİK SAAT (ALT PANEL)
# =============================================================================
def critical_hours(df, geoid):
    d=df[df["geoid"]==geoid]
    if d.empty:
        return None
    harm_col=_pick(d,["expected_harm","harm"])
    if not harm_col:
        return None
    g=d.groupby("hour")[harm_col].mean().sort_values(ascending=False)
    hrs=sorted(g.head(3).index.tolist())
    return f"{min(hrs):02d}:00 – {max(hrs)+1:02d}:00"


# =============================================================================
# MAP
# =============================================================================
def draw_map(gj):
    layer = pdk.Layer(
        "GeoJsonLayer",
        gj,
        filled=True,
        stroked=True,
        get_fill_color="properties.fill_color",
        get_line_color=[80,80,80],
        pickable=True,
        opacity=0.65,
    )

    tooltip = {
        "html":(
            "<b>GEOID:</b> {display_id}<br/>"
            "{harm_icon} <b>Saha Notu</b><br/>{note}"
        ),
        "style":{"backgroundColor":"#0b1220","color":"white"}
    }

    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(
                latitude=37.7749, longitude=-122.4194, zoom=10
            ),
            tooltip=tooltip,
            map_style="light",
        ),
        use_container_width=True
    )


# =============================================================================
# PAGE ENTRY
# =============================================================================
def render_anlik_risk_haritasi():
    _css()
    st.markdown("# 🗺️ Suç & Suç Zarar Tahmini")

    if _IMPORT_ERR:
        st.error("Veri modülü yüklenemedi.")
        st.code(repr(_IMPORT_ERR))
        return

    fc=load_fc()
    if fc.empty:
        st.error("Forecast verisi yok.")
        return

    # --- zaman seçimi
    now=datetime.now(ZoneInfo(TARGET_TZ))
    with st.container():
        c1,c2=st.columns([2,3])
        with c1:
            sel_date=st.date_input("📅 Tarih",value=now.date())
        with c2:
            hr_min,hr_max=st.slider("⏰ Saat aralığı",0,23,(18,21))

    fc["date"]=pd.to_datetime(fc[_pick(fc,["date"])])
    fc["geoid"]=fc[_pick(fc,["geoid","GEOID"])].map(_digits11)
    fc["hour"]=fc[_pick(fc,["hour","hour_of_day"])]

    df=fc[
        (fc["date"].dt.date==sel_date)&
        (fc["hour"]>=hr_min)&
        (fc["hour"]<=hr_max)
    ].copy()

    if df.empty:
        st.warning("Bu zaman aralığında veri yok.")
        return

    # --- GLOBAL UYARILAR (SIDEBAR)
    harm_col=_pick(df,["expected_harm","harm"])
    if harm_col:
        df["harm_lvl"]=_quintile(df[harm_col])
        crit=df[df["harm_lvl"]==5].head(3)
        if not crit.empty:
            st.sidebar.markdown("### 🚨 Kritik Durumlar")
            for _,r in crit.iterrows():
                st.sidebar.error(
                    f"{int(r['hour']):02d}:00 • {r['geoid']}\n{saha_notu(r) or 'Çok yüksek zarar riski.'}"
                )

    gj=load_geojson()
    gj=enrich_geojson(gj,df)
    draw_map(gj)

    # --- ALT PANEL: GEOID → NE ZAMAN?
    sel=st.text_input("🔎 GEOID (detay görmek için)")
    if sel:
        ch=critical_hours(fc,_digits11(sel))
        if ch:
            st.info(f"⏰ Bu bölgede en kritik zaman: **{ch}**")
