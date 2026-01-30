# pages/Anlik_Risk_Haritasi.py
# SUTAM — Anlık Risk Haritası (Operasyonel • PATCH’li)
# Amaç: Ne zaman, nerede, neye dikkat etmeli?

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
    _IMPORT_SRC_ERR = e
else:
    _IMPORT_SRC_ERR = None


# =============================================================================
# PATHS / CONSTANTS
# =============================================================================
DATA_DIR = os.getenv("DATA_DIR", "data").rstrip("/")
FC_CANDIDATES = [
    f"{DATA_DIR}/forecast_7d.parquet",
    f"{DATA_DIR}/full_fc.parquet",
    "data/forecast_7d.parquet",
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
# CSS (tooltip sabit, scroll yok)
# =============================================================================
def _apply_tooltip_css():
    st.markdown(
        """
        <style>
          .deckgl-tooltip {
            max-width: 340px !important;
            padding: 10px 12px !important;
            line-height: 1.25 !important;
            border-radius: 12px !important;
          }
          .deckgl-tooltip {
            transform: translate(12px, 12px) !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# HELPERS
# =============================================================================
def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def _digits11(x):
    s = "".join(ch for ch in str(x) if ch.isdigit())
    return s.zfill(11) if s else ""

def _pick_col(df, names):
    cols = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in cols:
            return cols[n.lower()]
    return None

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def _fmt_expected(x):
    v = _safe_float(x)
    if not np.isfinite(v):
        return "—"
    lo, hi = int(np.floor(v)), int(np.ceil(v))
    return f"~{lo}" if lo == hi else f"~{lo}–{hi}"

def _parse_range(tok):
    if not isinstance(tok, str) or "-" not in tok:
        return None
    a, b = tok.split("-", 1)
    try:
        s, e = int(a.strip()), int(b.strip())
    except Exception:
        return None
    return (max(0, s), min(24, e))


# =============================================================================
# LOADERS
# =============================================================================
@st.cache_data(show_spinner=False)
def load_forecast():
    p = _first_existing(FC_CANDIDATES)
    if not p or load_parquet_or_csv is None:
        return pd.DataFrame()

    fc = load_parquet_or_csv(p)
    if fc is None or fc.empty:
        return pd.DataFrame()

    if prepare_forecast:
        try:
            fc = prepare_forecast(fc, gp=None)
        except Exception:
            pass
    return fc

@st.cache_data(show_spinner=False)
def load_geojson():
    if os.path.exists(GEOJSON_PATH):
        with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# =============================================================================
# QUINTILE
# =============================================================================
def _compute_likert_quintiles(series):
    v = pd.to_numeric(series, errors="coerce")
    if v.notna().sum() < 10:
        return pd.Series([3] * len(v), index=v.index)
    return pd.qcut(v.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)


# =============================================================================
# SAHA NOTU (1–2 cümle)
# =============================================================================
def saha_notu(r):
    h = int(r.get("hour", -1))
    notes = []

    if r.get("bar_count", 0) > 0 and h >= 18:
        notes.append("Akşam ve bar yoğunluğu mevcut.")

    if r.get("school_count", 0) > 0 and 14 <= h <= 17:
        notes.append("Okul çıkış saatleri dikkat.")

    if r.get("neighbor_crime_7d", 0) > 0:
        notes.append("Komşu bölgelerde artış var.")

    return " ".join(notes[:2])


# =============================================================================
# GEOJSON ENRICH (PATCH’li)
# =============================================================================
def enrich_geojson(gj, df):
    if not gj or df.empty:
        return gj

    d = df.copy()
    d["geoid"] = d[_pick_col(d, ["GEOID", "geoid"])].map(_digits11)

    risk_col = _pick_col(d, ["risk_score", "p_event", "risk_prob"])
    harm_col = _pick_col(d, ["expected_harm", "harm"])
    exp_col = _pick_col(d, ["expected_count", "expected_crimes"])

    d["risk_likert"] = _compute_likert_quintiles(d[risk_col]) if risk_col else 3
    d["harm_likert"] = _compute_likert_quintiles(d[harm_col]) if harm_col else 3

    d["harm_icon"] = d["harm_likert"].apply(lambda x: "⚠️" if x == 5 else "")
    d["fill_color"] = d["risk_likert"].map(lambda k: LIKERT[int(k)][1])
    d["expected_txt"] = d[exp_col].map(_fmt_expected) if exp_col else "—"
    d["saha_notu"] = d.apply(saha_notu, axis=1)

    d = (
        d.sort_values(["harm_likert", "risk_likert"], ascending=False)
        .drop_duplicates("geoid", keep="first")
        .set_index("geoid")
    )

    feats = []
    for f in gj.get("features", []):
        p = dict(f.get("properties") or {})
        gid = _digits11(p.get("geoid") or p.get("GEOID"))

        p.update({
            "display_id": gid,
            "fill_color": DEFAULT_FILL,
            "expected_txt": "—",
            "harm_icon": "",
            "saha_notu": ""
        })

        if gid in d.index:
            r = d.loc[gid]
            p["fill_color"] = r["fill_color"]
            p["expected_txt"] = r["expected_txt"]
            p["harm_icon"] = r["harm_icon"]
            p["saha_notu"] = r["saha_notu"]

        feats.append({**f, "properties": p})

    return {**gj, "features": feats}


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
        get_line_color=[80, 80, 80],
        pickable=True,
        opacity=0.65,
    )

    tooltip = {
        "html": (
            "<b>GEOID:</b> {display_id}<br/>"
            "<b>Beklenen suç:</b> {expected_txt}<br/>"
            "<b>Beklenen zarar:</b> {harm_icon}<br/>"
            "<b>Saha notu:</b> {saha_notu}"
        ),
        "style": {"backgroundColor": "#0b1220", "color": "white"},
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
        use_container_width=True,
    )


# =============================================================================
# PAGE ENTRYPOINT
# =============================================================================
def render_anlik_risk_haritasi():
    _apply_tooltip_css()

    st.markdown("# 🗺️ Anlık Risk Haritası")
    st.caption(
        "Risk mekânı, zarar etkiyi gösterir. Zaman seçimi ile vardiya öncesi değerlendirme yapılabilir."
    )

    if _IMPORT_SRC_ERR:
        st.error("Veri modülü import edilemedi.")
        st.code(repr(_IMPORT_SRC_ERR))
        return

    fc = load_forecast()
    if fc.empty:
        st.error("Forecast verisi bulunamadı.")
        return

    # -------------------------------------------------------------------------
    # ÜST ZAMAN KONTROLÜ (PATCH)
    # -------------------------------------------------------------------------
    now = datetime.now(ZoneInfo(TARGET_TZ))
    c1, c2 = st.columns([2, 3])
    with c1:
        manual_date = st.date_input("📅 Tarih", value=None)
    with c2:
        hr_range = st.slider("⏰ Saat aralığı", 0, 23, value=(now.hour, min(now.hour + 2, 23)))

    date_col = _pick_col(fc, ["date"])
    hr_col = _pick_col(fc, ["hour", "hour_of_day", "hour_range"])
    geo_col = _pick_col(fc, ["GEOID", "geoid"])

    fc["date"] = pd.to_datetime(fc[date_col], errors="coerce")
    fc["date_norm"] = fc["date"].dt.normalize()
    fc["geoid"] = fc[geo_col].map(_digits11)
    fc["hour"] = fc[hr_col].astype(int)

    sel_date = manual_date if manual_date else now.date()

    df_hr = fc[
        (fc["date_norm"] == pd.Timestamp(sel_date)) &
        (fc["hour"] >= hr_range[0]) &
        (fc["hour"] <= hr_range[1])
    ].copy()

    if df_hr.empty:
        st.warning("Bu zaman dilimi için kayıt yok.")
        return

    # -------------------------------------------------------------------------
    # SIDEBAR — GLOBAL KRİTİKLER (PATCH)
    # -------------------------------------------------------------------------
    harm_col = _pick_col(df_hr, ["expected_harm", "harm"])
    if harm_col:
        df_hr["_harm_lik"] = _compute_likert_quintiles(df_hr[harm_col])
        crit = df_hr[df_hr["_harm_lik"] == 5].head(3)
        if not crit.empty:
            st.sidebar.markdown("### 🚨 Kritik Durumlar")
            for _, r in crit.iterrows():
                st.sidebar.error(
                    f"{int(r['hour']):02d}:00 • {r['geoid']}\n"
                    f"{saha_notu(r) or 'Çok yüksek zarar riski.'}"
                )

    # -------------------------------------------------------------------------
    # MAP
    # -------------------------------------------------------------------------
    gj = load_geojson()
    gj_enriched = enrich_geojson(gj, df_hr)
    draw_map(gj_enriched)

    # -------------------------------------------------------------------------
    # ALT PANEL — SEÇİLİ GEOID → NE ZAMAN? (PATCH)
    # -------------------------------------------------------------------------
    st.divider()
    st.subheader("⏰ Seçili Bölge İçin Kritik Zaman")

    sel_geoid = st.text_input("GEOID (isteğe bağlı)", "")
    if sel_geoid:
        gid = _digits11(sel_geoid)
        df_g = fc[fc["geoid"] == gid]
        if harm_col and not df_g.empty:
            g = (
                df_g.groupby("hour")[harm_col]
                .mean()
                .sort_values(ascending=False)
            )
            hrs = sorted(g.head(3).index.tolist())
            if hrs:
                st.info(
                    f"Bu bölgede en kritik zaman: "
                    f"**{min(hrs):02d}:00 – {max(hrs)+1:02d}:00**"
                )
