# -*- coding: utf-8 -*-
# pages/Anlik_Risk_Haritasi.py
# SUTAM — Anlık Risk Haritası (Operasyonel • OPS_READY kaynağı)
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
# SAFE IMPORT (opsiyonel)
# -----------------------------------------------------------------------------
try:
    from src.io_data import load_parquet_or_csv as _load_parquet_or_csv  # type: ignore
except Exception:
    _load_parquet_or_csv = None


# =============================================================================
# PATHS / CONSTANTS  (✅ YENİ DATA KAYNAĞI)
# =============================================================================
DATA_DIR = os.getenv("DATA_DIR", "data").rstrip("/")

OPS_CANDIDATES = [
    f"{DATA_DIR}/forecast_7d_ops_ready.parquet",
    f"{DATA_DIR}/forecast_7d_ops_ready.csv",
    "deploy/forecast_7d_ops_ready.parquet",
    "deploy/forecast_7d_ops_ready.csv",
    "data/forecast_7d_ops_ready.parquet",
    "data/forecast_7d_ops_ready.csv",
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

HOUR_RANGES = ["00-03", "03-06", "06-09", "09-12", "12-15", "15-18", "18-21", "21-24"]


# =============================================================================
# CSS (tooltip sabit, kompakt)
# =============================================================================
def _apply_tooltip_css():
    st.markdown(
        """
        <style>
          .deckgl-tooltip {
            max-width: 360px !important;
            padding: 10px 12px !important;
            line-height: 1.25 !important;
            border-radius: 12px !important;
            font-size: 12px !important;
          }
          .deckgl-tooltip {
            transform: translate(12px, 12px) !important;
          }
          /* Mobil/ dar ekranlarda taşma olmasın */
          @media (max-width: 768px) {
            .deckgl-tooltip { max-width: 300px !important; }
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
        if p and os.path.exists(p):
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
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def _fmt_expected(x):
    v = _safe_float(x)
    if not np.isfinite(v):
        return "—"
    lo, hi = int(np.floor(v)), int(np.ceil(v))
    return f"~{lo}" if lo == hi else f"~{lo}–{hi}"

def _fmt_harm(x):
    v = _safe_float(x)
    if not np.isfinite(v):
        return "—"
    # harm ölçeğin değişebilir: şimdilik 1 ondalık
    return f"{v:.1f}"

def _today_la():
    return datetime.now(ZoneInfo(TARGET_TZ)).date()

def _hour_range_now():
    h = datetime.now(ZoneInfo(TARGET_TZ)).hour
    # hangi slota denk geliyor?
    for r in HOUR_RANGES:
        a, b = r.split("-")
        if int(a) <= h < int(b):
            return r
    return "00-03"

def _compute_likert_quintiles(series: pd.Series):
    v = pd.to_numeric(series, errors="coerce")
    if v.notna().sum() < 10:
        return pd.Series([3] * len(v), index=v.index)
    try:
        return pd.qcut(v.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    except Exception:
        # tekil değer/bağ sorunu fallback
        return pd.Series([3] * len(v), index=v.index)


# =============================================================================
# LOADERS
# =============================================================================
def _local_load_parquet_or_csv(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Desteklenmeyen uzantı: {path}")

@st.cache_data(show_spinner=False)
def load_ops_ready() -> pd.DataFrame:
    p = _first_existing(OPS_CANDIDATES)
    if not p:
        return pd.DataFrame()

    try:
        if _load_parquet_or_csv is not None:
            df = _load_parquet_or_csv(p)
        else:
            df = _local_load_parquet_or_csv(p)
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # ---- normalize minimum columns
    geo_col = _pick_col(df, ["GEOID", "geoid"])
    date_col = _pick_col(df, ["date"])
    hr_col = _pick_col(df, ["hour_range", "hr_range", "hour"])

    if not geo_col or not date_col:
        return pd.DataFrame()

    df = df.copy()
    df["geoid"] = df[geo_col].map(_digits11)

    # date normalize -> date_only (string)
    dt = pd.to_datetime(df[date_col], errors="coerce")
    df["date_dt"] = dt
    df["date_only"] = dt.dt.date.astype(str)

    # hour_range normalize
    if hr_col and hr_col != "hour_range":
        df["hour_range"] = df[hr_col].astype(str)
    elif "hour_range" not in df.columns:
        df["hour_range"] = ""

    df["hour_range"] = df["hour_range"].astype(str).str.strip()

    # numeric cols normalize (varsa)
    for c in ["p_event", "risk_prob", "risk_score", "expected_crimes", "expected_count", "expected_harm"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

@st.cache_data(show_spinner=False)
def load_geojson():
    if os.path.exists(GEOJSON_PATH):
        with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# =============================================================================
# SAHA NOTU (1–2 cümle) — OPS_READY alanlarını kullanır
# =============================================================================
def saha_notu(r: pd.Series) -> str:
    notes = []

    # flags varsa kullan
    if bool(r.get("poi_flag", False)):
        notes.append("Riskli POI yoğunluğu.")
    if bool(r.get("calls_flag", False)):
        notes.append("Çağrı yoğunluğu.")
    if bool(r.get("neighbor_flag", False)):
        notes.append("Komşu baskısı.")
    if bool(r.get("time_flag", False)):
        notes.append("Zaman riski (gece/hafta sonu).")
    if bool(r.get("weather_flag", False)):
        notes.append("Hava etkisi olası.")

    return " ".join(notes[:2])


# =============================================================================
# GEOJSON ENRICH
# =============================================================================
def enrich_geojson(gj: dict, df_sel: pd.DataFrame) -> dict:
    if not gj or df_sel.empty:
        return gj

    d = df_sel.copy()

    # risk ve harm kolonları (ops_ready'de mevcut)
    risk_col = _pick_col(d, ["risk_score", "risk_prob", "p_event"])
    harm_col = _pick_col(d, ["expected_harm", "harm_expected"])
    exp_col = _pick_col(d, ["expected_crimes", "expected_count"])

    # likert
    d["risk_likert"] = _compute_likert_quintiles(d[risk_col]) if risk_col else 3
    d["harm_likert"] = _compute_likert_quintiles(d[harm_col]) if harm_col else 3

    # görsel alanlar
    d["fill_color"] = d["risk_likert"].map(lambda k: LIKERT[int(k)][1])
    d["expected_txt"] = d[exp_col].map(_fmt_expected) if exp_col else "—"
    d["harm_txt"] = d[harm_col].map(_fmt_harm) if harm_col else "—"
    d["harm_icon"] = d["harm_likert"].apply(lambda x: "⚠️" if int(x) >= 5 else "")
    d["saha_notu"] = d.apply(saha_notu, axis=1)

    # top3 kategoriler (ops_ready)
    for i in (1, 2, 3):
        c = f"top{i}_category"
        s = f"top{i}_share"
        if c not in d.columns:
            d[c] = ""
        if s not in d.columns:
            d[s] = np.nan

    # ops action (kısa)
    if "ops_actions_short" not in d.columns:
        d["ops_actions_short"] = ""

    # risk_level (varsa)
    if "risk_level" not in d.columns:
        # risk_bin -> label fallback
        if "risk_bin" in d.columns:
            mp = {1: "Very Low", 2: "Low", 3: "Medium", 4: "High", 5: "Critical"}
            d["risk_level"] = d["risk_bin"].map(mp).fillna("Unknown")
        else:
            d["risk_level"] = "Unknown"

    # aynı geoid için: seçilen filtre içinde "expected_harm" (yoksa risk_score) en yüksek olanı al
    rank_col = harm_col if harm_col else (risk_col if risk_col else None)
    if rank_col:
        d = d.sort_values(rank_col, ascending=False)
    d = d.drop_duplicates("geoid", keep="first").set_index("geoid")

    feats = []
    for f in gj.get("features", []):
        props = dict(f.get("properties") or {})
        gid = _digits11(props.get("geoid") or props.get("GEOID"))

        # defaults
        props.update({
            "display_id": gid,
            "fill_color": DEFAULT_FILL,
            "risk_level": "",
            "expected_txt": "—",
            "harm_txt": "—",
            "harm_icon": "",
            "saha_notu": "",
            "topcats": "",
            "ops_action": "",
        })

        if gid and gid in d.index:
            r = d.loc[gid]
            props["fill_color"] = r.get("fill_color", DEFAULT_FILL)
            props["risk_level"] = str(r.get("risk_level", ""))
            props["expected_txt"] = str(r.get("expected_txt", "—"))
            props["harm_txt"] = str(r.get("harm_txt", "—"))
            props["harm_icon"] = str(r.get("harm_icon", ""))
            props["saha_notu"] = str(r.get("saha_notu", ""))

            # top cats
            cats = []
            for i in (1, 2, 3):
                c = str(r.get(f"top{i}_category", "") or "").strip()
                sh = _safe_float(r.get(f"top{i}_share", np.nan))
                if c and c.lower() != "unknown":
                    if np.isfinite(sh) and sh > 0:
                        cats.append(f"{c} ({sh*100:.0f}%)")
                    else:
                        cats.append(c)
            props["topcats"] = " • ".join(cats[:3]) if cats else "—"

            props["ops_action"] = str(r.get("ops_actions_short", "") or "")

        feats.append({**f, "properties": props})

    return {**gj, "features": feats}


# =============================================================================
# MAP
# =============================================================================
def draw_map(gj: dict):
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
            "<div style='min-width:260px'>"
            "<b>GEOID:</b> {display_id}<br/>"
            "<b>Risk:</b> {risk_level}<br/>"
            "<b>Beklenen suç:</b> {expected_txt}<br/>"
            "<b>Beklenen zarar:</b> {harm_txt} {harm_icon}<br/>"
            "<b>Olası türler:</b> {topcats}<br/>"
            "<b>Saha notu:</b> {saha_notu}<br/>"
            "<b>Öneri:</b> {ops_action}"
            "</div>"
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
    st.caption("Operasyon paneli: seçilen tarih + saat aralığında en riskli bölgeleri gösterir.")

    df = load_ops_ready()
    if df.empty:
        st.error("OPS_READY verisi bulunamadı. (forecast_7d_ops_ready.parquet/csv)")
        st.caption("Aranan aday yollar: " + " | ".join(OPS_CANDIDATES))
        return

    # --- UI: Tarih + hour_range
    now_date = _today_la()
    now_hr = _hour_range_now()

    # Tarih seçim listesi (dosyadan)
    date_opts = sorted(pd.unique(df["date_only"].dropna()))
    if not date_opts:
        st.error("Veride date alanı okunamadı.")
        return

    # default date: bugün varsa bugün; yoksa en yakın (son)
    today_str = str(now_date)
    default_date = today_str if today_str in date_opts else date_opts[-1]

    c1, c2 = st.columns([2, 3])
    with c1:
        sel_date = st.selectbox("📅 Tarih", options=date_opts, index=date_opts.index(default_date))
    with c2:
        sel_ranges = st.multiselect(
            "⏰ Saat aralığı (hour_range)",
            options=HOUR_RANGES,
            default=[now_hr] if now_hr in HOUR_RANGES else [HOUR_RANGES[0]],
        )

    if not sel_ranges:
        st.warning("En az 1 saat aralığı seç.")
        return

    df_sel = df[(df["date_only"] == str(sel_date)) & (df["hour_range"].isin(sel_ranges))].copy()
    if df_sel.empty:
        st.warning("Bu tarih/saat seçimi için kayıt yok.")
        return

    # --- Sidebar: hızlı özet
    st.sidebar.markdown("### 📌 Özet")
    st.sidebar.write(f"**Tarih:** {sel_date}")
    st.sidebar.write(f"**Saat:** {', '.join(sel_ranges)}")
    st.sidebar.write(f"**Kayıt:** {len(df_sel):,}")

    # Kritik liste: expected_harm en yüksek Top-5
    harm_col = _pick_col(df_sel, ["expected_harm", "harm_expected"])
    rank_col = harm_col if harm_col else _pick_col(df_sel, ["risk_score", "risk_prob", "p_event"])
    if rank_col:
        top = df_sel.sort_values(rank_col, ascending=False).head(5)
        st.sidebar.markdown("### 🚨 En Kritik 5 Bölge")
        for _, r in top.iterrows():
            gid = r.get("geoid", "")
            hr = r.get("hour_range", "")
            eh = r.get(harm_col, np.nan) if harm_col else np.nan
            ex = r.get("expected_crimes", r.get("expected_count", np.nan))
            note = saha_notu(r) or (r.get("ops_actions_short", "") or "Görünür devriye önerilir.")
            st.sidebar.error(
                f"**{gid}** • {hr}\n\n"
                f"Beklenen suç: {_fmt_expected(ex)}\n\n"
                f"Zarar: {_fmt_harm(eh) if harm_col else '—'}\n\n"
                f"{note}"
            )

    # --- MAP
    gj = load_geojson()
    if not gj:
        st.error("GeoJSON bulunamadı: sf_cells.geojson")
        return

    gj_enriched = enrich_geojson(gj, df_sel)
    draw_map(gj_enriched)

    # --- ALT PANEL: Seçili GEOID -> kritik saatler
    st.divider()
    st.subheader("⏰ Seçili Bölge İçin Kritik Zaman")

    sel_geoid = st.text_input("GEOID (isteğe bağlı)", "")
    if sel_geoid:
        gid = _digits11(sel_geoid)
        g = df[(df["geoid"] == gid)].copy()
        if g.empty:
            st.warning("Bu GEOID için veri bulunamadı.")
            return

        g = g[g["date_only"] == str(sel_date)]
        if g.empty:
            st.warning("Bu GEOID için seçilen tarihte veri yok.")
            return

        if rank_col and rank_col in g.columns:
            score = pd.to_numeric(g[rank_col], errors="coerce")
            g["_score"] = score
            g = g.sort_values("_score", ascending=False)
            top_hr = g[["hour_range", "_score"]].dropna().head(3)

            if not top_hr.empty:
                hrs = top_hr["hour_range"].tolist()
                st.info(f"Bu bölgede en kritik saat aralıkları: **{', '.join(hrs)}**")

        # küçük tablo: ops metinleri
        show_cols = []
        for c in ["hour_range", "risk_level", "expected_crimes", "expected_harm", "top1_category", "top2_category", "top3_category", "ops_actions_short"]:
            if c in g.columns:
                show_cols.append(c)
        if show_cols:
            st.dataframe(g.sort_values(rank_col, ascending=False).head(12)[show_cols], use_container_width=True)


# Streamlit multipage çağırır (app.py içinden)
