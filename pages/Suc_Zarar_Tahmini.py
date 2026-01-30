# -*- coding: utf-8 -*-
# pages/Suc_Zarar_Tahmini.py
# SUTAM — Suç & Zarar (HARM) | Operasyon Paneli (Kolluk-dostu, kompakt)

from __future__ import annotations
import os, json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# ---------------- SAFE IMPORT ----------------
try:
    from src.io_data import load_parquet_or_csv, prepare_forecast
except Exception as e:
    load_parquet_or_csv = None
    prepare_forecast = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None

# ---------------- CONFIG ----------------
DATA_DIR = os.getenv("DATA_DIR", "data").rstrip("/")
TARGET_TZ = os.getenv("TARGET_TZ", "America/Los_Angeles")
GEOJSON_PATH = os.getenv("GEOJSON_PATH", "data/sf_cells.geojson")

OPS_CANDIDATES = [
    f"{DATA_DIR}/forecast_7d_ops_ready.parquet",
    f"{DATA_DIR}/forecast_7d_ops_ready.csv",
    "deploy/forecast_7d_ops_ready.parquet",
    "deploy/forecast_7d_ops_ready.csv",
    "data/forecast_7d_ops_ready.parquet",
    "data/forecast_7d_ops_ready.csv",
]

DEFAULT_FILL = [220, 220, 220]
LIKERT = {
    "Risk": {
        1: ("Çok Düşük",  [46, 204, 113]),
        2: ("Düşük",      [88, 214, 141]),
        3: ("Orta",       [241, 196, 15]),
        4: ("Yüksek",     [230, 126, 34]),
        5: ("Çok Yüksek", [192, 57, 43]),
    },
    "Zarar": {
        1: ("Düşük Etki",  [96, 165, 250]),
        2: ("Orta Etki",   [76, 147, 245]),
        3: ("Yüksek Etki", [241, 196, 15]),
        4: ("Çok Yüksek",  [230, 126, 34]),
        5: ("Kritik Etki", [192, 57, 43]),
    },
    "Ops Öncelik": {
        1: ("İzle",        [196, 226, 255]),
        2: ("Dikkat",      [148, 202, 255]),
        3: ("Öncelikli",   [241, 196, 15]),
        4: ("Çok Öncelik", [230, 126, 34]),
        5: ("Acil",        [192, 57, 43]),
    },
}

DRIVER_LABEL = {
    "risk_core": "Model riski",
    "calls": "Çağrı sinyali",
    "neighbor": "Komşu baskısı",
    "transit": "Transit etkisi",
    "poi": "Riskli POI",
    "weather": "Hava koşulu",
    "time": "Zaman deseni",
}

# ---------------- CSS ----------------
def apply_css():
    st.markdown(
        """
        <style>
          .block-container{padding-top:1rem;padding-bottom:2rem;}
          .sutam-card{border:1px solid rgba(148,163,184,.35);border-radius:16px;padding:14px;background:rgba(2,6,23,.25);box-shadow:0 14px 40px rgba(0,0,0,.12);}
          .sutam-card h3{margin:0 0 8px 0;font-size:14px;letter-spacing:.2px;}
          .sutam-kpi{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap;margin-top:6px;}
          .sutam-kpi .v{font-weight:900;font-size:20px;}
          .sutam-kpi .t{color:rgba(226,232,240,.9);font-size:12px;}
          .deckgl-tooltip{max-width:380px!important;max-height:360px!important;overflow:auto!important;padding:10px 12px!important;line-height:1.25!important;border-radius:12px!important;box-shadow:0 10px 30px rgba(0,0,0,.25)!important;transform:translate(12px,12px)!important;}
          .deckgl-tooltip hr{margin:8px 0!important;opacity:.25!important;}
          .badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:700;border:1px solid rgba(148,163,184,.35);margin-right:6px;}
          .stDataFrame{border-radius:12px;overflow:hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------------- SMALL UTILS ----------------
def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def digits11(x) -> str:
    s = "".join(ch for ch in str(x) if ch.isdigit())
    return s.zfill(11) if s else ""

def to_num(s, default=np.nan):
    return pd.to_numeric(s, errors="coerce").astype(float).fillna(default)

def fmt3(x):
    try:
        v = float(x)
        return "—" if not np.isfinite(v) else f"{v:.3f}"
    except Exception:
        return "—"

def fmt_expected(x):
    try:
        v = float(x)
        if not np.isfinite(v): return "—"
        v = max(0.0, v)
        lo, hi = int(np.floor(v)), int(np.ceil(v))
        return f"~{lo}" if lo == hi else f"~{lo}–{hi}"
    except Exception:
        return "—"

def fmt_harm(x):
    try:
        v = float(x)
        if not np.isfinite(v): return "—"
        if abs(v) >= 1000: return f"{v:,.0f}"
        if abs(v) >= 100:  return f"{v:.0f}"
        return f"{v:.1f}"
    except Exception:
        return "—"

def boolish(x) -> bool:
    if isinstance(x, (bool, np.bool_)): return bool(x)
    if x is None: return False
    return str(x).strip().lower() in ("1","true","t","yes","y")

def driver_label(x):
    return DRIVER_LABEL.get(str(x or "").strip(), str(x or "—"))

def flags_text(r) -> str:
    tags = []
    if boolish(r.get("calls_flag")): tags.append("☎️ Çağrı")
    if boolish(r.get("neighbor_flag")): tags.append("🧭 Komşu")
    if boolish(r.get("poi_flag")): tags.append("📍 POI")
    if boolish(r.get("transit_flag")): tags.append("🚇 Transit")
    if boolish(r.get("weather_flag")): tags.append("🌧️ Hava")
    if boolish(r.get("time_flag")): tags.append("🕒 Zaman")
    return " • ".join(tags) if tags else "—"

def action_fallback(r) -> str:
    a = str(r.get("ops_actions_short") or "").strip()
    if len(a) >= 3: return a
    a = str(r.get("ops_actions") or "").strip()
    if len(a) >= 3: return a
    lvl = str(r.get("risk_level") or "").lower()
    if "critical" in lvl or "çok yüksek" in lvl or "very high" in lvl:
        return "Acil görünürlük + hedefli müdahale"
    if "high" in lvl or "yüksek" in lvl:
        return "Hedefli devriye + giriş/çıkış kontrolü"
    if "medium" in lvl or "orta" in lvl:
        return "Kısa tur döngüsü + caydırıcılık"
    return "Rutin devriye + gözlemsel teyit"

def segmented(label, options, default):
    if hasattr(st, "segmented_control"):
        return st.segmented_control(label, options=options, default=default)
    idx = options.index(default) if default in options else 0
    return st.radio(label, options=options, index=idx, horizontal=True)

# ---------------- LOAD ----------------
@st.cache_data(show_spinner=False)
def load_geojson() -> dict:
    if os.path.exists(GEOJSON_PATH):
        with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data(show_spinner=False)
def load_ops_ready() -> pd.DataFrame:
    p = first_existing(OPS_CANDIDATES)
    if not p:
        return pd.DataFrame()

    if load_parquet_or_csv is not None:
        df = load_parquet_or_csv(p)
    else:
        df = pd.read_parquet(p) if p.lower().endswith(".parquet") else pd.read_csv(p)

    if df is None or getattr(df, "empty", True):
        return pd.DataFrame()

    if prepare_forecast is not None:
        try:
            df = prepare_forecast(df, gp=None)
        except Exception:
            pass
    return df

# ---------------- NORMALIZE (tek mapping ile) ----------------
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # kolon adı toleransı (tek yerde)
    MAP = {
        "geoid": ["geoid","GEOID"],
        "date": ["date","dt","datetime"],
        "hour_range": ["hour_range","hour_bucket"],
        "p_event": ["p_event","risk_prob","risk_score"],
        "expected_count": ["expected_count","expected_crimes"],
        "expected_harm": ["expected_harm","harm_expected","harm"],
        "ops_rank_score": ["ops_rank_score","risk_score"],
        "risk_level": ["risk_level"],
        "risk_bin": ["risk_bin"],
        "primary_driver": ["primary_driver"],
        "secondary_driver": ["secondary_driver"],
        "driver_profile": ["driver_profile"],
        "ops_actions_short": ["ops_actions_short"],
        "ops_actions": ["ops_actions"],
        "ops_reasons": ["ops_reasons"],
        "model_version": ["model_version"],
        "run_id": ["run_id"],
        "audit_tag": ["audit_tag"],
        "forecast_generated_at": ["forecast_generated_at"],
        "forecast_horizon_days": ["forecast_horizon_days"],
        "impact_flag": ["impact_flag"],
    }

    def col(name):
        cols = {c.lower(): c for c in d.columns}
        for cand in MAP.get(name, []):
            if cand.lower() in cols:
                return cols[cand.lower()]
        return None

    # required
    c_geoid = col("geoid")
    d["geoid"] = d[c_geoid].map(digits11) if c_geoid else ""

    c_date = col("date")
    _dt = pd.to_datetime(d[c_date], errors="coerce") if c_date else pd.NaT
    d["date_norm"] = pd.to_datetime(_dt).dt.normalize()

    c_hr = col("hour_range")
    d["hour_range"] = d[c_hr].astype(str) if c_hr else "00-24"

    for k in ("p_event","expected_count","expected_harm","ops_rank_score"):
        c = col(k)
        d[k] = pd.to_numeric(d[c], errors="coerce") if c else np.nan

    # risk_level
    c_rl = col("risk_level")
    if c_rl:
        d["risk_level"] = d[c_rl].astype(str)
    else:
        c_rb = col("risk_bin")
        mp = {1:"Very Low",2:"Low",3:"Medium",4:"High",5:"Critical"}
        d["risk_level"] = pd.to_numeric(d[c_rb], errors="coerce").map(mp).fillna("Unknown") if c_rb else "Unknown"

    # top categories (minimum)
    for i in (1,2,3):
        c = None
        cols = {x.lower(): x for x in d.columns}
        for cand in (f"top{i}_category", f"top{i}_cat"):
            if cand.lower() in cols:
                c = cols[cand.lower()]
                break
        d[f"top{i}_category"] = d[c].astype(str).replace("nan","").fillna("") if c else ""

    # text fields
    for k in ("primary_driver","secondary_driver","driver_profile","ops_actions_short","ops_actions","ops_reasons"):
        c = col(k)
        d[k] = d[c].astype(str).fillna("") if c else ""

    # flags (hepsi aynı pattern)
    for k in ("weather_flag","calls_flag","neighbor_flag","transit_flag","poi_flag","time_flag"):
        ck = k if k in d.columns else None
        d[k] = d[ck].apply(boolish) if ck else False

    # meta (opsiyonel)
    for k in ("model_version","run_id","audit_tag","forecast_generated_at","forecast_horizon_days","impact_flag"):
        c = col(k)
        d[k] = d[c] if c else ""

    return d

# ---------------- LIKERT + ENRICH ----------------
def quantile_likert(v: pd.Series) -> pd.Series:
    v = pd.to_numeric(v, errors="coerce")
    if v.notna().sum() < 10 or v.nunique(dropna=True) <= 1:
        return pd.Series([3]*len(v), index=v.index)
    try:
        return pd.qcut(v.rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    except Exception:
        return pd.Series([3]*len(v), index=v.index)

def enrich_geojson(gj: dict, df_slice: pd.DataFrame, mode: str) -> dict:
    if not gj or df_slice.empty:
        return gj

    # mode metric
    if mode == "Risk":
        src = "p_event"
    elif mode == "Zarar":
        src = "expected_harm"
    else:
        src = "ops_rank_score" if "ops_rank_score" in df_slice.columns else "expected_harm"

    pal = LIKERT[mode]
    d = df_slice.copy()
    d["_v"] = pd.to_numeric(d.get(src, np.nan), errors="coerce")
    d["_lik"] = quantile_likert(d["_v"]).clip(1,5)
    d["_tie"] = pd.to_numeric(d.get("expected_harm", np.nan), errors="coerce").fillna(-np.inf)

    # tek satır / geoid
    d = (d.sort_values(["_lik","_tie"], ascending=False)
           .drop_duplicates("geoid", keep="first")
           .set_index("geoid"))

    # hazırlık (tooltip alanları)
    d["_fill"] = d["_lik"].map(lambda k: pal[int(k)][1])
    d["_p"]    = d.get("p_event", np.nan).map(fmt3)
    d["_exp"]  = d.get("expected_count", np.nan).map(fmt_expected)
    d["_harm"] = d.get("expected_harm", np.nan).map(fmt_harm)

    def top3(r):
        arr=[]
        for i in (1,2,3):
            c=str(r.get(f"top{i}_category") or "").strip()
            if c and c.lower()!="unknown":
                arr.append(c)
        return " • ".join(arr) if arr else "—"

    d["_top3"]   = d.apply(top3, axis=1)
    d["_flags"]  = d.apply(flags_text, axis=1)
    d["_driver"] = d["primary_driver"].map(driver_label)
    d["_prof"]   = d["driver_profile"].replace("", "—")
    d["_act"]    = d.apply(action_fallback, axis=1)

    # geojson id extractor (tek fonksiyon)
    def feat_geoid(props: dict):
        for k in ("geoid","GEOID","cell_id","id","geoid11","geoid_11","display_id"):
            if k in props: return digits11(props[k]), props[k]
        for k,v in props.items():
            if "geoid" in str(k).lower():
                return digits11(v), v
        return "", ""

    feats=[]
    for feat in gj.get("features", []):
        props = dict(feat.get("properties") or {})
        key, raw = feat_geoid(props)

        props.update({
            "display_id": str(raw) if raw not in ("", None) else key,
            "mode_name": mode,
            "fill_color": DEFAULT_FILL,
            "risk_level": "",
            "p_event_txt": "—",
            "expected_txt": "—",
            "harm_txt": "—",
            "top3": "—",
            "driver_txt": "—",
            "profile_txt": "—",
            "flags_txt": "—",
            "ops_action": "—",
        })

        if key and key in d.index:
            r = d.loc[key]
            props["fill_color"]   = r.get("_fill", DEFAULT_FILL)
            props["risk_level"]   = str(r.get("risk_level","") or "")
            props["p_event_txt"]  = str(r.get("_p","—") or "—")
            props["expected_txt"] = str(r.get("_exp","—") or "—")
            props["harm_txt"]     = str(r.get("_harm","—") or "—")
            props["top3"]         = str(r.get("_top3","—") or "—")
            props["driver_txt"]   = str(r.get("_driver","—") or "—")
            props["profile_txt"]  = str(r.get("_prof","—") or "—")
            props["flags_txt"]    = str(r.get("_flags","—") or "—")
            props["ops_action"]   = str(r.get("_act","—") or "—")

        feats.append({**feat, "properties": props})

    return {**gj, "features": feats}

# ---------------- MAP ----------------
def draw_map(gj: dict):
    layer = pdk.Layer(
        "GeoJsonLayer",
        gj,
        stroked=True,
        get_line_color=[80, 80, 80],
        line_width_min_pixels=0.5,
        filled=True,
        get_fill_color="properties.fill_color",
        pickable=True,
        opacity=0.68,
    )

    tooltip = {
        "html": (
            "<div style='font-weight:900;font-size:14px;'>GEOID: {display_id}</div>"
            "<div style='opacity:.9;margin-top:2px;'><span class='badge'>{mode_name}</span> <b>{risk_level}</b></div>"
            "<hr/>"
            "<div><b>Olasılık:</b> {p_event_txt}</div>"
            "<div><b>Tahmini olay:</b> {expected_txt}</div>"
            "<div><b>Tahmini etki:</b> {harm_txt}</div>"
            "<hr/>"
            "<div><b>Hazırlıklı ol (Top3):</b> {top3}</div>"
            "<div><b>Ana neden:</b> {driver_txt} <span style='opacity:.75'>(profil: {profile_txt})</span></div>"
            "<div><b>Bağlam:</b> {flags_txt}</div>"
            "<hr/>"
            "<div style='font-weight:800;'>Ne yapmalı?</div>"
            "<div>{ops_action}</div>"
        ),
        "style": {"backgroundColor": "#0b1220", "color": "white"},
    }

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=37.7749, longitude=-122.4194, zoom=10),
        map_style="light",
        tooltip=tooltip,
    )
    st.pydeck_chart(deck, use_container_width=True)

# ---------------- TABLE + KPI ----------------
def kpis(df_slice: pd.DataFrame) -> dict:
    if df_slice.empty:
        return {"cells":0,"mean_p":np.nan,"sum_expected":0.0,"sum_harm":0.0,"hi_impact":0,"flags":{}}

    mean_p = to_num(df_slice["p_event"]).mean()
    sum_expected = to_num(df_slice["expected_count"], 0.0).sum()
    sum_harm = to_num(df_slice["expected_harm"], 0.0).sum()

    hi_impact = 0
    if "impact_flag" in df_slice.columns:
        hi_impact = int(df_slice["impact_flag"].astype(str).str.contains("High|Critical", case=False, regex=True).sum())

    flags = {k:int(df_slice[k].sum()) for k in ["calls_flag","neighbor_flag","poi_flag","transit_flag","weather_flag","time_flag"] if k in df_slice.columns}
    return {"cells":int(df_slice["geoid"].nunique()),"mean_p":float(mean_p),"sum_expected":float(sum_expected),"sum_harm":float(sum_harm),"hi_impact":hi_impact,"flags":flags}

def top_table(df_slice: pd.DataFrame, n: int, mode: str) -> pd.DataFrame:
    if df_slice.empty:
        return pd.DataFrame()

    rank_by = "p_event" if mode=="Risk" else ("expected_harm" if mode=="Zarar" else "ops_rank_score")
    d = df_slice.copy()
    d["_rk"] = pd.to_numeric(d.get(rank_by, np.nan), errors="coerce").fillna(-np.inf)

    def top3(r):
        arr=[]
        for i in (1,2,3):
            c=str(r.get(f"top{i}_category") or "").strip()
            if c and c.lower()!="unknown":
                arr.append(c)
        return ", ".join(arr[:3]) if arr else "—"

    out = (d.sort_values("_rk", ascending=False)
             .head(int(n))
             .assign(
                Top3=lambda x: x.apply(top3, axis=1),
                AnaNeden=lambda x: x["primary_driver"].map(driver_label),
                Olasılık=lambda x: x["p_event"].map(fmt3),
                TahminiOlay=lambda x: x["expected_count"].map(fmt_expected),
                TahminiEtki=lambda x: x["expected_harm"].map(fmt_harm),
             ))

    cols = ["geoid","risk_level","Olasılık","TahminiOlay","TahminiEtki","AnaNeden","driver_profile","ops_actions_short","Top3"]
    cols = [c for c in cols if c in out.columns]

    return out[cols].rename(columns={
        "geoid":"Bölge (GEOID)",
        "risk_level":"Alarm",
        "driver_profile":"Profil",
        "ops_actions_short":"Ne Yapmalı?",
        "Top3":"Hazırlıklı Ol (Top3)",
    })

# ---------------- GEOID (7 gün özet) ----------------
def geoid_week(df_all: pd.DataFrame, geoid: str) -> dict:
    g = df_all[df_all["geoid"] == geoid].copy()
    if g.empty: return {}

    g["_harm"] = pd.to_numeric(g.get("expected_harm", np.nan), errors="coerce").fillna(0.0)
    g["_ops"]  = pd.to_numeric(g.get("ops_rank_score", np.nan), errors="coerce").fillna(0.0)

    by_hr = (g.groupby("hour_range", dropna=False)[["_harm","_ops"]]
               .mean()
               .sort_values(["_harm","_ops"], ascending=False)
               .reset_index()
               .rename(columns={"hour_range":"Saat Dilimi","_harm":"Ort. Etki","_ops":"Ort. Ops"}))

    top_hours = by_hr.head(3)["Saat Dilimi"].astype(str).tolist()

    c = g.get("top1_category", pd.Series([], dtype=str)).astype(str).replace("nan","")
    c = c[c.str.len()>0]
    top_cats = c.value_counts().head(5).index.tolist()

    pdv = g.get("primary_driver", pd.Series([], dtype=str)).astype(str)
    pdv = pdv[pdv.str.len()>0]
    top_driver = pdv.value_counts().head(1).index.tolist()
    top_driver = top_driver[0] if top_driver else "risk_core"

    flags = {k:int(g[k].sum()) for k in ["calls_flag","neighbor_flag","poi_flag","transit_flag","weather_flag","time_flag"] if k in g.columns}

    summary = f"En kritik saatler: **{', '.join(top_hours) if top_hours else '—'}**. "
    if top_cats:
        summary += f"Öne çıkan suçlar: **{', '.join(top_cats[:3])}**. "
    summary += f"Ana etken eğilimi: **{driver_label(top_driver)}**."

    return {"summary":summary,"by_hr":by_hr,"top_cats":top_cats,"flags":flags}

# ---------------- MAIN ----------------
def render_suc_zarar_tahmini():
    apply_css()

    st.markdown("# 🧭 Suç & Zarar Etkisi — Operasyon Paneli")
    st.caption("Amaç: **Ne zaman, nerede, neye dikkat etmeli ve ne yapmalı?** (Karar desteğidir.)")

    if _IMPORT_ERR is not None:
        st.error("`src.io_data` import edilemedi.")
        st.code(repr(_IMPORT_ERR))
        return

    raw = load_ops_ready()
    if raw is None or raw.empty:
        st.error("Ops-ready veri bulunamadı/boş.\n" + "\n".join([f"- {p}" for p in OPS_CANDIDATES[:4]]))
        return

    df = normalize(raw)
    st.caption(f"✅ Ops-ready: {df.shape[0]:,} satır • GEOID: {df['geoid'].nunique():,}")

    gj = load_geojson()
    if not gj:
        st.error(f"GeoJSON bulunamadı: `{GEOJSON_PATH}`")
        return

    # --- state ---
    if "sz_mode" not in st.session_state: st.session_state.sz_mode = "Ops Öncelik"
    if "sz_topn" not in st.session_state: st.session_state.sz_topn = 15

    now_sf = datetime.now(ZoneInfo(TARGET_TZ))

    dates = sorted(df["date_norm"].dropna().unique())
    if not dates:
        st.error("Veride geçerli tarih yok.")
        return

    default_date = pd.Timestamp(now_sf.date())
    if default_date not in dates:
        past = [d for d in dates if d <= default_date]
        default_date = max(past) if past else dates[0]

    hrs = sorted(df["hour_range"].dropna().astype(str).unique().tolist())
    if not hrs:
        st.error("Veride hour_range yok/boş.")
        return

    c1, c2, c3, c4 = st.columns([1.25, 1.0, 1.2, 1.0])
    with c1:
        sel_date = st.selectbox("📅 Tarih", dates, index=dates.index(default_date), format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d"))
    with c2:
        sel_hr = st.selectbox("⏰ Saat dilimi", hrs, index=0)
    with c3:
        mode = segmented("🗺️ Harita modu", ["Risk","Zarar","Ops Öncelik"], st.session_state.sz_mode)
        st.session_state.sz_mode = mode
    with c4:
        topn = st.selectbox("📌 Top hücre", [10,15,20,30,50], index=[10,15,20,30,50].index(st.session_state.sz_topn))
        st.session_state.sz_topn = int(topn)

    df_slice = df[(df["date_norm"] == sel_date) & (df["hour_range"].astype(str) == str(sel_hr))].copy()
    if df_slice.empty:
        st.warning("Seçili tarih/saat dilimi için kayıt yok.")
        return

    # KPI
    k = kpis(df_slice)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"<div class='sutam-card'><h3>Hücre</h3><div class='sutam-kpi'><div class='v'>{k['cells']}</div><div class='t'>izlenecek bölge</div></div></div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div class='sutam-card'><h3>Ortalama Olasılık</h3><div class='sutam-kpi'><div class='v'>{fmt3(k['mean_p'])}</div><div class='t'>şehir geneli</div></div></div>", unsafe_allow_html=True)
    with k3:
        st.markdown(f"<div class='sutam-card'><h3>Tahmini Olay</h3><div class='sutam-kpi'><div class='v'>{int(round(k['sum_expected']))}</div><div class='t'>bu dilimde</div></div></div>", unsafe_allow_html=True)
    with k4:
        st.markdown(f"<div class='sutam-card'><h3>Tahmini Etki</h3><div class='sutam-kpi'><div class='v'>{fmt_harm(k['sum_harm'])}</div><div class='t'>kritik: {k['hi_impact']}</div></div></div>", unsafe_allow_html=True)

    # Sidebar (çok kısa)
    st.sidebar.markdown("### 🎯 Vardiya Özeti")
    st.sidebar.caption(f"SF: **{now_sf:%Y-%m-%d %H:%M}**")
    st.sidebar.write(f"**Tarih:** {pd.Timestamp(sel_date).strftime('%Y-%m-%d')}")
    st.sidebar.write(f"**Dilim:** {sel_hr}")
    st.sidebar.write(f"**Mod:** {mode}")
    if k.get("flags"):
        f = k["flags"]
        st.sidebar.markdown("### 🚩 Bağlam")
        st.sidebar.write(f"☎️ {f.get('calls_flag',0)} • 🧭 {f.get('neighbor_flag',0)} • 📍 {f.get('poi_flag',0)} • 🚇 {f.get('transit_flag',0)} • 🌧️ {f.get('weather_flag',0)} • 🕒 {f.get('time_flag',0)}")

    st.divider()

    # Map
    gj2 = enrich_geojson(gj, df_slice, mode)
    draw_map(gj2)
    st.caption("Hover ile detay. Harita modu: Risk / Zarar / Ops Öncelik.")

    st.divider()

    # Table
    st.subheader("📌 Operasyon Öncelik Listesi")
    tbl = top_table(df_slice, n=topn, mode=mode)
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    cexp1, cexp2 = st.columns(2)
    with cexp1:
        tmp = df_slice.copy()
        tmp["flags_txt"] = tmp.apply(flags_text, axis=1)
        tmp["primary_driver_label"] = tmp["primary_driver"].map(driver_label)
        st.download_button("⬇️ Dilim CSV", data=tmp.to_csv(index=False).encode("utf-8-sig"), file_name="ops_slice.csv", mime="text/csv")
    with cexp2:
        st.download_button("⬇️ Top liste CSV", data=tbl.to_csv(index=False).encode("utf-8-sig"), file_name="ops_top_list.csv", mime="text/csv")

    st.divider()

    # GEOID
    st.subheader("🔎 Bölge Detayı — 7 Günlük Özet")
    left, right = st.columns([1,2])
    with left:
        graw = st.text_input("GEOID (11 hane)", value="", placeholder="06075030101")
        geoid = digits11(graw) if graw else ""
        st.caption("GEOID gir → kritik saatler + tek cümle öneri")
    with right:
        if not geoid:
            st.info("Bölge kodunu girince özet görünecek.")
        else:
            prof = geoid_week(df, geoid)
            if not prof:
                st.warning("Bu GEOID için kayıt yok.")
            else:
                st.markdown(prof["summary"])

                dhr = prof["by_hr"].copy()
                dhr["Ort. Etki"] = dhr["Ort. Etki"].map(fmt_harm)
                dhr["Ort. Ops"] = dhr["Ort. Ops"].map(fmt3)
                st.dataframe(dhr.head(10), use_container_width=True, hide_index=True)

                if prof["top_cats"]:
                    st.markdown("**Öne çıkan suçlar:** " + " • ".join([f"`{c}`" for c in prof["top_cats"][:5]]))

                if prof["flags"]:
                    f = prof["flags"]
                    st.markdown(f"**Bağlam (7 gün):** ☎️ {f.get('calls_flag',0)} • 🧭 {f.get('neighbor_flag',0)} • 📍 {f.get('poi_flag',0)} • 🚇 {f.get('transit_flag',0)} • 🌧️ {f.get('weather_flag',0)} • 🕒 {f.get('time_flag',0)}")

                cur = df_slice[df_slice["geoid"] == geoid].copy()
                if not cur.empty:
                    r = cur.sort_values(["expected_harm","ops_rank_score"], ascending=False).head(1).iloc[0]
                    st.markdown("### 🧾 Bu dilim için tek cümle öneri")
                    st.success(action_fallback(r))
                    why = str(r.get("ops_reasons") or "").strip()
                    if not why:
                        why = f"{driver_label(r.get('primary_driver'))} • {flags_text(r)}"
                    st.markdown("### 🧠 Neden")
                    st.write(why)

    st.divider()

    # Footer (dipnot)
    with st.expander("ℹ️ Model bilgisi", expanded=False):
        def first_val(col):
            if col not in df_slice.columns: return "—"
            s = df_slice[col]
            s = s.dropna()
            return str(s.iloc[0]) if len(s) else "—"

        st.write(
            f"**model_version:** {first_val('model_version')} • "
            f"**run_id:** {first_val('run_id')} • "
            f"**horizon_days:** {first_val('forecast_horizon_days')} • "
            f"**generated_at:** {first_val('forecast_generated_at')}"
        )
        tag = first_val("audit_tag")
        if tag != "—":
            st.caption(f"audit_tag: {tag}")
