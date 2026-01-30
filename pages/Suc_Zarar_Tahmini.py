# -*- coding: utf-8 -*-
# pages/Suc_Zarar_Tahmini.py
# SUTAM — Suç + Zarar (HARM) Tahmini | Operasyonel Karar Destek (Kolluk-Dostu)
#
# REVİZE HEDEF:
# - TABLO YOK
# - Top-N kart listesi YOK (kalabalık yapıyor)
# - Harita: SADECE bağlam (hover yok)
# - Brifing: Kolluk GEOID seçince tek panelde gelsin
# - (Opsiyon) Folium açılırsa haritadan tıklayıp GEOID seçimi yapılabilir

from __future__ import annotations

import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# ---------------------------------------------------------------------
# SAFE IMPORT (src yoksa sayfa çökmesin)
# ---------------------------------------------------------------------
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
TARGET_TZ = os.getenv("TARGET_TZ", "America/Los_Angeles")
GEOJSON_PATH = os.getenv("GEOJSON_PATH", "data/sf_cells.geojson")

OPS_CANDIDATES = [
    "/mnt/data/forecast_7d_ops_harm_ready.csv",
    f"{DATA_DIR}/forecast_7d_ops_harm_ready.csv",
    f"{DATA_DIR}/forecast_7d_ops_ready.parquet",
    f"{DATA_DIR}/forecast_7d_ops_ready.csv",
    "deploy/forecast_7d_ops_ready.parquet",
    "deploy/forecast_7d_ops_ready.csv",
    "data/forecast_7d_ops_ready.parquet",
    "data/forecast_7d_ops_ready.csv",
]

DEFAULT_FILL = [220, 220, 220]

# =============================================================================
# CSS (compact)
# =============================================================================
def segmented_compat(label: str, options: list[str], default: str):
    if hasattr(st, "segmented_control"):
        return st.segmented_control(label, options=options, default=default)
    idx = options.index(default) if default in options else 0
    return st.radio(label, options=options, index=idx, horizontal=True)

def _apply_global_css():
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1180px; }

          .sutam-card {
            border: 1px solid rgba(148,163,184,.35);
            border-radius: 16px;
            padding: 14px 14px;
            background: rgba(2,6,23,.25);
            box-shadow: 0 14px 40px rgba(0,0,0,.12);
            margin-bottom: 10px;
          }

          .sutam-card h3 { margin: 0 0 8px 0; font-size: 14px; letter-spacing:.2px; }

          .sutam-kpi {
            display:flex; gap:12px; align-items:baseline; flex-wrap:wrap;
            margin-top:6px;
          }
          .sutam-kpi .v { font-weight:900; font-size: 20px; }
          .sutam-kpi .t { color: rgba(226,232,240,.9); font-size: 12px; }

          .badge {
            display:inline-block;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 800;
            border: 1px solid rgba(148,163,184,.35);
            margin-right: 6px;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# UTIL
# =============================================================================
def _first_existing(paths: list[str]) -> str | None:
    for p in paths:
        try:
            if os.path.exists(p):
                return p
        except Exception:
            pass
    return None

def _digits11(x) -> str:
    s = "".join(ch for ch in str(x) if ch.isdigit())
    return s.zfill(11) if s else ""

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k.lower() in cols:
            return cols[k.lower()]
    return None

def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _coerce_bool(x) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in ("1", "true", "t", "yes", "y")

def _parse_range(tok: str):
    if not isinstance(tok, str) or "-" not in tok:
        return None
    a, b = tok.split("-", 1)
    try:
        s = int(a.strip()); e = int(b.strip())
    except Exception:
        return None
    s = max(0, min(23, s))
    e = max(1, min(24, e))
    return (s, e)

def _hour_to_bucket(h: int, labels: list[str]) -> str | None:
    parsed = []
    for lab in labels:
        rg = _parse_range(str(lab))
        if rg:
            parsed.append((str(lab), rg[0], rg[1]))
    for lab, s, e in parsed:
        if s <= h < e:
            return lab
    for lab, s, e in parsed:
        if s > e and (h >= s or h < e):
            return lab
    return parsed[0][0] if parsed else None

def _dominant_flag_badges(r: pd.Series) -> str:
    badges = []
    if _coerce_bool(r.get("calls_flag")):
        badges.append("☎️ Çağrı")
    if _coerce_bool(r.get("neighbor_flag")):
        badges.append("🧭 Komşu")
    if _coerce_bool(r.get("poi_flag")):
        badges.append("📍 POI")
    if _coerce_bool(r.get("transit_flag")):
        badges.append("🚇 Transit")
    if _coerce_bool(r.get("weather_flag")):
        badges.append("🌧️ Hava")
    if _coerce_bool(r.get("time_flag")):
        badges.append("🕒 Zaman")
    return " • ".join(badges) if badges else "—"

def _driver_label(x: str) -> str:
    m = {
        "risk_core": "Model riski",
        "calls": "Çağrı sinyali",
        "neighbor": "Komşu baskısı",
        "transit": "Transit etkisi",
        "poi": "Riskli POI",
        "weather": "Hava koşulu",
        "time": "Zaman deseni",
    }
    return m.get(str(x or "").strip(), str(x or "—"))

def _risk_text_hint(level: str) -> str:
    level = str(level or "").strip().lower()
    if "critical" in level or "çok yüksek" in level or "very high" in level:
        return "P1/P2 hızlı intikal + görünür devriye (sıcak nokta)."
    if "high" in level or "yüksek" in level:
        return "Hedefli devriye + giriş/çıkış kontrolü + kısa tur döngüsü."
    if "medium" in level or "orta" in level:
        return "Kısa tur döngüsü + caydırıcı görünürlük + gözlemsel teyit."
    return "Rutin devriye + gözlemsel teyit."

# =============================================================================
# KOLLUK DİLİ: seviyelendirme
# =============================================================================
def _qband(value: float, qs: tuple[float, float], labels=("Düşük", "Orta", "Yoğun")) -> str:
    v = _safe_float(value, np.nan)
    if not np.isfinite(v):
        return "—"
    q1, q2 = qs
    if not (np.isfinite(q1) and np.isfinite(q2)):
        if v <= 0:
            return labels[0]
        if v <= 2:
            return labels[1]
        return labels[2]
    if v <= q1:
        return labels[0]
    if v <= q2:
        return labels[1]
    return labels[2]

def _make_thresholds(d: pd.DataFrame) -> dict:
    out = {}
    def qpair(col):
        s = pd.to_numeric(d.get(col, np.nan), errors="coerce").dropna()
        if len(s) < 20:
            return (np.nan, np.nan)
        return (float(s.quantile(0.33)), float(s.quantile(0.66)))

    for c in [
        "bus_stop_count","train_stop_count","poi_total_count","poi_risk_score",
        "911_request_count_hour_range","911_request_count_daily(before_24_hours)",
        "neighbor_crime_7d","distance_to_bus","distance_to_train"
    ]:
        out[c] = qpair(c)
    return out

def _weather_text(r: pd.Series) -> str:
    prcp = _safe_float(r.get("prcp"), np.nan)
    if not np.isfinite(prcp):
        return "Hava: Bilinmiyor"
    if prcp <= 0.0:
        return "Hava: Yağış beklenmiyor (kuru)"
    if prcp < 1.0:
        return "Hava: Hafif yağış ihtimali"
    if prcp < 5.0:
        return "Hava: Yağmurlu"
    return "Hava: Şiddetli yağış"

def _time_text(r: pd.Series) -> str:
    wk = _coerce_bool(r.get("is_weekend"))
    ng = _coerce_bool(r.get("is_night"))
    if wk and ng:
        return "Zaman: Hafta sonu gece (hareketlilik artar)"
    if wk:
        return "Zaman: Hafta sonu (kalabalık/etkinlik etkisi)"
    if ng:
        return "Zaman: Gece saatleri (görünür devriye önemli)"
    return "Zaman: Gündüz/hafta içi (rutin akış)"

def _transit_text(r: pd.Series, th: dict) -> str:
    b = r.get("bus_stop_count", np.nan)
    t = r.get("train_stop_count", np.nan)
    b_lvl = _qband(b, th.get("bus_stop_count",(np.nan,np.nan)), labels=("Düşük", "Orta", "Yoğun"))
    t_lvl = _qband(t, th.get("train_stop_count",(np.nan,np.nan)), labels=("Düşük", "Orta", "Yoğun"))

    db = _safe_float(r.get("distance_to_bus"), np.nan)
    dt = _safe_float(r.get("distance_to_train"), np.nan)

    def near_level(dist, key):
        q1, q2 = th.get(key, (np.nan, np.nan))
        if not np.isfinite(_safe_float(dist)):
            return "—"
        if not (np.isfinite(q1) and np.isfinite(q2)):
            if dist <= 200: return "Yakın"
            if dist <= 600: return "Orta"
            return "Uzak"
        if dist <= q1: return "Yakın"
        if dist <= q2: return "Orta"
        return "Uzak"

    db_lvl = near_level(db, "distance_to_bus")
    dt_lvl = near_level(dt, "distance_to_train")

    return f"Transit: Durak yoğunluğu (Otobüs:{b_lvl} / Tren:{t_lvl}) • Yakınlık (Otobüs:{db_lvl} / Tren:{dt_lvl})"

def _poi_text(r: pd.Series, th: dict) -> str:
    cnt = r.get("poi_total_count", np.nan)
    risk = r.get("poi_risk_score", np.nan)
    cnt_lvl = _qband(cnt, th.get("poi_total_count",(np.nan,np.nan)), labels=("Az", "Orta", "Çok"))
    risk_lvl = _qband(risk, th.get("poi_risk_score",(np.nan,np.nan)), labels=("Düşük", "Orta", "Yüksek"))

    if cnt_lvl == "Çok" and risk_lvl in ("Orta","Yüksek"):
        return "Çevre: POI yoğun • kalabalık/işletme kaynaklı risk artışı olası"
    if risk_lvl == "Yüksek":
        return "Çevre: Riskli POI baskın (kalabalık odak olasılığı)"
    if cnt_lvl == "Çok":
        return "Çevre: POI yoğun (yaya trafiği/kalabalık olasılığı)"
    if cnt_lvl == "Az" and risk_lvl == "Düşük":
        return "Çevre: POI etkisi düşük"
    return "Çevre: POI etkisi orta"

def _calls_text(r: pd.Series, th: dict) -> str:
    h = r.get("911_request_count_hour_range", np.nan)
    d1 = r.get("911_request_count_daily(before_24_hours)", np.nan)
    hl = _qband(h, th.get("911_request_count_hour_range",(np.nan,np.nan)), labels=("Düşük", "Orta", "Yüksek"))
    dl = _qband(d1, th.get("911_request_count_daily(before_24_hours)",(np.nan,np.nan)), labels=("Düşük", "Orta", "Yüksek"))
    return f"Çağrı: Bu dilimde {hl} • Son 24s {dl}"

def _neighbor_text(r: pd.Series, th: dict) -> str:
    n = r.get("neighbor_crime_7d", np.nan)
    lvl = _qband(n, th.get("neighbor_crime_7d",(np.nan,np.nan)), labels=("Düşük", "Orta", "Yüksek"))
    return f"Komşu baskısı: {lvl} (son 7g çevre eğilimi)"

# =============================================================================
# LOADERS
# =============================================================================
@st.cache_data(show_spinner=False)
def load_geojson() -> dict:
    if os.path.exists(GEOJSON_PATH):
        with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data(show_spinner=False)
def load_ops_ready() -> pd.DataFrame:
    p = _first_existing(OPS_CANDIDATES)
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

# =============================================================================
# NORMALIZE
# =============================================================================
def normalize_ops(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    geoid_col = _pick_col(d, ["GEOID", "geoid"])
    d["geoid"] = d[geoid_col].map(_digits11) if geoid_col else ""

    date_col = _pick_col(d, ["date", "dt", "datetime"])
    d["_dt"] = pd.to_datetime(d[date_col], errors="coerce") if date_col else pd.NaT
    d["date_norm"] = d["_dt"].dt.normalize()

    hr_col = _pick_col(d, ["hour_range", "hour_bucket"])
    d["hour_range"] = d[hr_col].astype(str) if hr_col else "00-24"

    p_col = _pick_col(d, ["p_event", "risk_prob", "risk_score"])
    d["p_event"] = pd.to_numeric(d[p_col], errors="coerce") if p_col else np.nan

    ex_col = _pick_col(d, ["expected_count", "expected_crimes"])
    d["expected_count"] = pd.to_numeric(d[ex_col], errors="coerce") if ex_col else np.nan

    harm_col = _pick_col(d, ["expected_harm", "harm_expected", "harm"])
    d["expected_harm"] = pd.to_numeric(d[harm_col], errors="coerce") if harm_col else np.nan

    ops_score_col = _pick_col(d, ["ops_rank_score"])
    d["ops_rank_score"] = pd.to_numeric(d[ops_score_col], errors="coerce") if ops_score_col else d["expected_harm"]

    rl_col = _pick_col(d, ["risk_level"])
    if rl_col:
        d["risk_level"] = d[rl_col].astype(str)
    else:
        rb_col = _pick_col(d, ["risk_bin"])
        if rb_col:
            mp = {1: "Very Low", 2: "Low", 3: "Medium", 4: "High", 5: "Critical"}
            d["risk_level"] = pd.to_numeric(d[rb_col], errors="coerce").map(mp).fillna("Unknown")
        else:
            d["risk_level"] = "Unknown"

    for i in (1, 2, 3):
        c = _pick_col(d, [f"top{i}_category", f"top{i}_cat"])
        d[f"top{i}_category"] = d[c].astype(str).replace("nan", "").fillna("") if c else ""

    for k in ["primary_driver", "secondary_driver", "driver_profile"]:
        col = _pick_col(d, [k])
        d[k] = d[col].astype(str) if col else ""

    for k in ["weather_flag", "calls_flag", "neighbor_flag", "transit_flag", "poi_flag", "time_flag"]:
        col = _pick_col(d, [k])
        d[k] = d[col].apply(_coerce_bool) if col else False

    for k in ["ops_actions_short", "ops_actions", "ops_reasons", "ops_actions_long", "ops_reasons_long"]:
        col = _pick_col(d, [k])
        d[k] = d[col].astype(str).fillna("") if col else ""

    for k in ["model_version", "run_id", "audit_tag", "forecast_generated_at", "forecast_horizon_days"]:
        col = _pick_col(d, [k])
        d[k] = d[col] if col else ""

    opt_numeric = [
        "prcp",
        "bus_stop_count","train_stop_count","distance_to_bus","distance_to_train",
        "poi_total_count","poi_risk_score",
        "911_request_count_hour_range","911_request_count_daily(before_24_hours)",
        "neighbor_crime_7d",
        "is_night","is_weekend",
    ]
    for k in opt_numeric:
        col = _pick_col(d, [k])
        if col:
            d[k] = pd.to_numeric(d[col], errors="coerce")

    return d

# =============================================================================
# LIKERT / COLOR
# =============================================================================
def _quantile_likert(series: pd.Series, n=5, neutral=3) -> pd.Series:
    v = pd.to_numeric(series, errors="coerce")
    if v.notna().sum() < 10 or v.nunique(dropna=True) <= 1:
        return pd.Series([neutral] * len(v), index=v.index)
    try:
        return pd.qcut(v.rank(method="first"), n, labels=list(range(1, n + 1))).astype(int)
    except Exception:
        return pd.Series([neutral] * len(v), index=v.index)

LIKERT_OPS = {
    1: ("İzle",        [196, 226, 255]),
    2: ("Dikkat",      [148, 202, 255]),
    3: ("Öncelikli",   [241, 196, 15]),
    4: ("Çok Öncelik", [230, 126, 34]),
    5: ("Acil",        [192, 57, 43]),
}
LIKERT_RISK = {
    1: ("Çok Düşük",  [46, 204, 113]),
    2: ("Düşük",      [88, 214, 141]),
    3: ("Orta",       [241, 196, 15]),
    4: ("Yüksek",     [230, 126, 34]),
    5: ("Çok Yüksek", [192, 57, 43]),
}
LIKERT_HARM = {
    1: ("Düşük Etki",  [96, 165, 250]),
    2: ("Orta Etki",   [76, 147, 245]),
    3: ("Yüksek Etki", [241, 196, 15]),
    4: ("Çok Yüksek",  [230, 126, 34]),
    5: ("Kritik Etki", [192, 57, 43]),
}

def compute_mode_likert(df_slice: pd.DataFrame, mode: str):
    if df_slice.empty:
        return pd.Series([], dtype=int), {"palette": LIKERT_OPS, "source_col": None}

    if mode == "Zarar":
        src = "expected_harm"
        palette = LIKERT_HARM
    elif mode == "Ops Öncelik":
        src = "ops_rank_score" if "ops_rank_score" in df_slice.columns else "expected_harm"
        palette = LIKERT_OPS
    else:
        src = "p_event"
        palette = LIKERT_RISK

    v = pd.to_numeric(df_slice.get(src, np.nan), errors="coerce")
    lik = _quantile_likert(v, n=5, neutral=3)
    return lik, {"palette": palette, "source_col": src}

# =============================================================================
# GEOJSON ENRICH (sadece renk; hover yok)
# =============================================================================
def enrich_geojson_fill_only(gj: dict, df_slice: pd.DataFrame, mode: str) -> dict:
    if not gj or df_slice.empty:
        return gj

    d = df_slice.copy()
    lik, meta = compute_mode_likert(d, mode)
    d["_lik"] = lik.clip(1, 5)
    palette = meta.get("palette") or LIKERT_OPS
    d["_fill"] = d["_lik"].map(lambda k: palette[int(k)][1])

    d["_tie"] = pd.to_numeric(d.get("ops_rank_score", d.get("expected_harm", np.nan)), errors="coerce").fillna(-np.inf)
    d = (
        d.sort_values(["_lik", "_tie"], ascending=[False, False])
          .drop_duplicates("geoid", keep="first")
          .set_index("geoid")
    )

    out_feats = []
    for feat in gj.get("features", []):
        props = dict(feat.get("properties") or {})
        raw = None
        for k in ("geoid", "GEOID", "cell_id", "id", "geoid11", "geoid_11", "display_id"):
            if k in props:
                raw = props[k]
                break
        if raw is None:
            for k, v in props.items():
                if "geoid" in str(k).lower():
                    raw = v
                    break

        key = _digits11(raw)
        props["display_id"] = str(raw) if raw not in (None, "") else key
        props["fill_color"] = DEFAULT_FILL

        if key and key in d.index:
            r = d.loc[key]
            props["fill_color"] = r.get("_fill", DEFAULT_FILL)

        out_feats.append({**feat, "properties": props})

    return {**gj, "features": out_feats}

# =============================================================================
# MAP RENDER (hover kapalı)
# =============================================================================
def draw_map_no_hover(gj: dict):
    layer = pdk.Layer(
        "GeoJsonLayer",
        gj,
        stroked=True,
        get_line_color=[80, 80, 80],
        line_width_min_pixels=0.5,
        filled=True,
        get_fill_color="properties.fill_color",
        pickable=False,  # hover/click yok
        opacity=0.68,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=37.7749, longitude=-122.4194, zoom=10),
        map_style="light",
        tooltip=None,  # hover kapalı
    )
    st.pydeck_chart(deck, use_container_width=True)

# =============================================================================
# SLICE
# =============================================================================
def _slice_by_date_hour(df: pd.DataFrame, sel_date: pd.Timestamp, hr_label: str) -> pd.DataFrame:
    return df[(df["date_norm"] == sel_date) & (df["hour_range"].astype(str) == str(hr_label))].copy()

# =============================================================================
# KPI (az sayı)
# =============================================================================
def _kpi_text_levels(df_slice: pd.DataFrame) -> dict:
    def lvl(col, labels):
        s = pd.to_numeric(df_slice.get(col, np.nan), errors="coerce")
        if s.notna().sum() < 20:
            return "—"
        q1 = float(s.quantile(0.33))
        q2 = float(s.quantile(0.66))
        v = float(s.mean())
        return _qband(v, (q1, q2), labels=labels)

    return {
        "cells": int(df_slice["geoid"].nunique()) if "geoid" in df_slice.columns else 0,
        "risk": lvl("p_event", labels=("Daha sakin", "Dikkat", "Yüksek risk")),
        "event": lvl("expected_count", labels=("Az olay", "Orta", "Yoğun olay")),
        "harm": lvl("expected_harm", labels=("Düşük etki", "Orta etki", "Yüksek etki")),
    }

# =============================================================================
# GEOID BRIEF (tek panel)
# =============================================================================
def build_geoid_brief(df_all: pd.DataFrame, df_slice: pd.DataFrame, geoid: str, sel_hr: str) -> dict:
    geoid = _digits11(geoid)
    if not geoid:
        return {}

    # 7g profil
    g = df_all[df_all["geoid"] == geoid].copy()
    if g.empty:
        return {}

    g["_harm"] = pd.to_numeric(g.get("expected_harm", np.nan), errors="coerce").fillna(0.0)
    by_hr = g.groupby("hour_range", dropna=False)["_harm"].mean().sort_values(ascending=False)
    top_hours = by_hr.head(3).index.astype(str).tolist()

    c = g.get("top1_category", pd.Series([], dtype=str)).astype(str).replace("nan", "")
    c = c[(c.str.len() > 0) & (c.str.lower() != "unknown")]
    top_cats = c.value_counts().head(5).index.tolist()

    pdv = g.get("primary_driver", pd.Series([], dtype=str)).astype(str)
    pdv = pdv[pdv.str.len() > 0]
    top_driver = pdv.value_counts().head(1).index.tolist()
    top_driver = top_driver[0] if top_driver else "risk_core"

    flags7 = {}
    for k in ["calls_flag","neighbor_flag","poi_flag","transit_flag","weather_flag","time_flag"]:
        if k in g.columns:
            flags7[k] = int(pd.to_numeric(g[k], errors="coerce").fillna(0).sum())

    # bu dilimde (seçili saat) tek kayıt
    cur = df_slice[df_slice["geoid"] == geoid].copy()
    cur_row = None
    if not cur.empty:
        cur = cur.sort_values(["ops_rank_score"], ascending=False)
        cur_row = cur.iloc[0]

    return {
        "geoid": geoid,
        "top_hours": top_hours,
        "top_cats": top_cats,
        "top_driver": top_driver,
        "flags7": flags7,
        "cur_row": cur_row,
        "sel_hr": sel_hr,
    }

def render_brief_panel(info: dict, df_slice: pd.DataFrame):
    geoid = info["geoid"]
    th = _make_thresholds(df_slice)

    st.markdown(
        f"""
        <div class="sutam-card">
          <div style="font-weight:900;font-size:16px;">📍 GEOID {geoid} — Brifing</div>
          <div style="opacity:.85;margin-top:4px;">Seçilen dilim için tek öneri + 7 günlük kısa özet.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 7 günlük özet
    s = []
    if info.get("top_hours"):
        s.append(f"⏱️ En kritik dilimler: **{' / '.join(info['top_hours'])}**")
    if info.get("top_cats"):
        s.append(f"🎯 Öne çıkan odak: **{' • '.join(info['top_cats'][:3])}**")
    s.append(f"🧠 Baskın neden: **{_driver_label(info.get('top_driver'))}**")
    st.markdown("### 📌 7 Günlük Özet")
    st.write("\n\n".join(s))

    # bağlam sinyali
    flags7 = info.get("flags7", {}) or {}
    if flags7:
        st.markdown(
            f"**Bağlam sinyali (7 gün):** "
            f"☎️ {flags7.get('calls_flag',0)} • 🧭 {flags7.get('neighbor_flag',0)} • 📍 {flags7.get('poi_flag',0)} • "
            f"🚇 {flags7.get('transit_flag',0)} • 🌧️ {flags7.get('weather_flag',0)} • 🕒 {flags7.get('time_flag',0)}"
        )

    # seçili dilim önerisi
    st.markdown("### ✅ Bu dilim için öneri")
    cur = info.get("cur_row", None)
    if cur is None:
        st.info("Bu GEOID seçili dilimde görünmüyor (başka saat/tarih dene).")
        return

    risk = str(cur.get("risk_level","—") or "—")
    driver = _driver_label(cur.get("primary_driver"))
    prof = str(cur.get("driver_profile") or "—").strip() or "—"
    flags_now = _dominant_flag_badges(cur)

    action = str(cur.get("ops_actions_short") or "").strip()
    if len(action) < 3:
        action = str(cur.get("ops_actions") or "").strip()
    if len(action) < 3:
        action = _risk_text_hint(risk)

    st.success(action)

    st.markdown("### 🧠 Neden / Bağlam (kolluk dili)")
    why_lines = [
        f"Risk seviyesi: {risk}",
        f"Ana neden: {driver} (profil: {prof})",
        f"Bağlam: {flags_now} • Dilim: {info.get('sel_hr','—')}",
        _weather_text(cur),
        _time_text(cur),
        _calls_text(cur, th),
        _neighbor_text(cur, th),
        _transit_text(cur, th),
        _poi_text(cur, th),
    ]
    st.write("\n\n".join(why_lines))

# =============================================================================
# MAIN
# =============================================================================
def render_suc_zarar_tahmini():
    _apply_global_css()

    st.markdown("# 🧭 Suç & Zarar Etkisi — Operasyon Paneli")
    st.caption("Amaç: **Ne zaman, nerede, neye dikkat etmeli ve ne yapmalı?** (Karar desteğidir.)")

    if _IMPORT_SRC_ERR is not None:
        st.error("`src.io_data` import edilemedi. `src/` klasörünü ve yolları kontrol edin.")
        st.code(repr(_IMPORT_SRC_ERR))
        return

    raw = load_ops_ready()
    if raw is None or raw.empty:
        st.error(
            "Ops-ready veri bulunamadı/boş.\n\nAranan dosyalardan bazıları:\n"
            + "\n".join([f"- {p}" for p in OPS_CANDIDATES[:6]])
        )
        return

    df = normalize_ops(raw)

    gj = load_geojson()
    if not gj:
        st.error(f"GeoJSON bulunamadı: `{GEOJSON_PATH}`")
        return

    now_sf = datetime.now(ZoneInfo(TARGET_TZ))
    dates = sorted(df["date_norm"].dropna().unique())
    if not dates:
        st.error("Veride geçerli tarih bulunamadı.")
        return

    default_date = pd.Timestamp(now_sf.date())
    if default_date not in dates:
        past = [d for d in dates if d <= default_date]
        default_date = max(past) if past else dates[0]

    hr_labels = sorted(df["hour_range"].dropna().astype(str).unique().tolist())
    default_hr = _hour_to_bucket(now_sf.hour, hr_labels) or (hr_labels[0] if hr_labels else "00-03")

    c1, c2, c3 = st.columns([1.25, 1.0, 1.4])
    with c1:
        sel_date = st.selectbox(
            "📅 Tarih",
            options=dates,
            index=dates.index(default_date) if default_date in dates else 0,
            format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d"),
        )
    with c2:
        sel_hr = st.selectbox("⏰ Saat dilimi", options=hr_labels, index=hr_labels.index(default_hr) if default_hr in hr_labels else 0)
    with c3:
        mode = segmented_compat("🗺️ Harita modu", options=["Risk", "Zarar", "Ops Öncelik"], default="Ops Öncelik")

    df_slice = _slice_by_date_hour(df, sel_date, sel_hr)
    if df_slice.empty:
        st.warning("Seçili tarih/saat dilimi için kayıt yok.")
        return

    # Sidebar summary
    st.sidebar.markdown("### 🎯 Vardiya Özeti")
    st.sidebar.caption(f"SF saati: **{now_sf:%Y-%m-%d %H:%M}**")
    st.sidebar.write(f"**Tarih:** {pd.Timestamp(sel_date).strftime('%Y-%m-%d')}")
    st.sidebar.write(f"**Dilim:** {sel_hr}")
    st.sidebar.write(f"**Mod:** {mode}")

    # KPI
    kpi = _kpi_text_levels(df_slice)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
            <div class="sutam-card">
              <h3>Aktif Bölge</h3>
              <div class="sutam-kpi"><div class="v">{kpi["cells"]}</div><div class="t">hücre (GEOID)</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="sutam-card">
              <h3>Şehir Geneli Risk</h3>
              <div class="sutam-kpi"><div class="v">{kpi["risk"]}</div><div class="t">bu dilim</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="sutam-card">
              <h3>Olay Yoğunluğu</h3>
              <div class="sutam-kpi"><div class="v">{kpi["event"]}</div><div class="t">beklenen</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div class="sutam-card">
              <h3>Etki (Zarar)</h3>
              <div class="sutam-kpi"><div class="v">{kpi["harm"]}</div><div class="t">beklenen</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # Map (hover yok)
    gj_enriched = enrich_geojson_fill_only(gj, df_slice, mode)
    draw_map_no_hover(gj_enriched)
    st.caption("Harita: sadece renkli bağlam. Hover kapalı. Brifing için GEOID seç.")
    st.divider()

    # GEOID Selection + Brief
    st.subheader("🔎 GEOID Seç → Brifing")
    geoids_in_slice = sorted(df_slice["geoid"].dropna().astype(str).unique().tolist())

    cL, cR = st.columns([1.0, 2.0])
    with cL:
        # hızlı arama: text + dropdown
        typed = st.text_input("GEOID (11 haneli) yaz", value="", placeholder="06075030101")
        typed_geoid = _digits11(typed) if typed else ""

        picked = st.selectbox(
            "veya listeden seç",
            options=[""] + geoids_in_slice,
            index=0,
            format_func=lambda x: ("Seçilmedi" if x == "" else x),
        )

        sel_geoid = typed_geoid if typed_geoid else picked
        if sel_geoid:
            st.caption(f"Seçilen GEOID: **{sel_geoid}**")
        else:
            st.caption("GEOID seçince brifing sağ tarafta görünür.")

    with cR:
        if not sel_geoid:
            st.info("Sağ panel: GEOID seçilince brifing + 7 günlük özet burada görünür.")
        else:
            info = build_geoid_brief(df_all=df, df_slice=df_slice, geoid=sel_geoid, sel_hr=sel_hr)
            if not info:
                st.warning("Bu GEOID için veri bulunamadı.")
            else:
                render_brief_panel(info, df_slice=df_slice)

    st.divider()

    # Model/meta footer (çok kısa)
    st.subheader("🧷 Model & Çalıştırma Bilgisi")
    meta_cols = ["model_version", "run_id", "forecast_generated_at", "forecast_horizon_days"]
    parts = []
    for c in meta_cols:
        if c in df.columns:
            v = df_slice[c].dropna()
            if len(v):
                parts.append(f"**{c}:** {str(v.iloc[0])}")
    st.write(" • ".join(parts) if parts else "—")
