# -*- coding: utf-8 -*-
# pages/Suc_Zarar_Tahmini.py
# SUTAM — Suç + Zarar (HARM) Tahmini  |  Operasyonel Karar Destek (Kurumsal, Kolluk-Dostu)
#
# Kaynak: deploy/forecast_7d_ops_ready.parquet (ya da data/forecast_7d_ops_ready.parquet)
# - Bu sayfa "vay be" için ops_ready'nin tüm ops_* + driver + flags + harm/risk alanlarını kullanır.
# - Harita modları: Risk / Zarar / Ops Öncelik
# - Tooltip: risk + p + beklenen suç + beklenen zarar + top3 + driver + bayraklar + ops aksiyon
# - Sidebar: vardiya özeti, kritik hücreler, top tablo, bayrak sayacı
# - Alt: seçili GEOID için 7 günlük kritik saatler + suç karması + açıklama/eylem
#
# NOT: POI merge vb. upstream veri sorunları bu sayfayı bozmaz; alanlar varsa kullanır, yoksa gracefully düşer.

from __future__ import annotations

import os
import json
import math
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# -----------------------------------------------------------------------------
# SAFE IMPORT (src yoksa sayfa çökmesin)
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

# Kurumsal Likert (Risk)
LIKERT_RISK = {
    1: ("Çok Düşük",  [46, 204, 113]),
    2: ("Düşük",      [88, 214, 141]),
    3: ("Orta",       [241, 196, 15]),
    4: ("Yüksek",     [230, 126, 34]),
    5: ("Çok Yüksek", [192, 57, 43]),
}
DEFAULT_FILL = [220, 220, 220]

# Zarar (HARM) Likert: ayrı renk mantığı (daha “uyarı” ton)
LIKERT_HARM = {
    1: ("Düşük Etki",  [96, 165, 250]),
    2: ("Orta Etki",   [76, 147, 245]),
    3: ("Yüksek Etki", [241, 196, 15]),
    4: ("Çok Yüksek",  [230, 126, 34]),
    5: ("Kritik Etki", [192, 57, 43]),
}

# Ops öncelik (kolluk dili): ops_rank_score / expected_harm öncelikli
LIKERT_OPS = {
    1: ("İzle",        [196, 226, 255]),
    2: ("Dikkat",      [148, 202, 255]),
    3: ("Öncelikli",   [241, 196, 15]),
    4: ("Çok Öncelik", [230, 126, 34]),
    5: ("Acil",        [192, 57, 43]),
}

# =============================================================================
# CSS (Kurumsal + Tooltip fix + compact)
# =============================================================================
def _apply_global_css():
    st.markdown(
        """
        <style>
          /* Sayfa max genişlik */
          .block-container { padding-top: 1rem; padding-bottom: 2rem; }

          /* Kurumsal kart */
          .sutam-card {
            border: 1px solid rgba(148,163,184,.35);
            border-radius: 16px;
            padding: 14px 14px;
            background: rgba(2,6,23,.25);
            box-shadow: 0 14px 40px rgba(0,0,0,.12);
          }
          .sutam-card h3 { margin: 0 0 8px 0; font-size: 14px; letter-spacing:.2px; }
          .sutam-kpi {
            display:flex; gap:12px; align-items:baseline; flex-wrap:wrap;
            margin-top:6px;
          }
          .sutam-kpi .v { font-weight:900; font-size: 20px; }
          .sutam-kpi .t { color: rgba(226,232,240,.9); font-size: 12px; }

          /* Deck.gl tooltip */
          .deckgl-tooltip {
            max-width: 380px !important;
            max-height: 360px !important;
            overflow: auto !important;
            padding: 10px 12px !important;
            line-height: 1.25 !important;
            border-radius: 12px !important;
            box-shadow: 0 10px 30px rgba(0,0,0,.25) !important;
          }
          .deckgl-tooltip hr { margin: 8px 0 !important; opacity: .25 !important; }
          .deckgl-tooltip {
            transform: translate(12px, 12px) !important;
          }

          /* Küçük rozet */
          .badge {
            display:inline-block;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            border: 1px solid rgba(148,163,184,.35);
            margin-right: 6px;
          }

          /* Sidebar başlıkları kompakt */
          section[data-testid="stSidebar"] .stMarkdown h3 { margin-bottom: .35rem; }
          section[data-testid="stSidebar"] .stMarkdown p { margin-bottom: .35rem; }

          /* Tablo başlıkları */
          .stDataFrame { border-radius: 12px; overflow: hidden; }

        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# UTIL HELPERS
# =============================================================================
def _first_existing(paths: list[str]) -> str | None:
    for p in paths:
        if os.path.exists(p):
            return p
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

def _fmt3(x) -> str:
    v = _safe_float(x, np.nan)
    return "—" if not np.isfinite(v) else f"{v:.3f}"

def _fmt_intish(x) -> str:
    v = _safe_float(x, np.nan)
    if not np.isfinite(v):
        return "—"
    return f"{int(round(v))}"

def _fmt_expected_range(x) -> str:
    v = _safe_float(x, np.nan)
    if not np.isfinite(v):
        return "—"
    v = max(0.0, v)
    lo = int(np.floor(v))
    hi = int(np.ceil(v))
    return f"~{lo}" if lo == hi else f"~{lo}–{hi}"

def _fmt_money_like(x) -> str:
    # harm "para" değil ama etki büyüklüğü; okunur format
    v = _safe_float(x, np.nan)
    if not np.isfinite(v):
        return "—"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 100:
        return f"{v:.0f}"
    return f"{v:.1f}"

def _parse_range(tok: str):
    # "21-24" -> (21,24) end exclusive
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

    # wrap-around e.g. "21-3"
    for lab, s, e in parsed:
        if s > e and (h >= s or h < e):
            return lab

    return parsed[0][0] if parsed else None

def _coerce_bool(x) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x is None:
        return False
    s = str(x).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    return False

def _pct(x) -> str:
    v = _safe_float(x, np.nan)
    if not np.isfinite(v):
        return "—"
    return f"{100*v:.0f}%"

def _clip01(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    return x.fillna(0.0).clip(0.0, 1.0)

def _quantile_likert(series: pd.Series, n=5, neutral=3) -> pd.Series:
    v = pd.to_numeric(series, errors="coerce")
    if v.notna().sum() < 10 or v.nunique(dropna=True) <= 1:
        return pd.Series([neutral] * len(v), index=v.index)
    try:
        return pd.qcut(v.rank(method="first"), n, labels=list(range(1, n + 1))).astype(int)
    except Exception:
        qs = [v.quantile(i / n) for i in range(1, n)]
        out = pd.Series(neutral, index=v.index)
        prev = -np.inf
        for k, qv in enumerate(qs, start=1):
            out[(v > prev) & (v <= qv)] = k
            prev = qv
        out[v > prev] = n
        return out.astype(int)

def _dominant_flag_badges(r: pd.Series) -> str:
    # Kolluk için "bağlam bayrakları" rozetleri
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

def _impact_badge(x: str) -> str:
    s = str(x or "").strip()
    if s.lower().startswith("high") or s.lower().startswith("critical"):
        return "⚠️ Yüksek Etki"
    return "Normal Etki"

def _risk_text_hint(level: str) -> str:
    level = str(level or "").strip()
    if level in ("Critical", "Çok Yüksek", "Çok Yüksek Risk", "Very High"):
        return "Acil görünürlük + hedefli müdahale"
    if level in ("High", "Yüksek"):
        return "Hedefli devriye + giriş/çıkış kontrolü"
    if level in ("Medium", "Orta"):
        return "Kısa tur döngüsü + caydırıcılık"
    return "Rutin devriye + gözlemsel teyit"


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
        # fallback
        if p.lower().endswith(".parquet"):
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)

    if df is None or getattr(df, "empty", True):
        return pd.DataFrame()

    # prepare_forecast varsa kullan (kolon normalize/temizlik)
    if prepare_forecast is not None:
        try:
            df = prepare_forecast(df, gp=None)
        except Exception:
            pass

    return df


# =============================================================================
# NORMALIZE + FILTER CORE
# =============================================================================
def normalize_ops(df: pd.DataFrame) -> pd.DataFrame:
    """
    ops_ready kolonlarını toleranslı şekilde normalize eder.
    Bu fonksiyon sayfanın bel kemiği.
    """
    d = df.copy()

    # GEOID
    geoid_col = _pick_col(d, ["GEOID", "geoid"])
    if geoid_col:
        d["geoid"] = d[geoid_col].map(_digits11)
    else:
        d["geoid"] = ""

    # date
    date_col = _pick_col(d, ["date", "dt", "datetime"])
    if date_col:
        d["_dt"] = pd.to_datetime(d[date_col], errors="coerce")
    else:
        d["_dt"] = pd.NaT
    d["date_norm"] = d["_dt"].dt.normalize()

    # hour_range (ops_ready genelde var)
    hr_col = _pick_col(d, ["hour_range", "hour_bucket"])
    if hr_col:
        d["hour_range"] = d[hr_col].astype(str)
    else:
        d["hour_range"] = "00-24"

    # p_event / risk_prob
    p_col = _pick_col(d, ["p_event", "risk_prob", "risk_score"])
    if p_col:
        d["p_event"] = pd.to_numeric(d[p_col], errors="coerce")
    else:
        d["p_event"] = np.nan

    # expected_count
    ex_col = _pick_col(d, ["expected_count", "expected_crimes"])
    if ex_col:
        d["expected_count"] = pd.to_numeric(d[ex_col], errors="coerce")
    else:
        d["expected_count"] = np.nan

    # expected_harm
    harm_col = _pick_col(d, ["expected_harm", "harm_expected", "harm"])
    if harm_col:
        d["expected_harm"] = pd.to_numeric(d[harm_col], errors="coerce")
    else:
        d["expected_harm"] = np.nan

    # ops_rank_score
    ops_score_col = _pick_col(d, ["ops_rank_score", "risk_score"])
    if ops_score_col:
        d["ops_rank_score"] = pd.to_numeric(d[ops_score_col], errors="coerce")
    else:
        d["ops_rank_score"] = d["expected_harm"]

    # risk_level / risk_bin
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

    # top categories
    for i in (1, 2, 3):
        c = _pick_col(d, [f"top{i}_category", f"top{i}_cat"])
        d[f"top{i}_category"] = d[c].astype(str).replace("nan", "").fillna("") if c else ""

        sh = _pick_col(d, [f"top{i}_share"])
        d[f"top{i}_share"] = pd.to_numeric(d[sh], errors="coerce") if sh else np.nan

    # drivers & profile
    for k in ["primary_driver", "secondary_driver", "driver_profile"]:
        col = _pick_col(d, [k])
        d[k] = d[col].astype(str) if col else ""

    # flags
    for k in ["weather_flag", "calls_flag", "neighbor_flag", "transit_flag", "poi_flag", "time_flag"]:
        col = _pick_col(d, [k])
        if col:
            d[k] = d[col].apply(_coerce_bool)
        else:
            d[k] = False

    # ops texts
    for k in ["ops_actions_short", "ops_actions", "ops_reasons", "ops_reasons_long", "ops_actions_long"]:
        col = _pick_col(d, [k])
        d[k] = d[col].astype(str).fillna("") if col else ""

    # audit / meta
    for k in ["model_version", "run_id", "audit_tag", "forecast_generated_at", "forecast_horizon_days", "risk_rank"]:
        col = _pick_col(d, [k])
        d[k] = d[col] if col else ""

    return d


# =============================================================================
# LIKERT COMPUTE FOR MODES (RISK / HARM / OPS)
# =============================================================================
def compute_mode_likert(df_slice: pd.DataFrame, mode: str) -> tuple[pd.Series, dict]:
    """
    mode:
      - "Risk": p_event / risk_score dağılım quintile
      - "Zarar": expected_harm dağılım quintile
      - "Ops Öncelik": ops_rank_score (yoksa expected_harm) quintile
    """
    if df_slice.empty:
        lik = pd.Series([], dtype=int)
        return lik, {"cuts": [np.nan] * 4, "source_col": None}

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
    cuts = v.quantile([0.2, 0.4, 0.6, 0.8]).values.tolist() if v.notna().sum() >= 10 else [np.nan]*4
    return lik, {"cuts": cuts, "source_col": src, "palette": palette}


# =============================================================================
# GEOJSON ENRICH (OPS READY)
# =============================================================================
def enrich_geojson_ops(gj: dict, df_slice: pd.DataFrame, mode: str) -> dict:
    if not gj or df_slice.empty:
        return gj

    d = df_slice.copy()

    # Likert
    lik, meta = compute_mode_likert(d, mode)
    d["_lik"] = lik.clip(1, 5)
    palette = meta.get("palette") or LIKERT_RISK

    # Fill
    d["_fill"] = d["_lik"].map(lambda k: palette[int(k)][1])

    # Tooltip values (metin)
    d["_p_txt"] = d["p_event"].map(_fmt3)
    d["_exp_txt"] = d["expected_count"].map(_fmt_expected_range)
    d["_harm_txt"] = d["expected_harm"].map(_fmt_money_like)

    d["_impact_badge"] = d.get("impact_flag", "").astype(str).apply(_impact_badge) if "impact_flag" in d.columns else "—"

    # Tek cümle “saha aksiyonu”
    d["_action_1"] = d["ops_actions_short"]
    d.loc[d["_action_1"].astype(str).str.len() < 3, "_action_1"] = d["ops_actions"]
    d.loc[d["_action_1"].astype(str).str.len() < 3, "_action_1"] = d["risk_level"].apply(_risk_text_hint)

    # Bayrak rozetleri
    d["_flags"] = d.apply(_dominant_flag_badges, axis=1)

    # Driver etiketleri
    d["_driver"] = d["primary_driver"].apply(_driver_label)
    d["_driver2"] = d["secondary_driver"].apply(_driver_label)
    d["_profile"] = d["driver_profile"].replace("", "—")

    # Top3 readable
    def _top3_str(r):
        arr = []
        for i in (1, 2, 3):
            c = str(r.get(f"top{i}_category") or "").strip()
            if c and c.lower() != "unknown":
                arr.append(c)
        return " • ".join(arr) if arr else "—"
    d["_top3"] = d.apply(_top3_str, axis=1)

    # Aynı GEOID birden fazla satırsa:
    #  - seçilen mod için en “yüksek” olan kalsın (likert + ops_rank_score tie-break)
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

        # defaults
        props["fill_color"] = DEFAULT_FILL
        props["mode_name"] = mode

        props["risk_level"] = ""
        props["p_event_txt"] = "—"
        props["expected_txt"] = "—"
        props["harm_txt"] = "—"
        props["top3"] = "—"
        props["driver_txt"] = "—"
        props["profile_txt"] = "—"
        props["flags_txt"] = "—"
        props["ops_action"] = "—"

        if key and key in d.index:
            r = d.loc[key]
            props["fill_color"] = r.get("_fill", DEFAULT_FILL)

            props["risk_level"] = str(r.get("risk_level", "") or "")
            props["p_event_txt"] = str(r.get("_p_txt", "—") or "—")
            props["expected_txt"] = str(r.get("_exp_txt", "—") or "—")
            props["harm_txt"] = str(r.get("_harm_txt", "—") or "—")

            props["top3"] = str(r.get("_top3", "—") or "—")
            props["driver_txt"] = str(r.get("_driver", "—") or "—")
            props["profile_txt"] = str(r.get("_profile", "—") or "—")
            props["flags_txt"] = str(r.get("_flags", "—") or "—")
            props["ops_action"] = str(r.get("_action_1", "—") or "—")

        out_feats.append({**feat, "properties": props})

    return {**gj, "features": out_feats}


# =============================================================================
# MAP RENDER
# =============================================================================
def draw_map_ops(gj: dict):
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
            "<div><b>Suç olasılığı (p):</b> {p_event_txt}</div>"
            "<div><b>Beklenen suç:</b> {expected_txt}</div>"
            "<div><b>Beklenen zarar:</b> {harm_txt}</div>"
            "<hr/>"
            "<div><b>En olası suçlar:</b> {top3}</div>"
            "<div><b>Birincil etken:</b> {driver_txt} <span style='opacity:.75'>(profil: {profile_txt})</span></div>"
            "<div><b>Bağlam:</b> {flags_txt}</div>"
            "<hr/>"
            "<div style='font-weight:800;'>Kolluk Eylemi</div>"
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


# =============================================================================
# OPS BRIEF HELPERS
# =============================================================================
def _slice_by_date_hour(df: pd.DataFrame, sel_date: pd.Timestamp, hr_label: str) -> pd.DataFrame:
    out = df[(df["date_norm"] == sel_date) & (df["hour_range"].astype(str) == str(hr_label))].copy()
    return out

def _topn_table(df_slice: pd.DataFrame, n=15, rank_by="expected_harm") -> pd.DataFrame:
    if df_slice.empty:
        return pd.DataFrame()

    if rank_by not in df_slice.columns:
        rank_by = "ops_rank_score" if "ops_rank_score" in df_slice.columns else "expected_harm"

    d = df_slice.copy()
    d["_rk"] = pd.to_numeric(d.get(rank_by, np.nan), errors="coerce").fillna(-np.inf)

    # kısa ops metin
    def _cat3(r):
        arr = []
        for i in (1, 2, 3):
            c = str(r.get(f"top{i}_category") or "").strip()
            if c and c.lower() != "unknown":
                arr.append(c)
        return ", ".join(arr[:3]) if arr else "—"

    short_cols = [
        "geoid", "risk_level", "p_event", "expected_count", "expected_harm",
        "primary_driver", "driver_profile",
        "ops_actions_short"
    ]
    keep = [c for c in short_cols if c in d.columns]
    d["top3"] = d.apply(_cat3, axis=1)
    keep += ["top3"]

    out = d.sort_values("_rk", ascending=False).head(n)[keep].copy()
    out = out.rename(columns={
        "geoid": "GEOID",
        "risk_level": "Risk",
        "p_event": "p",
        "expected_count": "Beklenen Suç",
        "expected_harm": "Beklenen Zarar",
        "primary_driver": "Birincil Etken",
        "driver_profile": "Profil",
        "ops_actions_short": "Öneri",
        "top3": "Olası Suçlar",
    })

    # format
    if "p" in out.columns:
        out["p"] = out["p"].map(_fmt3)
    if "Beklenen Suç" in out.columns:
        out["Beklenen Suç"] = out["Beklenen Suç"].map(_fmt_expected_range)
    if "Beklenen Zarar" in out.columns:
        out["Beklenen Zarar"] = out["Beklenen Zarar"].map(_fmt_money_like)
    if "Birincil Etken" in out.columns:
        out["Birincil Etken"] = out["Birincil Etken"].map(_driver_label)

    return out

def _ops_kpis(df_slice: pd.DataFrame) -> dict:
    if df_slice.empty:
        return {
            "cells": 0,
            "mean_p": np.nan,
            "sum_expected": np.nan,
            "sum_harm": np.nan,
            "hi_impact": 0,
            "flags": {},
        }

    p_mean = pd.to_numeric(df_slice["p_event"], errors="coerce").mean()
    sum_expected = pd.to_numeric(df_slice["expected_count"], errors="coerce").fillna(0).sum()
    sum_harm = pd.to_numeric(df_slice["expected_harm"], errors="coerce").fillna(0).sum()

    impact_col = _pick_col(df_slice, ["impact_flag"])
    if impact_col:
        hi_impact = (df_slice[impact_col].astype(str).str.contains("High|Critical", case=False, regex=True)).sum()
    else:
        hi_impact = 0

    flags = {
        "calls_flag": int(df_slice["calls_flag"].sum()) if "calls_flag" in df_slice.columns else 0,
        "neighbor_flag": int(df_slice["neighbor_flag"].sum()) if "neighbor_flag" in df_slice.columns else 0,
        "poi_flag": int(df_slice["poi_flag"].sum()) if "poi_flag" in df_slice.columns else 0,
        "transit_flag": int(df_slice["transit_flag"].sum()) if "transit_flag" in df_slice.columns else 0,
        "weather_flag": int(df_slice["weather_flag"].sum()) if "weather_flag" in df_slice.columns else 0,
        "time_flag": int(df_slice["time_flag"].sum()) if "time_flag" in df_slice.columns else 0,
    }

    return {
        "cells": int(df_slice["geoid"].nunique()),
        "mean_p": p_mean,
        "sum_expected": float(sum_expected),
        "sum_harm": float(sum_harm),
        "hi_impact": int(hi_impact),
        "flags": flags,
    }


# =============================================================================
# SELECTED GEOID ANALYSIS
# =============================================================================
def geoid_week_profile(df_all: pd.DataFrame, geoid: str) -> dict:
    """
    Seçili GEOID için 7 gün:
      - en kritik saat dilimleri (harm + ops)
      - top kategoriler (top1 ağırlıklı)
      - kısa özet cümle
    """
    g = df_all[df_all["geoid"] == geoid].copy()
    if g.empty:
        return {}

    # En kritik saat dilimleri: expected_harm ort / ops_rank_score ort
    g["_harm"] = pd.to_numeric(g["expected_harm"], errors="coerce").fillna(0.0)
    g["_ops"] = pd.to_numeric(g["ops_rank_score"], errors="coerce").fillna(0.0)
    by_hr = g.groupby("hour_range", dropna=False)[["_harm", "_ops"]].mean().sort_values(["_harm", "_ops"], ascending=False)

    top_hours = by_hr.head(3).index.astype(str).tolist()

    # Top kategoriler: top1_category sayım + share
    c = g["top1_category"].astype(str).replace("nan", "")
    c = c[c.str.len() > 0]
    top_cats = c.value_counts().head(5).index.tolist()

    # Sık bayrak
    flags = {}
    for k in ["calls_flag", "neighbor_flag", "poi_flag", "transit_flag", "weather_flag", "time_flag"]:
        if k in g.columns:
            flags[k] = int(g[k].sum())

    # Driver dağılımı
    pdv = g["primary_driver"].astype(str)
    pdv = pdv[pdv.str.len() > 0]
    top_driver = pdv.value_counts().head(1).index.tolist()
    top_driver = top_driver[0] if top_driver else "risk_core"

    # Özet
    summary = f"En kritik dilimler: **{', '.join(top_hours) if top_hours else '—'}**. "
    if top_cats:
        summary += f"Öne çıkan suçlar: **{', '.join(top_cats[:3])}**. "
    summary += f"Birincil etken eğilimi: **{_driver_label(top_driver)}**."

    return {
        "top_hours": top_hours,
        "top_cats": top_cats,
        "flags": flags,
        "top_driver": top_driver,
        "summary": summary,
        "by_hr": by_hr.reset_index().rename(columns={"hour_range": "Saat Dilimi", "_harm": "Ort. Zarar", "_ops": "Ort. Ops"}),
    }


# =============================================================================
# LEGEND POPOVER (Risk/Harm/Ops)
# =============================================================================
def legend_popover(mode: str, meta: dict):
    cuts = meta.get("cuts", [np.nan] * 4)
    src = meta.get("source_col") or "—"
    palette = meta.get("palette") or (LIKERT_RISK if mode == "Risk" else (LIKERT_HARM if mode == "Zarar" else LIKERT_OPS))

    with st.popover("🎨 Ölçek", use_container_width=False):
        st.markdown(
            f"**{mode}** düzeyi, **seçili tarih + saat dilimindeki hücre dağılımına göre** (%20’lik dilimler) üretilir."
        )
        st.caption(f"Kaynak metrik: **{src}**")

        def _qtxt(x):
            return "—" if not np.isfinite(_safe_float(x)) else f"{float(x):.3f}"

        q20, q40, q60, q80 = cuts if len(cuts) == 4 else [np.nan]*4
        rows = [
            (1, palette[1][0], "0–20",  None, _qtxt(q20)),
            (2, palette[2][0], "20–40", _qtxt(q20), _qtxt(q40)),
            (3, palette[3][0], "40–60", _qtxt(q40), _qtxt(q60)),
            (4, palette[4][0], "60–80", _qtxt(q60), _qtxt(q80)),
            (5, palette[5][0], "80–100",_qtxt(q80), None),
        ]

        for k, label, pct, lo, hi in rows:
            rgb = palette[k][1]
            rng = f"{lo}–{hi}" if (lo is not None and hi is not None) else (f"≤ {hi}" if lo is None else f"> {lo}")
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; padding:8px 10px; border:1px solid #e2e8f0; border-radius:12px; margin-bottom:8px;">
                  <div style="display:flex; align-items:center; gap:10px;">
                    <div style="width:14px;height:14px;border-radius:4px;background:rgb({rgb[0]},{rgb[1]},{rgb[2]});"></div>
                    <div style="font-weight:900;">{k}</div>
                    <div>{label}</div>
                  </div>
                  <div style="color:#64748b; font-size:12px; text-align:right;">
                    %{pct}<br/>
                    <span style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;">
                      {rng}
                    </span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =============================================================================
# MAIN PAGE ENTRY
# =============================================================================
def render_suc_zarar_tahmini():
    _apply_global_css()

    # Header
    st.markdown("# 🧭 Suç & Zarar Etkisi — Operasyonel Karar Destek")
    st.caption(
        "Bu ekranın amacı: **Ne zaman, nerede, neye dikkat etmeli ve ne yapmalı?** "
        "Model çıktıları karar desteğidir; saha bilgisi ve amir değerlendirmesiyle birlikte yorumlanır."
    )

    # Import check
    if _IMPORT_SRC_ERR is not None:
        st.error("`src.io_data` modülü import edilemedi. `src/` klasörünü ve dosya yollarını kontrol edin.")
        st.code(repr(_IMPORT_SRC_ERR))
        return

    # Load
    raw = load_ops_ready()
    if raw is None or raw.empty:
        st.error(
            "Ops-ready veri bulunamadı/boş.\n\nBeklenen dosyalardan biri:\n"
            + "\n".join([f"- {p}" for p in OPS_CANDIDATES[:4]])
        )
        return

    df = normalize_ops(raw)

    # GeoJSON
    gj = load_geojson()
    if not gj:
        st.error(f"GeoJSON bulunamadı: `{GEOJSON_PATH}`")
        return

    # Time controls
    now_sf = datetime.now(ZoneInfo(TARGET_TZ))
    dates = sorted(df["date_norm"].dropna().unique())
    if not dates:
        st.error("Veride geçerli tarih bulunamadı.")
        return

    default_date = pd.Timestamp(now_sf.date())
    if default_date not in dates:
        # en yakın geçmiş tarih
        past = [d for d in dates if d <= default_date]
        default_date = max(past) if past else dates[0]

    hr_labels = sorted(df["hour_range"].dropna().astype(str).unique().tolist())
    default_hr = _hour_to_bucket(now_sf.hour, hr_labels) or (hr_labels[0] if hr_labels else "00-03")

    # Top controls row
    c1, c2, c3, c4 = st.columns([1.25, 1.0, 1.2, 1.0])
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
        mode = st.segmented_control(
            "🗺️ Harita modu",
            options=["Risk", "Zarar", "Ops Öncelik"],
            default="Ops Öncelik",
        )
    with c4:
        topn = st.selectbox("📌 Top hücre", options=[10, 15, 20, 30, 50], index=1)

    df_slice = _slice_by_date_hour(df, sel_date, sel_hr)
    if df_slice.empty:
        st.warning("Seçili tarih/saat dilimi için kayıt yok.")
        return

    # Mode Likert meta for legend
    lik, meta = compute_mode_likert(df_slice, mode)

    # KPI row
    k = _ops_kpis(df_slice)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
            <div class="sutam-card">
              <h3>Hücre Sayısı</h3>
              <div class="sutam-kpi"><div class="v">{k["cells"]}</div><div class="t">aktif GEOID</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="sutam-card">
              <h3>Ortalama Olasılık</h3>
              <div class="sutam-kpi"><div class="v">{_fmt3(k["mean_p"])}</div><div class="t">p_event ort</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="sutam-card">
              <h3>Beklenen Suç</h3>
              <div class="sutam-kpi"><div class="v">{_fmt_intish(k["sum_expected"])}</div><div class="t">toplam (slice)</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div class="sutam-card">
              <h3>Beklenen Zarar</h3>
              <div class="sutam-kpi"><div class="v">{_fmt_money_like(k["sum_harm"])}</div><div class="t">toplam (slice) • HiImpact: {k["hi_impact"]}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Sidebar ops summary
    st.sidebar.markdown("### 🎯 Vardiya Özeti")
    st.sidebar.caption(f"SF saati: **{now_sf:%Y-%m-%d %H:%M}**")
    st.sidebar.write(f"**Tarih:** {pd.Timestamp(sel_date).strftime('%Y-%m-%d')}")
    st.sidebar.write(f"**Dilim:** {sel_hr}")
    st.sidebar.write(f"**Mod:** {mode}")

    # Flags count
    flags = k["flags"] or {}
    if flags:
        st.sidebar.markdown("### 🚩 Bağlam Bayrakları (adet)")
        st.sidebar.write(
            f"☎️ Çağrı: **{flags.get('calls_flag',0)}**  • "
            f"🧭 Komşu: **{flags.get('neighbor_flag',0)}**  • "
            f"📍 POI: **{flags.get('poi_flag',0)}**"
        )
        st.sidebar.write(
            f"🚇 Transit: **{flags.get('transit_flag',0)}**  • "
            f"🌧️ Hava: **{flags.get('weather_flag',0)}**  • "
            f"🕒 Zaman: **{flags.get('time_flag',0)}**"
        )

    # Legend popover
    legend_popover(mode, meta)

    st.divider()

    # Map
    gj_enriched = enrich_geojson_ops(gj, df_slice, mode)
    draw_map_ops(gj_enriched)

    st.caption("İpucu: Hücre üzerine gel (hover) → detayları gör. Harita modu: Risk / Zarar / Ops Öncelik.")

    st.divider()

    # Ops Brief Table
    st.subheader("📌 Operasyon Öncelik Listesi")
    rank_by = "expected_harm" if mode in ("Zarar", "Ops Öncelik") else "p_event"
    tbl = _topn_table(df_slice, n=int(topn), rank_by=rank_by)

    if tbl.empty:
        st.info("Tablo üretilemedi.")
    else:
        st.dataframe(tbl, use_container_width=True, hide_index=True)

        # Export filtered slice + top table
        cexp1, cexp2 = st.columns([1, 1])
        with cexp1:
            csv1 = df_slice.copy()
            # insan-okur alanları ekleyelim
            csv1["flags_txt"] = csv1.apply(_dominant_flag_badges, axis=1)
            csv1["primary_driver_label"] = csv1["primary_driver"].map(_driver_label)
            csv1["secondary_driver_label"] = csv1["secondary_driver"].map(_driver_label)
            out_bytes = csv1.to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇️ Dilim CSV indir (tümü)", data=out_bytes, file_name="ops_slice.csv", mime="text/csv")
        with cexp2:
            out_bytes2 = tbl.to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇️ Top Liste CSV indir", data=out_bytes2, file_name="ops_top_list.csv", mime="text/csv")

    st.divider()

    # Selected GEOID Analysis (manual input)
    st.subheader("🔎 Seçili Bölge Analizi (GEOID)")
    left, right = st.columns([1.0, 2.0])
    with left:
        sel_geoid_raw = st.text_input("GEOID gir (11 haneli)", value="", placeholder="06075030101")
        sel_geoid = _digits11(sel_geoid_raw) if sel_geoid_raw else ""
        st.caption("Bu panel, seçili GEOID için **7 gün** boyunca kritik saatleri ve sürücüleri özetler.")

    with right:
        if not sel_geoid:
            st.info("GEOID girince, 7 günlük kritik zamanlar + suç karması + etken profili burada görünecek.")
        else:
            prof = geoid_week_profile(df, sel_geoid)
            if not prof:
                st.warning("Bu GEOID için veride kayıt yok.")
            else:
                st.markdown(prof["summary"])

                # 7 günlük kritik saat tablosu
                dhr = prof.get("by_hr")
                if isinstance(dhr, pd.DataFrame) and not dhr.empty:
                    dhr2 = dhr.copy()
                    dhr2["Ort. Zarar"] = dhr2["Ort. Zarar"].map(_fmt_money_like)
                    dhr2["Ort. Ops"] = dhr2["Ort. Ops"].map(_fmt3)
                    st.dataframe(dhr2.head(10), use_container_width=True, hide_index=True)

                # suç karması
                top_cats = prof.get("top_cats", [])
                if top_cats:
                    st.markdown("**Öne çıkan suçlar:** " + " • ".join([f"`{c}`" for c in top_cats[:5]]))

                # bayraklar
                flags_g = prof.get("flags", {})
                if flags_g:
                    st.markdown(
                        f"**Bayrak sayımı (7 gün):** "
                        f"☎️ {flags_g.get('calls_flag',0)} • 🧭 {flags_g.get('neighbor_flag',0)} • 📍 {flags_g.get('poi_flag',0)} • "
                        f"🚇 {flags_g.get('transit_flag',0)} • 🌧️ {flags_g.get('weather_flag',0)} • 🕒 {flags_g.get('time_flag',0)}"
                    )

                # En güncel (seçili dilim) satırdan öneri çekelim
                cur = df_slice[df_slice["geoid"] == sel_geoid].copy()
                if not cur.empty:
                    cur = cur.sort_values(["expected_harm", "ops_rank_score"], ascending=False).head(1).iloc[0]
                    st.markdown("### 🧾 Bu dilim için tek cümle eylem")
                    action = str(cur.get("ops_actions_short") or "") or str(cur.get("ops_actions") or "")
                    if not action:
                        action = _risk_text_hint(str(cur.get("risk_level")))
                    st.success(action)

                    st.markdown("### 🧠 Neden (kısa, kolluk dili)")
                    why = str(cur.get("ops_reasons") or "")
                    if not why:
                        why = f"{_driver_label(cur.get('primary_driver'))} • {_dominant_flag_badges(cur)}"
                    st.write(why)

    st.divider()

    # Audit / model footer
    st.subheader("🧷 Model & Çalıştırma Bilgisi")
    # tek satır meta
    meta_cols = ["model_version", "run_id", "audit_tag", "forecast_generated_at", "forecast_horizon_days"]
    m = {}
    for c in meta_cols:
        if c in df.columns:
            # slice'tan ilk dolu değeri
            s = df_slice[c]
            val = ""
            if isinstance(s, pd.Series):
                s2 = s.dropna()
                val = str(s2.iloc[0]) if len(s2) else ""
            m[c] = val
    st.write(
        f"**model_version:** {m.get('model_version','—')}  •  "
        f"**run_id:** {m.get('run_id','—')}  •  "
        f"**horizon_days:** {m.get('forecast_horizon_days','—')}  •  "
        f"**generated_at:** {m.get('forecast_generated_at','—')}"
    )
    if m.get("audit_tag"):
        st.caption(f"audit_tag: {m.get('audit_tag')}")

