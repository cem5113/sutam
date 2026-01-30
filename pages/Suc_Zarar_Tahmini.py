# -*- coding: utf-8 -*-
# pages/Suc_Zarar_Tahmini.py
# SUTAM — Suç & Zarar (HARM) | Operasyonel Karar Destek (Kolluk-dostu)
# - Veri: deploy/forecast_7d_ops_ready.(parquet|csv) veya data/
# - Harita: Risk / Zarar / Ops Öncelik
# - Hover: p, beklenen suç, beklenen zarar, top3, etken, bağlam, tek cümle aksiyon
# - Alt: Top liste + GEOID (7 gün) özet + dipnot (model bilgisi)

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
    _IMPORT_SRC_ERR = e
else:
    _IMPORT_SRC_ERR = None

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
LIKERT_OPS = {
    1: ("İzle",        [196, 226, 255]),
    2: ("Dikkat",      [148, 202, 255]),
    3: ("Öncelikli",   [241, 196, 15]),
    4: ("Çok Öncelik", [230, 126, 34]),
    5: ("Acil",        [192, 57, 43]),
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

# ---------------- UI CSS (kısa) ----------------
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

# ---------------- tiny helpers ----------------
def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def digits11(x) -> str:
    s = "".join(ch for ch in str(x) if ch.isdigit())
    return s.zfill(11) if s else ""

def pick_col(df, names):
    cols = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in cols:
            return cols[n.lower()]
    return None

def ffloat(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def fmt3(x):
    v = ffloat(x, np.nan)
    return "—" if not np.isfinite(v) else f"{v:.3f}"

def fmt_intish(x):
    v = ffloat(x, np.nan)
    return "—" if not np.isfinite(v) else f"{int(round(v))}"

def fmt_expected(x):
    v = ffloat(x, np.nan)
    if not np.isfinite(v):
        return "—"
    v = max(0.0, v)
    lo, hi = int(np.floor(v)), int(np.ceil(v))
    return f"~{lo}" if lo == hi else f"~{lo}–{hi}"

def fmt_harm(x):
    v = ffloat(x, np.nan)
    if not np.isfinite(v):
        return "—"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 100:
        return f"{v:.0f}"
    return f"{v:.1f}"

def coerce_bool(x) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in ("1", "true", "t", "yes", "y")

def risk_hint(level: str) -> str:
    s = str(level or "").strip().lower()
    if "critical" in s or "çok yüksek" in s or "very high" in s:
        return "Acil görünürlük + hedefli müdahale"
    if "high" in s or "yüksek" in s:
        return "Hedefli devriye + giriş/çıkış kontrolü"
    if "medium" in s or "orta" in s:
        return "Kısa tur döngüsü + caydırıcılık"
    return "Rutin devriye + gözlemsel teyit"

def flags_txt(r) -> str:
    out = []
    if coerce_bool(r.get("calls_flag")): out.append("☎️ Çağrı")
    if coerce_bool(r.get("neighbor_flag")): out.append("🧭 Komşu")
    if coerce_bool(r.get("poi_flag")): out.append("📍 POI")
    if coerce_bool(r.get("transit_flag")): out.append("🚇 Transit")
    if coerce_bool(r.get("weather_flag")): out.append("🌧️ Hava")
    if coerce_bool(r.get("time_flag")): out.append("🕒 Zaman")
    return " • ".join(out) if out else "—"

def driver_label(x: str) -> str:
    return DRIVER_LABEL.get(str(x or "").strip(), str(x or "—"))

def segmented(label, options, default):
    if hasattr(st, "segmented_control"):
        return st.segmented_control(label, options=options, default=default)
    idx = options.index(default) if default in options else 0
    return st.radio(label, options=options, index=idx, horizontal=True)

# ---------------- LOADERS ----------------
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

# ---------------- NORMALIZE ----------------
def normalize_ops(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    geoid_col = pick_col(d, ["geoid", "GEOID"])
    d["geoid"] = d[geoid_col].map(digits11) if geoid_col else ""

    date_col = pick_col(d, ["date", "dt", "datetime"])
    d["_dt"] = pd.to_datetime(d[date_col], errors="coerce") if date_col else pd.NaT
    d["date_norm"] = pd.to_datetime(d["_dt"]).dt.normalize()

    hr_col = pick_col(d, ["hour_range", "hour_bucket"])
    d["hour_range"] = d[hr_col].astype(str) if hr_col else "00-24"

    p_col = pick_col(d, ["p_event", "risk_prob", "risk_score"])
    d["p_event"] = pd.to_numeric(d[p_col], errors="coerce") if p_col else np.nan

    ex_col = pick_col(d, ["expected_count", "expected_crimes"])
    d["expected_count"] = pd.to_numeric(d[ex_col], errors="coerce") if ex_col else np.nan

    harm_col = pick_col(d, ["expected_harm", "harm_expected", "harm"])
    d["expected_harm"] = pd.to_numeric(d[harm_col], errors="coerce") if harm_col else np.nan

    ops_col = pick_col(d, ["ops_rank_score", "risk_score"])
    d["ops_rank_score"] = pd.to_numeric(d[ops_col], errors="coerce") if ops_col else d["expected_harm"]

    rl_col = pick_col(d, ["risk_level"])
    if rl_col:
        d["risk_level"] = d[rl_col].astype(str)
    else:
        rb_col = pick_col(d, ["risk_bin"])
        mp = {1: "Very Low", 2: "Low", 3: "Medium", 4: "High", 5: "Critical"}
        d["risk_level"] = pd.to_numeric(d[rb_col], errors="coerce").map(mp).fillna("Unknown") if rb_col else "Unknown"

    # top categories
    for i in (1, 2, 3):
        c = pick_col(d, [f"top{i}_category", f"top{i}_cat"])
        d[f"top{i}_category"] = d[c].astype(str).replace("nan", "").fillna("") if c else ""
        sh = pick_col(d, [f"top{i}_share"])
        d[f"top{i}_share"] = pd.to_numeric(d[sh], errors="coerce") if sh else np.nan

    for k in ["primary_driver", "secondary_driver", "driver_profile"]:
        col = pick_col(d, [k])
        d[k] = d[col].astype(str) if col else ""

    for k in ["weather_flag", "calls_flag", "neighbor_flag", "transit_flag", "poi_flag", "time_flag"]:
        col = pick_col(d, [k])
        d[k] = d[col].apply(coerce_bool) if col else False

    for k in ["ops_actions_short", "ops_actions", "ops_reasons"]:
        col = pick_col(d, [k])
        d[k] = d[col].astype(str).fillna("") if col else ""

    for k in ["model_version", "run_id", "audit_tag", "forecast_generated_at", "forecast_horizon_days"]:
        col = pick_col(d, [k])
        d[k] = d[col] if col else ""

    return d

# ---------------- LIKERT ----------------
def quantile_likert(series: pd.Series, n=5, neutral=3) -> pd.Series:
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

def mode_meta(df_slice: pd.DataFrame, mode: str):
    if mode == "Zarar":
        src, pal = "expected_harm", LIKERT_HARM
    elif mode == "Ops Öncelik":
        src, pal = ("ops_rank_score" if "ops_rank_score" in df_slice.columns else "expected_harm"), LIKERT_OPS
    else:
        src, pal = "p_event", LIKERT_RISK

    v = pd.to_numeric(df_slice.get(src, np.nan), errors="coerce")
    lik = quantile_likert(v, n=5, neutral=3).clip(1, 5)
    cuts = v.quantile([0.2, 0.4, 0.6, 0.8]).values.tolist() if v.notna().sum() >= 10 else [np.nan] * 4
    return lik, {"src": src, "pal": pal, "cuts": cuts}

# ---------------- GEOJSON ENRICH ----------------
def enrich_geojson(gj: dict, df_slice: pd.DataFrame, mode: str) -> dict:
    if not gj or df_slice.empty:
        return gj

    d = df_slice.copy()
    lik, meta = mode_meta(d, mode)
    pal = meta["pal"]
    d["_lik"] = lik
    d["_fill"] = d["_lik"].map(lambda k: pal[int(k)][1])

    d["_p_txt"] = d["p_event"].map(fmt3)
    d["_exp_txt"] = d["expected_count"].map(fmt_expected)
    d["_harm_txt"] = d["expected_harm"].map(fmt_harm)

    # tek cümle eylem
    act = d["ops_actions_short"].copy()
    act[act.astype(str).str.len() < 3] = d["ops_actions"]
    act[act.astype(str).str.len() < 3] = d["risk_level"].map(risk_hint)
    d["_action"] = act

    # top3
    def top3(r):
        arr = []
        for i in (1, 2, 3):
            c = str(r.get(f"top{i}_category") or "").strip()
            if c and c.lower() != "unknown":
                arr.append(c)
        return " • ".join(arr) if arr else "—"
    d["_top3"] = d.apply(top3, axis=1)

    d["_flags"] = d.apply(flags_txt, axis=1)
    d["_driver"] = d["primary_driver"].map(driver_label)
    d["_profile"] = d["driver_profile"].replace("", "—")

    # geoide tek satır
    d["_tie"] = pd.to_numeric(d.get("ops_rank_score", d.get("expected_harm", np.nan)), errors="coerce").fillna(-np.inf)
    d = (
        d.sort_values(["_lik", "_tie"], ascending=[False, False])
        .drop_duplicates("geoid", keep="first")
        .set_index("geoid")
    )

    feats = []
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

        key = digits11(raw)
        props["display_id"] = str(raw) if raw not in (None, "") else key
        props["mode_name"] = mode

        # default
        props.update({
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
            props["fill_color"] = r.get("_fill", DEFAULT_FILL)
            props["risk_level"] = str(r.get("risk_level", "") or "")
            props["p_event_txt"] = str(r.get("_p_txt", "—") or "—")
            props["expected_txt"] = str(r.get("_exp_txt", "—") or "—")
            props["harm_txt"] = str(r.get("_harm_txt", "—") or "—")
            props["top3"] = str(r.get("_top3", "—") or "—")
            props["driver_txt"] = str(r.get("_driver", "—") or "—")
            props["profile_txt"] = str(r.get("_profile", "—") or "—")
            props["flags_txt"] = str(r.get("_flags", "—") or "—")
            props["ops_action"] = str(r.get("_action", "—") or "—")

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

# ---------------- SLICE + KPI + TABLE ----------------
def slice_df(df, sel_date, sel_hr):
    return df[(df["date_norm"] == sel_date) & (df["hour_range"].astype(str) == str(sel_hr))].copy()

def kpis(df_slice):
    if df_slice.empty:
        return {"cells": 0, "mean_p": np.nan, "sum_expected": np.nan, "sum_harm": np.nan, "hi_impact": 0, "flags": {}}

    mean_p = pd.to_numeric(df_slice["p_event"], errors="coerce").mean()
    sum_expected = pd.to_numeric(df_slice["expected_count"], errors="coerce").fillna(0).sum()
    sum_harm = pd.to_numeric(df_slice["expected_harm"], errors="coerce").fillna(0).sum()

    # impact flag opsiyonel
    impact_col = pick_col(df_slice, ["impact_flag"])
    hi_impact = int(df_slice[impact_col].astype(str).str.contains("High|Critical", case=False, regex=True).sum()) if impact_col else 0

    flags = {k: int(df_slice[k].sum()) for k in ["calls_flag","neighbor_flag","poi_flag","transit_flag","weather_flag","time_flag"] if k in df_slice.columns}
    return {"cells": int(df_slice["geoid"].nunique()), "mean_p": mean_p, "sum_expected": float(sum_expected), "sum_harm": float(sum_harm), "hi_impact": hi_impact, "flags": flags}

def top_table(df_slice, n, rank_by):
    if df_slice.empty:
        return pd.DataFrame()

    d = df_slice.copy()
    if rank_by not in d.columns:
        rank_by = "ops_rank_score" if "ops_rank_score" in d.columns else "expected_harm"
    d["_rk"] = pd.to_numeric(d[rank_by], errors="coerce").fillna(-np.inf)

    def top3(r):
        arr = []
        for i in (1, 2, 3):
            c = str(r.get(f"top{i}_category") or "").strip()
            if c and c.lower() != "unknown":
                arr.append(c)
        return ", ".join(arr[:3]) if arr else "—"

    out = (
        d.sort_values("_rk", ascending=False)
         .head(int(n))
         .assign(
            Top3=lambda x: x.apply(top3, axis=1),
            AnaNeden=lambda x: x["primary_driver"].map(driver_label),
         )
    )

    cols = ["geoid","risk_level","p_event","expected_count","expected_harm","AnaNeden","driver_profile","ops_actions_short","Top3"]
    cols = [c for c in cols if c in out.columns]

    out = out[cols].rename(columns={
        "geoid": "Bölge (GEOID)",
        "risk_level": "Alarm",
        "p_event": "Olasılık",
        "expected_count": "Tahmini Olay",
        "expected_harm": "Tahmini Etki",
        "driver_profile": "Profil",
        "ops_actions_short": "Ne Yapmalı?",
        "Top3": "Hazırlıklı Ol (Top3)",
        "AnaNeden": "Ana Neden",
    }).copy()

    if "Olasılık" in out.columns: out["Olasılık"] = out["Olasılık"].map(fmt3)
    if "Tahmini Olay" in out.columns: out["Tahmini Olay"] = out["Tahmini Olay"].map(fmt_expected)
    if "Tahmini Etki" in out.columns: out["Tahmini Etki"] = out["Tahmini Etki"].map(fmt_harm)

    return out

# ---------------- GEOID (7 gün) ----------------
def geoid_week(df_all: pd.DataFrame, geoid: str) -> dict:
    g = df_all[df_all["geoid"] == geoid].copy()
    if g.empty:
        return {}

    g["_harm"] = pd.to_numeric(g["expected_harm"], errors="coerce").fillna(0.0)
    g["_ops"]  = pd.to_numeric(g["ops_rank_score"], errors="coerce").fillna(0.0)

    by_hr = (g.groupby("hour_range", dropna=False)[["_harm","_ops"]]
               .mean()
               .sort_values(["_harm","_ops"], ascending=False))

    top_hours = by_hr.head(3).index.astype(str).tolist()

    c = g.get("top1_category", pd.Series([], dtype=str)).astype(str).replace("nan", "")
    c = c[c.str.len() > 0]
    top_cats = c.value_counts().head(5).index.tolist()

    pdv = g["primary_driver"].astype(str)
    pdv = pdv[pdv.str.len() > 0]
    top_driver = pdv.value_counts().head(1).index.tolist()
    top_driver = top_driver[0] if top_driver else "risk_core"

    flags = {k: int(g[k].sum()) for k in ["calls_flag","neighbor_flag","poi_flag","transit_flag","weather_flag","time_flag"] if k in g.columns}

    summary = f"En kritik saatler: **{', '.join(top_hours) if top_hours else '—'}**. "
    if top_cats:
        summary += f"Öne çıkan suçlar: **{', '.join(top_cats[:3])}**. "
    summary += f"Ana etken eğilimi: **{driver_label(top_driver)}**."

    by_hr_df = by_hr.reset_index().rename(columns={"hour_range":"Saat Dilimi","_harm":"Ort. Etki","_ops":"Ort. Ops"})
    return {"summary": summary, "by_hr": by_hr_df, "top_cats": top_cats, "flags": flags}

# ---------------- MAIN ----------------
def render_suc_zarar_tahmini():
    apply_css()

    st.markdown("# 🧭 Suç & Zarar Etkisi — Operasyonel Karar Destek")
    st.caption("Amaç: **Ne zaman, nerede, neye dikkat etmeli ve ne yapmalı?** (Karar desteği; saha bilgisiyle birlikte yorumlanır.)")

    if _IMPORT_SRC_ERR is not None:
        st.error("`src.io_data` import edilemedi. `src/` klasörünü ve yolları kontrol edin.")
        st.code(repr(_IMPORT_SRC_ERR))
        return

    raw = load_ops_ready()
    if raw is None or raw.empty:
        st.error("Ops-ready veri bulunamadı/boş.\n" + "\n".join([f"- {p}" for p in OPS_CANDIDATES[:4]]))
        return

    df = normalize_ops(raw)
    st.caption(f"✅ Ops-ready: {df.shape[0]:,} satır • {df.shape[1]} kolon • GEOID: {df['geoid'].nunique():,}")

    gj = load_geojson()
    if not gj:
        st.error(f"GeoJSON bulunamadı: `{GEOJSON_PATH}`")
        return

    # ---------- state (reset olmasın) ----------
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

    hr_labels = sorted(df["hour_range"].dropna().astype(str).unique().tolist())
    default_hr = hr_labels[0] if hr_labels else "00-24"

    # ---------- controls ----------
    c1, c2, c3, c4 = st.columns([1.25, 1.0, 1.2, 1.0])
    with c1:
        sel_date = st.selectbox("📅 Tarih", options=dates,
                                index=dates.index(default_date) if default_date in dates else 0,
                                format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d"))
    with c2:
        sel_hr = st.selectbox("⏰ Saat dilimi", options=hr_labels,
                              index=hr_labels.index(default_hr) if default_hr in hr_labels else 0)
    with c3:
        mode = segmented("🗺️ Harita modu", ["Risk", "Zarar", "Ops Öncelik"], st.session_state.sz_mode)
        st.session_state.sz_mode = mode
    with c4:
        topn = st.selectbox("📌 Top hücre", options=[10, 15, 20, 30, 50],
                            index=[10,15,20,30,50].index(st.session_state.sz_topn) if st.session_state.sz_topn in [10,15,20,30,50] else 1)
        st.session_state.sz_topn = int(topn)

    df_slice = slice_df(df, sel_date, sel_hr)
    if df_slice.empty:
        st.warning("Seçili tarih/saat dilimi için kayıt yok.")
        return

    # ---------- KPI ----------
    k = kpis(df_slice)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"<div class='sutam-card'><h3>Hücre Sayısı</h3><div class='sutam-kpi'><div class='v'>{k['cells']}</div><div class='t'>izlenecek bölge</div></div></div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div class='sutam-card'><h3>Ortalama Olasılık</h3><div class='sutam-kpi'><div class='v'>{fmt3(k['mean_p'])}</div><div class='t'>şehir geneli alarm</div></div></div>", unsafe_allow_html=True)
    with k3:
        st.markdown(f"<div class='sutam-card'><h3>Beklenen Suç</h3><div class='sutam-kpi'><div class='v'>{fmt_intish(k['sum_expected'])}</div><div class='t'>bu dilimde olay</div></div></div>", unsafe_allow_html=True)
    with k4:
        st.markdown(f"<div class='sutam-card'><h3>Beklenen Zarar</h3><div class='sutam-kpi'><div class='v'>{fmt_harm(k['sum_harm'])}</div><div class='t'>toplam etki • kritik: {k['hi_impact']}</div></div></div>", unsafe_allow_html=True)

    # ---------- sidebar (kısa) ----------
    st.sidebar.markdown("### 🎯 Vardiya Özeti")
    st.sidebar.caption(f"SF saati: **{now_sf:%Y-%m-%d %H:%M}**")
    st.sidebar.write(f"**Tarih:** {pd.Timestamp(sel_date).strftime('%Y-%m-%d')}")
    st.sidebar.write(f"**Dilim:** {sel_hr}")
    st.sidebar.write(f"**Mod:** {mode}")

    if k.get("flags"):
        st.sidebar.markdown("### 🚩 Bağlam (adet)")
        f = k["flags"]
        st.sidebar.write(
            f"☎️ {f.get('calls_flag',0)} • 🧭 {f.get('neighbor_flag',0)} • 📍 {f.get('poi_flag',0)} • "
            f"🚇 {f.get('transit_flag',0)} • 🌧️ {f.get('weather_flag',0)} • 🕒 {f.get('time_flag',0)}"
        )

    st.divider()

    # ---------- map (aynı) ----------
    gj2 = enrich_geojson(gj, df_slice, mode)
    draw_map(gj2)
    st.caption("İpucu: Hücre üzerine gel (hover) → detay. Harita modu: Risk / Zarar / Ops Öncelik.")

    st.divider()

    # ---------- table ----------
    st.subheader("📌 Operasyon Öncelik Listesi")
    st.caption("Seçili tarih ve saat diliminde **öncelikli müdahale** gerektiren bölgeler.")
    rank_by = "expected_harm" if mode in ("Zarar", "Ops Öncelik") else "p_event"
    tbl = top_table(df_slice, n=topn, rank_by=rank_by)

    if tbl.empty:
        st.info("Tablo üretilemedi.")
    else:
        st.dataframe(tbl, use_container_width=True, hide_index=True)

        cexp1, cexp2 = st.columns([1, 1])
        with cexp1:
            csv1 = df_slice.copy()
            csv1["flags_txt"] = csv1.apply(flags_txt, axis=1)
            csv1["primary_driver_label"] = csv1["primary_driver"].map(driver_label)
            csv1["secondary_driver_label"] = csv1["secondary_driver"].map(driver_label)
            st.download_button("⬇️ Bu dilimin tam listesi (CSV)", data=csv1.to_csv(index=False).encode("utf-8-sig"),
                               file_name="ops_slice.csv", mime="text/csv")
        with cexp2:
            st.download_button("⬇️ Öncelikli bölgeler (Top liste CSV)", data=tbl.to_csv(index=False).encode("utf-8-sig"),
                               file_name="ops_top_list.csv", mime="text/csv")

    st.divider()

    # ---------- GEOID panel ----------
    st.subheader("🔎 Bölge Detayı — 7 Günlük Özet")
    left, right = st.columns([1.0, 2.0])

    with left:
        raw_geoid = st.text_input("GEOID (11 hane)", value="", placeholder="06075030101")
        sel_geoid = digits11(raw_geoid) if raw_geoid else ""
        st.caption("GEOID gir → **kritik saatler** + **tek cümle öneri**")

    with right:
        if not sel_geoid:
            st.info("Bölge kodunu girince burada özet görünecek.")
        else:
            prof = geoid_week(df, sel_geoid)
            if not prof:
                st.warning("Bu GEOID için veride kayıt yok.")
            else:
                st.markdown(prof["summary"])

                dhr = prof.get("by_hr")
                if isinstance(dhr, pd.DataFrame) and not dhr.empty:
                    dhr2 = dhr.copy()
                    dhr2["Ort. Etki"] = dhr2["Ort. Etki"].map(fmt_harm)
                    dhr2["Ort. Ops"] = dhr2["Ort. Ops"].map(fmt3)
                    st.dataframe(dhr2.head(10), use_container_width=True, hide_index=True)

                top_cats = prof.get("top_cats", [])
                if top_cats:
                    st.markdown("**Öne çıkan suçlar:** " + " • ".join([f"`{c}`" for c in top_cats[:5]]))

                f = prof.get("flags", {})
                if f:
                    st.markdown(
                        f"**Bağlam (7 gün sayımı):** ☎️ {f.get('calls_flag',0)} • 🧭 {f.get('neighbor_flag',0)} • 📍 {f.get('poi_flag',0)} • "
                        f"🚇 {f.get('transit_flag',0)} • 🌧️ {f.get('weather_flag',0)} • 🕒 {f.get('time_flag',0)}"
                    )

                # seçili dilim için tek cümle
                cur = df_slice[df_slice["geoid"] == sel_geoid].copy()
                if not cur.empty:
                    r = cur.sort_values(["expected_harm", "ops_rank_score"], ascending=False).head(1).iloc[0]
                    action = str(r.get("ops_actions_short") or "") or str(r.get("ops_actions") or "")
                    if not action:
                        action = risk_hint(str(r.get("risk_level")))
                    st.markdown("### 🧾 Bu dilim için tek cümle öneri")
                    st.success(action)

                    why = str(r.get("ops_reasons") or "")
                    if not why:
                        why = f"{driver_label(r.get('primary_driver'))} • {flags_txt(r)}"
                    st.markdown("### 🧠 Neden")
                    st.write(why)

    st.divider()

    # ---------- footer dipnot ----------
    with st.expander("ℹ️ Model & çalıştırma bilgisi (dipnot)", expanded=False):
        def first_nonempty(col):
            if col not in df_slice.columns:
                return "—"
            s = df_slice[col]
            s2 = s.dropna()
            return str(s2.iloc[0]) if len(s2) else "—"

        st.write(
            f"**model_version:** {first_nonempty('model_version')}  •  "
            f"**run_id:** {first_nonempty('run_id')}  •  "
            f"**horizon_days:** {first_nonempty('forecast_horizon_days')}  •  "
            f"**generated_at:** {first_nonempty('forecast_generated_at')}"
        )
        tag = first_nonempty("audit_tag")
        if tag != "—":
            st.caption(f"audit_tag: {tag}")
