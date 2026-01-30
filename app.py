# pages/Suc_Zarar_Tahmini.py
# SUTAM — Suç & Zarar Tahmini (Saha odaklı, basit)
# - Sol: tarih + saat dilimi seçimi (SF saatine göre)
# - Üst: Sekmeler (Suç / Zarar)
# - Harita: seçilen metriğe göre renklendirme + hover tooltip
# - Alt: Top riskli GEOID listesi + saha notları (varsa)

from __future__ import annotations

import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# --- Güvenli import ---
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

# Ops brief (varsa tabloya zenginlik katar)
OPS_TOPK_CANDIDATES = [
    f"{DATA_DIR}/ops_brief_topk.csv",
    "deploy/ops_brief_topk.csv",
    "data/ops_brief_topk.csv",
]
OPS_DAILY_CANDIDATES = [
    f"{DATA_DIR}/ops_brief_geoid_daily.csv",
    "deploy/ops_brief_geoid_daily.csv",
    "data/ops_brief_geoid_daily.csv",
]

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
# UI
# =============================================================================
def _apply_tooltip_css():
    st.markdown(
        """
        <style>
          .deckgl-tooltip{
            max-width: 380px !important;
            max-height: 320px !important;
            overflow: auto !important;
            padding: 10px 12px !important;
            line-height: 1.22 !important;
            border-radius: 12px !important;
            box-shadow: 0 10px 30px rgba(0,0,0,.25) !important;
            transform: translate(10px, 10px) !important;
          }
          .deckgl-tooltip .tt-sep { margin: 8px 0; opacity: .25; }
          .deckgl-tooltip .tt-li { margin: 2px 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# HELPERS
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

def _fmt_expected(x) -> str:
    v = _safe_float(x, np.nan)
    if not np.isfinite(v):
        return "—"
    v = max(0.0, v)
    lo = int(np.floor(v))
    hi = int(np.ceil(v))
    return f"~{lo}" if lo == hi else f"~{lo}–{hi}"

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

    # normal
    for lab, s, e in parsed:
        if s <= h < e:
            return lab

    # wrap-around "21-3"
    for lab, s, e in parsed:
        if s > e and (h >= s or h < e):
            return lab

    return parsed[0][0] if parsed else None

# =============================================================================
# LOADERS
# =============================================================================
@st.cache_data(show_spinner=False)
def load_forecast() -> pd.DataFrame:
    p = _first_existing(FC_CANDIDATES)
    if not p or load_parquet_or_csv is None:
        return pd.DataFrame()

    fc = load_parquet_or_csv(p)
    if fc is None or getattr(fc, "empty", True):
        return pd.DataFrame()

    if prepare_forecast is not None:
        try:
            fc = prepare_forecast(fc, gp=None)  # hız için
        except TypeError:
            pass
        except Exception:
            pass

    return fc

@st.cache_data(show_spinner=False)
def load_geojson() -> dict:
    if os.path.exists(GEOJSON_PATH):
        with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data(show_spinner=False)
def load_ops_csv(candidates: list[str]) -> pd.DataFrame:
    p = _first_existing(candidates)
    if not p:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

# =============================================================================
# METRIC (crime vs harm)
# =============================================================================
def _get_metric_cols(df: pd.DataFrame) -> dict:
    p_col  = _pick_col(df, ["p_event", "risk_prob", "prob_event"])
    exp_col = _pick_col(df, ["expected_count", "expected_crimes", "mu", "lambda"])
    harm_col = _pick_col(df, ["expected_harm", "harm_expected", "expected_damage", "expected_loss"])
    harm_index_col = _pick_col(df, ["harm_index", "harm_multiplier", "harm_weight", "severity_index"])
    return {
        "p_col": p_col,
        "exp_col": exp_col,
        "harm_col": harm_col,
        "harm_index_col": harm_index_col,
    }

def _compute_metric_series(df: pd.DataFrame, mode: str) -> tuple[pd.Series, str]:
    cols = _get_metric_cols(df)
    p_col = cols["p_col"]
    exp_col = cols["exp_col"]
    harm_col = cols["harm_col"]
    harm_index_col = cols["harm_index_col"]

    if mode == "crime":
        # saha için en anlaşılır: beklenen olay sayısı
        if exp_col:
            return pd.to_numeric(df[exp_col], errors="coerce"), "Beklenen suç"
        if p_col:
            return pd.to_numeric(df[p_col], errors="coerce"), "Suç olasılığı (p)"
        return pd.Series([np.nan]*len(df), index=df.index), "Suç metriği yok"

    # harm
    if harm_col:
        return pd.to_numeric(df[harm_col], errors="coerce"), "Beklenen zarar"
    if exp_col and harm_index_col:
        exp = pd.to_numeric(df[exp_col], errors="coerce")
        hx  = pd.to_numeric(df[harm_index_col], errors="coerce")
        return exp * hx, "Beklenen zarar (basit)"
    if exp_col:
        return pd.to_numeric(df[exp_col], errors="coerce"), "Zarar için fallback: Beklenen suç"
    return pd.Series([np.nan]*len(df), index=df.index), "Zarar metriği yok"

def _compute_likert_quintiles(metric: pd.Series) -> tuple[pd.Series, list[float]]:
    v = pd.to_numeric(metric, errors="coerce")
    if v.notna().sum() < 10:
        return pd.Series([3]*len(v), index=v.index), [np.nan]*4

    try:
        bins = pd.qcut(v.rank(method="first"), 5, labels=[1,2,3,4,5])
        lik = bins.astype(int)
    except Exception:
        qs = v.quantile([0.2,0.4,0.6,0.8]).values.tolist()
        q20,q40,q60,q80 = qs
        lik = pd.Series(3, index=v.index)
        lik[v <= q20] = 1
        lik[(v > q20) & (v <= q40)] = 2
        lik[(v > q40) & (v <= q60)] = 3
        lik[(v > q60) & (v <= q80)] = 4
        lik[v > q80] = 5

    cuts = v.quantile([0.2,0.4,0.6,0.8]).values.tolist()
    return lik.clip(1,5), cuts

# =============================================================================
# SAHA NOTU (basit, kolon varsa)
# =============================================================================
def _build_saha_notlari(row: pd.Series, hr_label: str) -> list[str]:
    notes: list[str] = []
    rg = _parse_range(str(hr_label)) or (None, None)
    hr_mid = None
    if rg[0] is not None and rg[1] is not None:
        hr_mid = int((rg[0] + rg[1]) / 2)

    # bar
    bar_col = None
    for c in ("bar_count","bars","poi_bar_count"):
        if c in row.index:
            bar_col = c; break
    if bar_col:
        bar_v = _safe_float(row.get(bar_col), np.nan)
        if np.isfinite(bar_v) and bar_v >= 5 and (hr_mid is None or hr_mid >= 18):
            notes.append("Bar yoğunluğu + akşam: asayiş/alkol kaynaklı olaylara dikkat.")

    # school
    school_col = None
    for c in ("school_count","schools","poi_school_count"):
        if c in row.index:
            school_col = c; break
    if school_col:
        sc = _safe_float(row.get(school_col), np.nan)
        if np.isfinite(sc) and sc >= 1 and (hr_mid is None or 14 <= hr_mid <= 18):
            notes.append("Okul çevresi: çıkış saatlerinde yaya hareketi artabilir.")

    # neighbor effect
    neigh_col = None
    for c in ("neighbor_crime_7d","neighbor_risk_7d","adjacent_crime_7d"):
        if c in row.index:
            neigh_col = c; break
    if neigh_col:
        nv = _safe_float(row.get(neigh_col), np.nan)
        if np.isfinite(nv) and nv > 0:
            notes.append("Komşu hücrelerde yakın dönem yoğunluk: sınır bölgelerde tur faydalı olabilir.")

    # transit
    if "train_stop_count" in row.index:
        tv = _safe_float(row.get("train_stop_count"), np.nan)
        if np.isfinite(tv) and tv >= 3:
            notes.append("İstasyon çevresi: giriş-çıkış akışında kapkaç/hırsızlık artabilir.")
    if "bus_stop_count" in row.index:
        bv = _safe_float(row.get("bus_stop_count"), np.nan)
        if np.isfinite(bv) and bv >= 8:
            notes.append("Durak yoğunluğu: kalabalık noktalarda kısa görünürlük etkili olabilir.")

    return notes[:3]

# =============================================================================
# GEOJSON ENRICH
# =============================================================================
def enrich_geojson(gj: dict, df_hr: pd.DataFrame, mode: str, hr_label: str) -> tuple[dict, pd.DataFrame, str]:
    if not gj or df_hr.empty:
        return gj, df_hr, ""

    df = df_hr.copy()

    geoid_col = _pick_col(df, ["geoid","GEOID"])
    if not geoid_col:
        return gj, df_hr, ""

    df["geoid"] = df[geoid_col].map(_digits11)

    cols = _get_metric_cols(df)
    p_col = cols["p_col"]
    exp_col = cols["exp_col"]

    metric, metric_label = _compute_metric_series(df, mode)
    df["_metric"] = metric

    df["p_event_txt"] = pd.to_numeric(df[p_col], errors="coerce").map(_fmt3) if p_col else "—"
    df["expected_txt"] = pd.to_numeric(df[exp_col], errors="coerce").map(_fmt_expected) if exp_col else "—"

    # metric text: zarar/expected için basit gösterim
    df["metric_txt"] = pd.to_numeric(df["_metric"], errors="coerce").map(_fmt_expected)

    for i in (1,2,3):
        c = _pick_col(df, [f"top{i}_category", f"top{i}_cat", f"cat{i}"])
        df[f"top{i}_category"] = df[c].astype(str).replace("nan","").fillna("") if c else ""

    lik, _cuts = _compute_likert_quintiles(df["_metric"])
    df["risk_likert"] = lik
    df["likert_label"] = df["risk_likert"].map(lambda k: LIKERT[int(k)][0])
    df["fill_color"] = df["risk_likert"].map(lambda k: LIKERT[int(k)][1])

    # saha note
    df["saha_note"] = df.apply(lambda r: " • " + "\n • ".join(_build_saha_notlari(r, hr_label)) if _build_saha_notlari(r, hr_label) else "", axis=1)

    # tek satır/geoid
    df["_metric_num"] = pd.to_numeric(df["_metric"], errors="coerce").fillna(-1.0)
    df = df.sort_values(["risk_likert","_metric_num"], ascending=[False, False]).drop_duplicates("geoid", keep="first")
    dmap = df.set_index("geoid")

    feats_out = []
    for feat in gj.get("features", []):
        props = dict(feat.get("properties") or {})

        raw = None
        for k in ("geoid","GEOID","cell_id","id","geoid11","geoid_11","display_id"):
            if k in props:
                raw = props[k]
                break
        if raw is None:
            for k,v in props.items():
                if "geoid" in str(k).lower():
                    raw = v
                    break

        key = _digits11(raw)
        props["display_id"] = str(raw) if raw not in (None,"") else key

        # defaults
        props["likert_label"] = ""
        props["p_event_txt"] = "—"
        props["expected_txt"] = "—"
        props["metric_label"] = metric_label
        props["metric_txt"] = "—"
        props["top1_category"] = ""
        props["top2_category"] = ""
        props["top3_category"] = ""
        props["saha_note"] = ""
        props["fill_color"] = DEFAULT_FILL

        if key and key in dmap.index:
            row = dmap.loc[key]
            props["likert_label"] = str(row.get("likert_label","") or "")
            props["p_event_txt"] = str(row.get("p_event_txt","—") or "—")
            props["expected_txt"] = str(row.get("expected_txt","—") or "—")
            props["metric_txt"] = str(row.get("metric_txt","—") or "—")
            props["top1_category"] = str(row.get("top1_category","") or "")
            props["top2_category"] = str(row.get("top2_category","") or "")
            props["top3_category"] = str(row.get("top3_category","") or "")
            props["saha_note"] = str(row.get("saha_note","") or "")
            props["fill_color"] = row.get("fill_color", DEFAULT_FILL)

        feats_out.append({**feat, "properties": props})

    return {**gj, "features": feats_out}, df, metric_label

# =============================================================================
# MAP
# =============================================================================
def draw_map(gj: dict, mode: str):
    layer = pdk.Layer(
        "GeoJsonLayer",
        gj,
        stroked=True,
        get_line_color=[80,80,80],
        line_width_min_pixels=0.5,
        filled=True,
        get_fill_color="properties.fill_color",
        pickable=True,
        opacity=0.65,
    )

    metric_title = "Suç göstergesi" if mode == "crime" else "Zarar göstergesi"

    tooltip = {
        "html": (
            "<div style='font-weight:800; font-size:14px;'>GEOID: {display_id}</div>"
            "<div><b>Risk:</b> {likert_label}</div>"
            "<div><b>p (olasılık):</b> {p_event_txt}</div>"
            "<div><b>Beklenen suç:</b> {expected_txt}</div>"
            "<div><b>" + metric_title + ":</b> {metric_txt}</div>"
            "<div class='tt-sep'></div>"
            "<div style='font-weight:800;'>En olası 3 olay</div>"
            "<div class='tt-li'>• {top1_category}</div>"
            "<div class='tt-li'>• {top2_category}</div>"
            "<div class='tt-li'>• {top3_category}</div>"
            "<div class='tt-sep'></div>"
            "<div style='font-weight:800;'>Saha Notu</div>"
            "<div style='white-space:pre-line;'>{saha_note}</div>"
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
# PAGE
# =============================================================================
def _render_page():
    _apply_tooltip_css()
    st.markdown("# 🎯 Suç & Zarar Tahmini")

    st.caption(
        "Basit saha ekranı: tarih + saat seç → haritada riskli hücreleri gör. "
        "Çıktılar karar destek amaçlıdır; saha gözlemi ve amir değerlendirmesi ile birlikte yorumlanmalıdır."
    )

    if _IMPORT_SRC_ERR is not None:
        st.error("`src.io_data` import edilemedi. `src/` klasörünü ve bağımlılıkları kontrol edin.")
        st.code(repr(_IMPORT_SRC_ERR))
        return

    fc = load_forecast()
    if fc.empty:
        st.error("Forecast verisi bulunamadı/boş. `deploy/full_fc.parquet` veya `data/forecast_7d.parquet` gerekli.")
        return

    date_col = _pick_col(fc, ["date"])
    hr_col = _pick_col(fc, ["hour_range","hour_bucket"])
    geoid_col = _pick_col(fc, ["geoid","GEOID"])
    if not date_col or not hr_col or not geoid_col:
        st.error("Forecast içinde gerekli kolonlar yok: `date`, `hour_range`, `geoid`.")
        st.caption(f"Bulunan kolonlar: {list(fc.columns)[:40]}")
        return

    fc = fc.copy()
    fc[date_col] = pd.to_datetime(fc[date_col], errors="coerce")
    fc["date_norm"] = fc[date_col].dt.normalize()
    fc["geoid_norm"] = fc[geoid_col].map(_digits11)
    fc["hr_norm"] = fc[hr_col].astype(str)

    # seçenekler
    now_sf = datetime.now(ZoneInfo(TARGET_TZ))
    today = pd.Timestamp(now_sf.date())

    dates = sorted(fc["date_norm"].dropna().unique())
    if not dates:
        st.error("Forecast içinde geçerli tarih yok.")
        return

    labels = sorted(fc["hr_norm"].dropna().unique().tolist())
    if not labels:
        st.error("Forecast içinde saat dilimi yok.")
        return

    default_date = today if today in dates else max([d for d in dates if d <= today], default=dates[0])
    default_hr = _hour_to_bucket(now_sf.hour, labels) or labels[0]

    # sidebar controls
    with st.sidebar:
        st.markdown("### 🎛️ Seçim")
        sel_date = st.date_input("Tarih (SF)", value=pd.Timestamp(default_date).date())
        sel_hr = st.selectbox("Saat dilimi (SF)", options=labels, index=labels.index(default_hr) if default_hr in labels else 0)

    sel_date_ts = pd.Timestamp(sel_date)

    df_hr = fc[(fc["date_norm"] == sel_date_ts) & (fc["hr_norm"] == str(sel_hr))].copy()
    if df_hr.empty:
        st.warning("Bu tarih/saat dilimi için kayıt bulunamadı.")
        return

    gj = load_geojson()
    if not gj:
        st.error(f"GeoJSON bulunamadı: `{GEOJSON_PATH}`")
        return

    # tabs
    tab_crime, tab_harm = st.tabs(["🟦 Suç", "🟥 Zarar"])

    def _render(mode: str):
        gj_enriched, df_one_per_geoid, metric_label = enrich_geojson(gj, df_hr, mode=mode, hr_label=str(sel_hr))
        draw_map(gj_enriched, mode=mode)

        st.caption(f"Seçim: **{sel_date_ts.date()}** • **{sel_hr}** • Gösterim: **{metric_label}**")

        # Top list (saha için basit)
        topn = 15
        df_list = df_one_per_geoid.copy()
        df_list["metric_num"] = pd.to_numeric(df_list["_metric"], errors="coerce").fillna(-1.0)
        df_list = df_list.sort_values(["risk_likert","metric_num"], ascending=[False, False]).head(topn)

        # Ops brief ile zenginleştir (varsa)
        ops_topk = load_ops_csv(OPS_TOPK_CANDIDATES)
        ops_daily = load_ops_csv(OPS_DAILY_CANDIDATES)

        if not ops_topk.empty:
            gcol = _pick_col(ops_topk, ["geoid","GEOID"])
            if gcol:
                ops_topk = ops_topk.copy()
                ops_topk["geoid_norm"] = ops_topk[gcol].map(_digits11)
                df_list = df_list.merge(
                    ops_topk.drop(columns=[gcol], errors="ignore"),
                    left_on="geoid",
                    right_on="geoid_norm",
                    how="left",
                    suffixes=("","_ops")
                )

        if not ops_daily.empty:
            gcol = _pick_col(ops_daily, ["geoid","GEOID"])
            if gcol:
                ops_daily = ops_daily.copy()
                ops_daily["geoid_norm"] = ops_daily[gcol].map(_digits11)
                # date varsa bağla
                dcol = _pick_col(ops_daily, ["date","day"])
                if dcol:
                    ops_daily[dcol] = pd.to_datetime(ops_daily[dcol], errors="coerce").dt.normalize()
                    df_list = df_list.merge(
                        ops_daily.drop(columns=[gcol], errors="ignore"),
                        left_on=["geoid"],
                        right_on=["geoid_norm"],
                        how="left",
                        suffixes=("","_daily")
                    )

        # Saha tablosu: en basit kolonlar
        show_cols = []
        for c in ["geoid","risk_likert","likert_label","p_event_txt","expected_txt","metric_txt","top1_category","top2_category","top3_category","saha_note"]:
            if c in df_list.columns:
                show_cols.append(c)

        st.subheader(f"📌 En riskli {len(df_list)} hücre (özet)")
        st.dataframe(df_list[show_cols], use_container_width=True, height=420)

    with tab_crime:
        _render("crime")

    with tab_harm:
        _render("harm")

# ✅ app.py’nin aradığı ENTRYPOINT:
def render_suc_zarar_tahmini():
    _render_page()
