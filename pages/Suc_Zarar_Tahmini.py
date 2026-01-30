# pages/Suc_Zarar_Tahmini.py
# SUTAM — Suç & Zarar Tahmini (Saha odaklı, basit)
# - Sol: tarih + saat dilimi seçimi (SF saatine göre)
# - Üst: Sekmeler (Suç / Zarar)
# - Harita: seçilen metriğe göre renklendirme + hover tooltip
# - Alt: Top riskli GEOID listesi + saha notları (kolon varsa göster)

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

# İsteğe bağlı destek dosyaları (varsa saha notlarını güçlendirir)
GEOID_PROFILE_CANDIDATES = [
    f"{DATA_DIR}/geoid_profile.parquet",
    "deploy/geoid_profile.parquet",
    "data/geoid_profile.parquet",
]
GEOID_STATS_CANDIDATES = [
    f"{DATA_DIR}/geoid_stats_5y.parquet",
    "deploy/geoid_stats_5y.parquet",
    "data/geoid_stats_5y.parquet",
]
CITY_BASELINE_CANDIDATES = [
    f"{DATA_DIR}/sf_city_baseline_5y.parquet",
    "deploy/sf_city_baseline_5y.parquet",
    "data/sf_city_baseline_5y.parquet",
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
# UI (tooltip compact)
# =============================================================================
def _apply_tooltip_css():
    st.markdown(
        """
        <style>
          .deckgl-tooltip{
            max-width: 360px !important;
            max-height: 320px !important;
            overflow: auto !important;
            padding: 10px 12px !important;
            line-height: 1.25 !important;
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
            fc = prepare_forecast(fc, gp=None)
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
def load_optional_table(candidates: list[str]) -> pd.DataFrame:
    p = _first_existing(candidates)
    if not p or load_parquet_or_csv is None:
        return pd.DataFrame()
    try:
        df = load_parquet_or_csv(p)
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# =============================================================================
# METRIC SELECTION (crime vs harm)
# =============================================================================
def _get_metric_cols(df: pd.DataFrame) -> dict:
    """
    Elimizde ne varsa ona göre:
      - Crime: p_event, expected_count
      - Harm: expected_harm (tercih), harm_expected, expected_damage,
              yoksa expected_count * harm_index (varsa)
    """
    p_col  = _pick_col(df, ["p_event", "risk_prob", "prob_event"])
    exp_col = _pick_col(df, ["expected_count", "expected_crimes", "mu", "lambda"])

    harm_col = (
        _pick_col(df, ["expected_harm", "harm_expected", "expected_damage", "expected_loss"])
    )
    harm_index_col = _pick_col(df, ["harm_index", "harm_multiplier", "harm_weight", "severity_index"])

    return {
        "p_col": p_col,
        "exp_col": exp_col,
        "harm_col": harm_col,
        "harm_index_col": harm_index_col,
    }

def _compute_metric_series(df: pd.DataFrame, mode: str) -> tuple[pd.Series, str]:
    """
    mode: "crime" or "harm"
    returns: (metric_series, metric_label)
    """
    cols = _get_metric_cols(df)
    p_col = cols["p_col"]
    exp_col = cols["exp_col"]
    harm_col = cols["harm_col"]
    harm_index_col = cols["harm_index_col"]

    if mode == "crime":
        # Öncelik: expected_count (saha için daha anlaşılır) yoksa p_event
        if exp_col:
            s = pd.to_numeric(df[exp_col], errors="coerce")
            return s, f"Beklenen Suç (≈ {exp_col})"
        if p_col:
            s = pd.to_numeric(df[p_col], errors="coerce")
            return s, f"Suç Olasılığı (p ≈ {p_col})"
        return pd.Series([np.nan] * len(df), index=df.index), "Suç metriği bulunamadı"

    # harm
    if harm_col:
        s = pd.to_numeric(df[harm_col], errors="coerce")
        return s, f"Beklenen Zarar (≈ {harm_col})"
    if exp_col and harm_index_col:
        exp = pd.to_numeric(df[exp_col], errors="coerce")
        hx  = pd.to_numeric(df[harm_index_col], errors="coerce")
        s = exp * hx
        return s, f"Beklenen Zarar (≈ {exp_col} × {harm_index_col})"
    if exp_col:
        s = pd.to_numeric(df[exp_col], errors="coerce")
        return s, f"Zarar için fallback: Beklenen Suç (≈ {exp_col})"
    return pd.Series([np.nan] * len(df), index=df.index), "Zarar metriği bulunamadı"

# =============================================================================
# RISK -> LIKERT (quintile on selected metric)
# =============================================================================
def _compute_likert_quintiles(metric: pd.Series) -> tuple[pd.Series, list]:
    v = pd.to_numeric(metric, errors="coerce")
    if v.notna().sum() < 10:
        lik = pd.Series([3] * len(v), index=v.index)
        return lik, [np.nan, np.nan, np.nan, np.nan]
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
    return lik, cuts

# =============================================================================
# SAHA NOTU (kolon varsa üret, yoksa sessizce geç)
# =============================================================================
def _build_saha_notlari(row: pd.Series, hr_label: str) -> list[str]:
    notes: list[str] = []

    # Saat bilgisi (etiketten türet)
    rg = _parse_range(str(hr_label)) or (None, None)
    hr_mid = None
    if rg[0] is not None and rg[1] is not None:
        hr_mid = int((rg[0] + rg[1]) / 2)

    # Bar / gece ekonomisi
    bar_col = _pick_col(pd.DataFrame([row]), ["bar_count", "bars", "poi_bar_count"])
    if bar_col:
        bar_v = _safe_float(row.get(bar_col), np.nan)
        if np.isfinite(bar_v) and bar_v >= 5 and (hr_mid is None or hr_mid >= 18):
            notes.append("Bar yoğunluğu yüksek + akşam saatleri: asayiş/alkol kaynaklı olaylara dikkat.")

    # Okul / çıkış saatleri
    school_col = _pick_col(pd.DataFrame([row]), ["school_count", "schools", "poi_school_count"])
    if school_col:
        sc = _safe_float(row.get(school_col), np.nan)
        if np.isfinite(sc) and sc >= 1 and (hr_mid is None or 14 <= hr_mid <= 18):
            notes.append("Okul/öğrenci yoğunluğu: çıkış saatlerinde yaya hareketi ve çevre kontrolü artırılabilir.")

    # Komşuluk etkisi
    neigh_col = _pick_col(pd.DataFrame([row]), ["neighbor_crime_7d", "neighbor_risk_7d", "adjacent_crime_7d"])
    if neigh_col:
        nv = _safe_float(row.get(neigh_col), np.nan)
        if np.isfinite(nv) and nv > 0:
            notes.append("Komşu hücrelerde yakın dönem yoğunluk: sınır bölgelerde kısa kontrollü tur faydalı olabilir.")

    # Transit yakınlığı (genel)
    tr_col = _pick_col(pd.DataFrame([row]), ["train_stop_count", "station_count"])
    bus_col = _pick_col(pd.DataFrame([row]), ["bus_stop_count", "bus_count"])
    if tr_col:
        tv = _safe_float(row.get(tr_col), np.nan)
        if np.isfinite(tv) and tv >= 3:
            notes.append("İstasyon/hat yoğunluğu: giriş-çıkış akışına bağlı kapkaç/hırsızlık riski artabilir.")
    if bus_col:
        bv = _safe_float(row.get(bus_col), np.nan)
        if np.isfinite(bv) and bv >= 8:
            notes.append("Otobüs durağı yoğunluğu: kalabalık noktalar kısa süreli devriye için uygun olabilir.")

    return notes[:3]  # sahaya basit: max 3 not

# =============================================================================
# GEOJSON ENRICH (per selected metric)
# =============================================================================
def enrich_geojson(gj: dict, df_hr: pd.DataFrame, mode: str, hr_label: str,
                  geoid_profile: pd.DataFrame, geoid_stats: pd.DataFrame) -> dict:
    if not gj or df_hr.empty:
        return gj

    df = df_hr.copy()

    geoid_col = _pick_col(df, ["GEOID", "geoid"])
    df["geoid"] = df[geoid_col].map(_digits11) if geoid_col else ""

    # metrics
    cols = _get_metric_cols(df)
    p_col = cols["p_col"]
    exp_col = cols["exp_col"]

    metric, metric_label = _compute_metric_series(df, mode)
    df["_metric"] = metric

    # Basit metinler
    df["p_event_txt"] = pd.to_numeric(df[p_col], errors="coerce").map(_fmt3) if p_col else "—"
    df["expected_txt"] = pd.to_numeric(df[exp_col], errors="coerce").map(_fmt_expected) if exp_col else "—"
    df["metric_txt"] = pd.to_numeric(df["_metric"], errors="coerce").map(_fmt_expected)

    # Top categories (varsa)
    for i in (1,2,3):
        c = _pick_col(df, [f"top{i}_category", f"top{i}_cat", f"cat{i}"])
        df[f"top{i}_category"] = df[c].astype(str).replace("nan","").fillna("") if c else ""

    # Likert
    lik, _cuts = _compute_likert_quintiles(df["_metric"])
    df["risk_likert"] = lik.clip(1,5)
    df["likert_label"] = df["risk_likert"].map(lambda k: LIKERT[int(k)][0])
    df["fill_color"] = df["risk_likert"].map(lambda k: LIKERT[int(k)][1])

    # Profil/stat join (varsa)
    df_join = df
    if not geoid_profile.empty:
        gcol = _pick_col(geoid_profile, ["geoid", "GEOID"])
        if gcol:
            gp = geoid_profile.copy()
            gp["geoid"] = gp[gcol].map(_digits11)
            df_join = df_join.merge(gp.drop(columns=[gcol], errors="ignore"), on="geoid", how="left")

    if not geoid_stats.empty:
        gcol = _pick_col(geoid_stats, ["geoid", "GEOID"])
        if gcol:
            gs = geoid_stats.copy()
            gs["geoid"] = gs[gcol].map(_digits11)
            # mümkünse hour_range varsa bağla; yoksa sadece geoid
            hr_col = _pick_col(gs, ["hour_range", "hour_bucket"])
            if hr_col:
                df_join = df_join.merge(
                    gs.drop(columns=[gcol], errors="ignore"),
                    left_on=["geoid"],
                    right_on=["geoid"],
                    how="left",
                    suffixes=("","_stats")
                )
            else:
                df_join = df_join.merge(
                    gs.drop(columns=[gcol], errors="ignore"),
                    on="geoid",
                    how="left",
                    suffixes=("","_stats")
                )

    # Saha notu üret (row bazında)
    saha_notes = []
    for _, r in df_join.iterrows():
        notes = _build_saha_notlari(r, hr_label)
        saha_notes.append(" • " + "\n • ".join(notes) if notes else "")
    df_join["saha_note"] = saha_notes

    # Aynı GEOID birden fazla satırsa: metric yüksek olan kalsın
    df_join["_metric_num"] = pd.to_numeric(df_join["_metric"], errors="coerce").fillna(-1.0)
    df_join = (
        df_join.sort_values(["risk_likert","_metric_num"], ascending=[False, False])
               .drop_duplicates("geoid", keep="first")
    )
    dmap = df_join.set_index("geoid")

    feats_out = []
    for feat in gj.get("features", []):
        props = dict(feat.get("properties") or {})

        raw = None
        for k in ("geoid","GEOID","cell_id","id","geoid11","geoid_11","display_id"):
            if k in props:
                raw = props[k]; break
        if raw is None:
            for k,v in props.items():
                if "geoid" in str(k).lower():
                    raw = v; break

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

    return {**gj, "features": feats_out}

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

    # Tooltip: sahaya basit
    if mode == "crime":
        metric_title = "Suç metriği"
    else:
        metric_title = "Zarar metriği"

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
# PAGE ENTRYPOINT
# =============================================================================
def render_suc_zarar_tahmini():
    _apply_tooltip_css()
    st.markdown("# 🎯 Suç & Zarar Tahmini (Seçilebilir)")

    st.caption(
        "Saha odaklı basit ekran: tarih + saat seç → haritada riskli hücreleri gör. "
        "Bu çıktı karar destek amaçlıdır; saha gözlemi ve amir değerlendirmesi ile birlikte yorumlanmalıdır."
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
    hr_col = _pick_col(fc, ["hour_range", "hour_bucket"])
    geoid_col = _pick_col(fc, ["geoid", "GEOID"])
