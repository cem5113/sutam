# pages/Suc_Zarar_Tahmini.py
# SUTAM — Suç & Zarar Tahmini (Saha odaklı, basit)
# - Sol: tarih + saat dilimi seçimi (SF saatine göre)
# - Üst: Sekmeler (Suç / Zarar)
# - Harita: seçilen metriğe göre (quintile->Likert 1–5) renklendirme + hover tooltip
# - Alt: Top riskli GEOID listesi (ops_brief_topk varsa) + kısa saha notu
#
# Not: Parquet okumak için src.io_data.load_parquet_or_csv kullanır (pyarrow gerekir).
#      Eğer src import edilemezse sayfa kendini güvenli şekilde kapatır.

from __future__ import annotations

import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# --- Güvenli import (src modülleri yoksa sayfa tamamen çökmesin) ---
try:
    from src.io_data import load_parquet_or_csv, prepare_forecast  # gp yok (hız)
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

# ops brief dosyaları (varsa “Top-K listesi” için)
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

# GEOID profil (varsa saha notlarını güçlendirir) — kullanıcı bazen csv bazen parquet veriyor
GEOID_PROFILE_CANDIDATES = [
    f"{DATA_DIR}/geoid_profile.parquet",
    f"{DATA_DIR}/geoid_profile.csv",
    "deploy/geoid_profile.parquet",
    "deploy/geoid_profile.csv",
    "data/geoid_profile.parquet",
    "data/geoid_profile.csv",
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


def _fmt_big(x) -> str:
    """Zarar gibi daha büyük sayılar için (çok basit)"""
    v = _safe_float(x, np.nan)
    if not np.isfinite(v):
        return "—"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 100:
        return f"{v:,.1f}"
    return f"{v:,.2f}"


def _parse_range(tok: str):
    # "21-24" -> (21,24) end exclusive
    if not isinstance(tok, str) or "-" not in tok:
        return None
    a, b = tok.split("-", 1)
    try:
        s = int(a.strip())
        e = int(b.strip())
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


def _likert_advice_simple(k: int, mode: str) -> str:
    # Tek satır, sahaya uygun
    if mode == "harm":
        if k >= 5:
            return "Öneri: Yüksek zarar riski — görünür devriye + hızlı müdahale planı öne alınabilir."
        if k == 4:
            return "Öneri: Zarar artışı olası — ana akslar ve yoğun toplanma noktaları izlenebilir."
        if k == 3:
            return "Öneri: Orta — rutin dolaşım, caydırıcılık ve kısa kontrol turu."
        if k == 2:
            return "Öneri: Düşük — rutin devriye sürdürülür."
        return "Öneri: Çok düşük — temel görünürlük yeterli olabilir."
    # crime
    if k >= 5:
        return "Öneri: Kritik yoğunluk — görünür devriye ve kısa kontrollü tur artırılabilir."
    if k == 4:
        return "Öneri: Risk artışı olası — transit/ana arter çevresinde kısa tur planlanabilir."
    if k == 3:
        return "Öneri: Orta — rutin devriye + caydırıcılık odaklı dolaşım."
    if k == 2:
        return "Öneri: Düşük — rutin dolaşım sürdürülür."
    return "Öneri: Çok düşük — temel izleme yeterli olabilir."


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

    # prepare_forecast varsa normalize eder (gp yok -> hız)
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
def load_ops_topk() -> pd.DataFrame:
    p = _first_existing(OPS_TOPK_CANDIDATES)
    if not p:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_ops_daily() -> pd.DataFrame:
    p = _first_existing(OPS_DAILY_CANDIDATES)
    if not p:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_geoid_profile() -> pd.DataFrame:
    p = _first_existing(GEOID_PROFILE_CANDIDATES)
    if not p:
        return pd.DataFrame()
    try:
        if p.lower().endswith(".csv"):
            return pd.read_csv(p)
        if load_parquet_or_csv is not None:
            df = load_parquet_or_csv(p)
            return df if df is not None else pd.DataFrame()
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# =============================================================================
# METRIC LOGIC
# =============================================================================
def _get_metric_cols(df: pd.DataFrame) -> dict:
    p_col = _pick_col(df, ["p_event", "risk_prob", "prob_event"])
    exp_col = _pick_col(df, ["expected_count", "expected_crimes", "mu", "lambda"])
    harm_col = _pick_col(df, ["expected_harm", "harm_expected", "expected_damage", "expected_loss"])
    return {"p_col": p_col, "exp_col": exp_col, "harm_col": harm_col}


def _compute_metric(df: pd.DataFrame, mode: str) -> tuple[pd.Series, str]:
    cols = _get_metric_cols(df)
    p_col, exp_col, harm_col = cols["p_col"], cols["exp_col"], cols["harm_col"]

    if mode == "harm":
        if harm_col:
            return pd.to_numeric(df[harm_col], errors="coerce"), "Beklenen zarar"
        # fallback
        if exp_col:
            return pd.to_numeric(df[exp_col], errors="coerce"), "Zarar yok → Beklenen suç (fallback)"
        if p_col:
            return pd.to_numeric(df[p_col], errors="coerce"), "Zarar yok → Olasılık (fallback)"
        return pd.Series([np.nan] * len(df), index=df.index), "Zarar metriği yok"
    # crime
    if exp_col:
        return pd.to_numeric(df[exp_col], errors="coerce"), "Beklenen suç"
    if p_col:
        return pd.to_numeric(df[p_col], errors="coerce"), "Suç olasılığı"
    return pd.Series([np.nan] * len(df), index=df.index), "Suç metriği yok"


def _compute_likert_quintiles(metric: pd.Series) -> tuple[pd.Series, list[float]]:
    v = pd.to_numeric(metric, errors="coerce")
    if v.notna().sum() < 10:
        lik = pd.Series([3] * len(v), index=v.index)
        return lik, [np.nan, np.nan, np.nan, np.nan]
    try:
        bins = pd.qcut(v.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        lik = bins.astype(int)
    except Exception:
        qs = v.quantile([0.2, 0.4, 0.6, 0.8]).values.tolist()
        q20, q40, q60, q80 = qs
        lik = pd.Series(3, index=v.index)
        lik[v <= q20] = 1
        lik[(v > q20) & (v <= q40)] = 2
        lik[(v > q40) & (v <= q60)] = 3
        lik[(v > q60) & (v <= q80)] = 4
        lik[v > q80] = 5
    cuts = v.quantile([0.2, 0.4, 0.6, 0.8]).values.tolist()
    return lik, cuts


# =============================================================================
# GEOJSON ENRICH (mode: crime/harm)
# =============================================================================
def enrich_geojson(
    gj: dict,
    df_hr: pd.DataFrame,
    mode: str,
    hr_label: str,
    geoid_profile: pd.DataFrame,
) -> dict:
    if not gj or df_hr.empty:
        return gj

    df = df_hr.copy()

    # geoid normalize
    geoid_col = _pick_col(df, ["GEOID", "geoid"])
    df["geoid"] = df[geoid_col].map(_digits11) if geoid_col else ""

    # core cols
    cols = _get_metric_cols(df)
    p_col, exp_col, harm_col = cols["p_col"], cols["exp_col"], cols["harm_col"]

    # compute metric
    metric, metric_label = _compute_metric(df, mode)
    df["_metric"] = metric

    # texts
    df["p_event_txt"] = pd.to_numeric(df[p_col], errors="coerce").map(_fmt3) if p_col else "—"
    df["expected_txt"] = pd.to_numeric(df[exp_col], errors="coerce").map(_fmt_expected) if exp_col else "—"
    if mode == "harm" and harm_col:
        df["metric_txt"] = pd.to_numeric(df[harm_col], errors="coerce").map(_fmt_big)
    else:
        # metric suçsa veya fallback ise
        df["metric_txt"] = pd.to_numeric(df["_metric"], errors="coerce").map(_fmt_expected)

    # top categories
    for i in (1, 2, 3):
        c = _pick_col(df, [f"top{i}_category", f"top{i}_cat", f"cat{i}"])
        df[f"top{i}_category"] = df[c].astype(str).replace("nan", "").fillna("") if c else ""

    # optional profile join (bar/school/transit etc. kolonları varsa)
    if not geoid_profile.empty:
        gp = geoid_profile.copy()
        gp_geoid = _pick_col(gp, ["geoid", "GEOID"])
        if gp_geoid:
            gp["geoid"] = gp[gp_geoid].map(_digits11)
            gp = gp.drop(columns=[gp_geoid], errors="ignore")
            df = df.merge(gp, on="geoid", how="left")

    # likert from metric
    lik, _cuts = _compute_likert_quintiles(df["_metric"])
    df["risk_likert"] = lik.clip(1, 5)
    df["likert_label"] = df["risk_likert"].map(lambda k: LIKERT[int(k)][0])
    df["fill_color"] = df["risk_likert"].map(lambda k: LIKERT[int(k)][1])
    df["advice_txt"] = df["risk_likert"].map(lambda k: _likert_advice_simple(int(k), mode))

    # basit saha notu: profile içindeki hazır kolonlardan üret
    # geoid_profile.csv’de bar/school yok; ama neighbor_crime_7d, poi_total_count, is_near_police vb. var.
    def _saha_notes(row: pd.Series) -> str:
        notes: list[str] = []

        # yakın polis / kamu
        in_pol = row.get("is_near_police", None)
        if str(in_pol).lower() in ("1", "true", "yes"):
            notes.append("Polis noktası yakın: görünürlük ile caydırıcılık artırılabilir.")

        in_gov = row.get("is_near_government", None)
        if str(in_gov).lower() in ("1", "true", "yes"):
            notes.append("Kamu binası yakın: giriş/çıkış saatlerinde çevre kontrolü düşünülebilir.")

        # komşu yoğunluk
        n7 = _safe_float(row.get("neighbor_crime_7d", np.nan), np.nan)
        if np.isfinite(n7) and n7 > 0:
            notes.append("Komşu hücrelerde yakın dönem yoğunluk: sınır bölgelerde kısa tur faydalı olabilir.")

        # POI yoğunluğu
        poi = _safe_float(row.get("poi_total_count", np.nan), np.nan)
        if np.isfinite(poi) and poi >= 50:
            notes.append("Yoğun aktivite/POI: kalabalık noktalar kısa süreli devriye için uygun olabilir.")

        # transit
        bus = _safe_float(row.get("bus_stop_count", np.nan), np.nan)
        trn = _safe_float(row.get("train_stop_count", np.nan), np.nan)
        if np.isfinite(trn) and trn >= 3:
            notes.append("İstasyon/hat yoğunluğu: kapkaç/hırsızlık riski artabilir.")
        elif np.isfinite(bus) and bus >= 8:
            notes.append("Otobüs durağı yoğunluğu: kalabalık noktalarda gözlem artırılabilir.")

        # saat bağlamı (etiketten)
        rg = _parse_range(str(hr_label))
        if rg:
            s, e = rg
            if s >= 18 or e <= 6:
                notes.append("Akşam/gece dilimi: asayiş ve fırsat suçlarına karşı görünür devriye.")

        # fazla uzatma
        notes = notes[:3]
        if not notes:
            return ""
        return " • " + "\n • ".join(notes)

    df["saha_note"] = df.apply(_saha_notes, axis=1)

    # tek geoid birden fazla satırsa: metric yüksek olan kalsın
    df["_metric_num"] = pd.to_numeric(df["_metric"], errors="coerce").fillna(-1.0)
    df = (
        df.sort_values(["risk_likert", "_metric_num"], ascending=[False, False])
          .drop_duplicates("geoid", keep="first")
    )
    dmap = df.set_index("geoid")

    feats_out = []
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
        props["likert_label"] = ""
        props["p_event_txt"] = "—"
        props["expected_txt"] = "—"
        props["metric_label"] = metric_label
        props["metric_txt"] = "—"
        props["top1_category"] = ""
        props["top2_category"] = ""
        props["top3_category"] = ""
        props["advice_txt"] = ""
        props["saha_note"] = ""
        props["fill_color"] = DEFAULT_FILL

        if key and key in dmap.index:
            row = dmap.loc[key]
            props["likert_label"] = str(row.get("likert_label", "") or "")
            props["p_event_txt"] = str(row.get("p_event_txt", "—") or "—")
            props["expected_txt"] = str(row.get("expected_txt", "—") or "—")
            props["metric_txt"] = str(row.get("metric_txt", "—") or "—")
            props["top1_category"] = str(row.get("top1_category", "") or "")
            props["top2_category"] = str(row.get("top2_category", "") or "")
            props["top3_category"] = str(row.get("top3_category", "") or "")
            props["advice_txt"] = str(row.get("advice_txt", "") or "")
            props["saha_note"] = str(row.get("saha_note", "") or "")
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
        get_line_color=[80, 80, 80],
        line_width_min_pixels=0.5,
        filled=True,
        get_fill_color="properties.fill_color",
        pickable=True,
        opacity=0.65,
    )

    metric_title = "Beklenen zarar" if mode == "harm" else "Beklenen suç"

    tooltip = {
        "html": (
            "<div style='font-weight:800; font-size:14px;'>GEOID: {display_id}</div>"
            "<div><b>Risk:</b> {likert_label}</div>"
            "<div><b>p (olasılık):</b> {p_event_txt}</div>"
            "<div><b>Beklenen suç:</b> {expected_txt}</div>"
            f"<div><b>{metric_title}:</b> {{metric_txt}}</div>"
            "<div class='tt-sep'></div>"
            "<div style='font-weight:800;'>En olası 3 olay</div>"
            "<div class='tt-li'>• {top1_category}</div>"
            "<div class='tt-li'>• {top2_category}</div>"
            "<div class='tt-li'>• {top3_category}</div>"
            "<div class='tt-sep'></div>"
            "<div style='font-weight:800;'>Saha Notu</div>"
            "<div style='white-space:pre-line;'>{saha_note}</div>"
            "<div class='tt-sep'></div>"
            "<div style='white-space:pre-line;'>{advice_txt}</div>"
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
# TOP-K PANEL (ops_brief_topk)
# =============================================================================
def render_topk_panel(ops_topk: pd.DataFrame, sel_date: pd.Timestamp, hr_label: str, mode: str):
    if ops_topk.empty:
        return

    df = ops_topk.copy()
    dcol = _pick_col(df, ["date"])
    if dcol:
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce").dt.normalize()
    else:
        return

    hcol = _pick_col(df, ["hour_range", "hour_bucket"])
    if not hcol:
        return

    df = df[(df[dcol] == pd.Timestamp(sel_date).normalize()) & (df[hcol].astype(str) == str(hr_label))].copy()
    if df.empty:
        return

    # mode'a göre sıralama
    sort_col = "expected_harm" if (mode == "harm" and "expected_harm" in df.columns) else ("expected_count" if "expected_count" in df.columns else None)
    if sort_col:
        df = df.sort_values(sort_col, ascending=False)

    # sahaya basit kolonlar
    keep = []
    for c in ["rank", "GEOID", "p_event", "expected_count", "expected_harm", "crime_count_hist", "top1_category", "top2_category", "top3_category"]:
        if c in df.columns:
            keep.append(c)
    df_view = df[keep].head(15).copy()

    # format
    if "p_event" in df_view.columns:
        df_view["p_event"] = pd.to_numeric(df_view["p_event"], errors="coerce").map(_fmt3)
    if "expected_count" in df_view.columns:
        df_view["expected_count"] = pd.to_numeric(df_view["expected_count"], errors="coerce").map(_fmt_expected)
    if "expected_harm" in df_view.columns:
        df_view["expected_harm"] = pd.to_numeric(df_view["expected_harm"], errors="coerce").map(_fmt_big)

    st.markdown("### 🔝 Bu dilimde en riskli hücreler (Top 15)")
    st.dataframe(df_view, use_container_width=True, hide_index=True)


# =============================================================================
# PAGE ENTRYPOINT
# =============================================================================
def render_suc_zarar_tahmini():
    _apply_tooltip_css()

    st.markdown("# 🎯 Suç & Zarar Tahmini")
    st.caption(
        "Saha odaklı basit ekran: **tarih + saat seç → haritada riskli hücreleri gör**. "
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
    hr_col = _pick_col(fc, ["hour_range", "hour_bucket"])
    geoid_col = _pick_col(fc, ["geoid", "GEOID"])
    if not date_col or not hr_col or not geoid_col:
        st.error("Forecast içinde `date`, `hour_range` ve `geoid/GEOID` kolonları gerekli. `prepare_forecast` çıktısını kontrol edin.")
        return

    fc = fc.copy()
    fc[date_col] = pd.to_datetime(fc[date_col], errors="coerce")
    fc["date_norm"] = fc[date_col].dt.normalize()
    fc["geoid_norm"] = fc[geoid_col].map(_digits11)

    # şimdi SF saatine göre default seç
    now_sf = datetime.now(ZoneInfo(TARGET_TZ))
    today = pd.Timestamp(now_sf.date())

    dates = sorted(fc["date_norm"].dropna().unique())
    if not dates:
        st.error("Forecast içinde geçerli tarih bulunamadı.")
        return

    labels = sorted(fc[hr_col].dropna().astype(str).unique().tolist())
    if not labels:
        st.error("Forecast içinde saat dilimi bulunamadı (`hour_range`).")
        return

    # defaults
    def_date = today if today in dates else max([d for d in dates if d <= today], default=dates[0])
    def_hr = _hour_to_bucket(now_sf.hour, labels) or labels[0]

    # ===== Sidebar selectors =====
    with st.sidebar:
        st.markdown("## 🕒 Zaman Seçimi (SF)")
        st.caption(f"SF saati: **{now_sf:%Y-%m-%d %H:%M}**")
        sel_date = st.date_input("Tarih", value=pd.Timestamp(def_date).date(), min_value=pd.Timestamp(min(dates)).date(), max_value=pd.Timestamp(max(dates)).date())
        sel_date = pd.Timestamp(sel_date).normalize()
        hr_label = st.selectbox("Saat dilimi", options=labels, index=labels.index(def_hr) if def_hr in labels else 0)

        st.markdown("---")
        st.markdown("## 🎚️ Görünüm")
        # basit: kaç hücre vurgulansın (liste için)
        topn = st.slider("Listede kaç hücre gösterilsin?", min_value=10, max_value=50, value=15, step=5)

    st.caption(f"Tarih: **{pd.Timestamp(sel_date).date()}**  •  Dilim: **{hr_label}**")

    df_hr = fc[(fc["date_norm"] == sel_date) & (fc[hr_col].astype(str) == str(hr_label))].copy()
    if df_hr.empty:
        st.warning("Bu tarih/saat dilimi için kayıt bulunamadı.")
        return

    gj = load_geojson()
    if not gj:
        st.error(f"GeoJSON bulunamadı: `{GEOJSON_PATH}` (polygonlar gerekli).")
        return

    ops_topk = load_ops_topk()
    ops_daily = load_ops_daily()
    geoid_profile = load_geoid_profile()

    # ===== Tabs (Suç / Zarar) =====
    tab_crime, tab_harm = st.tabs(["🚓 Suç", "💥 Zarar"])

    # ----------------------
    # TAB: CRIME
    # ----------------------
    with tab_crime:
        mode = "crime"

        gj_enriched = enrich_geojson(
            gj=gj,
            df_hr=df_hr,
            mode=mode,
            hr_label=str(hr_label),
            geoid_profile=geoid_profile,
        )
        draw_map(gj_enriched, mode=mode)

        # Top-K panel (ops_brief_topk varsa) — crime
        render_topk_panel(ops_topk, sel_date, str(hr_label), mode=mode)

        # Basit özet kartı (ops_daily varsa hızlı bilgi)
        if not ops_daily.empty:
            dcol = _pick_col(ops_daily, ["date"])
            if dcol:
                od = ops_daily.copy()
                od[dcol] = pd.to_datetime(od[dcol], errors="coerce").dt.normalize()
                od = od[od[dcol] == sel_date]
                if not od.empty and "expected_count" in od.columns:
                    v = pd.to_numeric(od["expected_count"], errors="coerce")
                    st.info(f"Bu gün genelinde (şehir toplamı gibi düşünme; hücre bazlı özet) **beklenen suç** değerleri medyanı: **{_fmt_expected(v.median())}**")

        st.caption("İpucu: Hücrelerin üzerine gelerek (hover) sahaya yönelik kısa notları görebilirsiniz.")

    # ----------------------
    # TAB: HARM
    # ----------------------
    with tab_harm:
        mode = "harm"

        gj_enriched = enrich_geojson(
            gj=gj,
            df_hr=df_hr,
            mode=mode,
            hr_label=str(hr_label),
            geoid_profile=geoid_profile,
        )
        draw_map(gj_enriched, mode=mode)

        # Top-K panel (ops_brief_topk varsa) — harm
        render_topk_panel(ops_topk, sel_date, str(hr_label), mode=mode)

        st.caption("İpucu: Zarar görünümü, **expected_harm** varsa onu kullanır; yoksa otomatik olarak daha basit bir fallback uygular.")

    # ===== Optional: Alt kısımda “Top-N hızlı liste” (harita bağımsız, sahaya hızlı)
    st.markdown("### 📌 Hızlı Liste (Bu dilimde en yüksek riskli hücreler)")
    # burada mode seçimi: kullanıcı hangi tabdaysa onu anlamak zor; basitçe iki küçük buton
    c1, c2 = st.columns(2)
    with c1:
        show_mode = st.radio("Liste metriği", options=["Suç", "Zarar"], horizontal=True, index=0)
    mode2 = "harm" if show_mode == "Zarar" else "crime"

    metric, _ = _compute_metric(df_hr, mode2)
    lik, _cuts = _compute_likert_quintiles(metric)
    tmp = df_hr.copy()
    tmp["_metric"] = pd.to_numeric(metric, errors="coerce")
    tmp["_lik"] = lik
    tmp["_geoid"] = tmp[geoid_col].map(_digits11)

    # sort
    tmp = tmp.sort_values(["_lik", "_metric"], ascending=[False, False]).drop_duplicates("_geoid", keep="first")
    view_cols = ["_geoid"]
    for c in ["p_event", "expected_count", "expected_harm", "top1_category", "top2_category", "top3_category"]:
        if c in tmp.columns:
            view_cols.append(c)
    out = tmp[view_cols].head(int(topn)).copy()
    out = out.rename(columns={"_geoid": "GEOID"})

    # format columns
    if "p_event" in out.columns:
        out["p_event"] = pd.to_numeric(out["p_event"], errors="coerce").map(_fmt3)
    if "expected_count" in out.columns:
        out["expected_count"] = pd.to_numeric(out["expected_count"], errors="coerce").map(_fmt_expected)
    if "expected_harm" in out.columns:
        out["expected_harm"] = pd.to_numeric(out["expected_harm"], errors="coerce").map(_fmt_big)

    st.dataframe(out, use_container_width=True, hide_index=True)
