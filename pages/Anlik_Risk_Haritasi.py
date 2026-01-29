# pages/Anlik_Risk_Haritasi.py
# SUTAM — Anlık Risk Haritası (ANLIK • SF saatine göre hour_range)
# - Likert (1-5): SEÇİLİ saat dilimindeki risk dağılımına göre (quantile / "çan eğrisi" mantığı)
# - Tooltip: GEOID + Risk seviyesi + p_event + expected + top1-3 + mikro kolluk önerisi
# - Seçili hücre analizi yok (kaldırıldı)
# - Legend: %0-20 ... %80-100 (saat dilimi içi göreli)

from __future__ import annotations

import os, json
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

DATA_DIR = os.getenv("DATA_DIR", "data").rstrip("/")
FC_CANDIDATES = [
    f"{DATA_DIR}/forecast_7d.parquet",
    f"{DATA_DIR}/full_fc.parquet",
    "data/forecast_7d.parquet",
    "deploy/full_fc.parquet",
    "data/full_fc.parquet",
]
GEOJSON_PATH = os.getenv("GEOJSON_PATH", "data/sf_cells.geojson")
TARGET_TZ = "America/Los_Angeles"

# Renkler (kurumsal-yumuşak)
LIKERT = {
    1: ("Çok Düşük",  [56, 189, 137]),
    2: ("Düşük",      [104, 207, 162]),
    3: ("Orta",       [241, 196, 15]),
    4: ("Yüksek",     [235, 147, 80]),
    5: ("Çok Yüksek", [220, 88, 76]),
}
DEFAULT_FILL = [220, 220, 220]

# --- helpers ---
def _first_existing(paths: list[str]) -> str | None:
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def digits11(x) -> str:
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

def _fmt_expected_band(x) -> str:
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

# --- loaders ---
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

# --- risk to likert (SADECE saat dilimi içinde quantile) ---
def compute_relative_likert(df_hr: pd.DataFrame) -> tuple[pd.Series, str]:
    """
    Likert'i o saat dilimindeki dağılıma göre 5'e böler.
    Öncelik: risk_score -> risk_prob -> p_event
    """
    col = _pick_col(df_hr, ["risk_score", "risk_prob", "p_event"])
    if not col:
        return pd.Series([3] * len(df_hr), index=df_hr.index), "risk_score/risk_prob/p_event yok"

    vals = pd.to_numeric(df_hr[col], errors="coerce")
    if vals.notna().sum() < 10:
        # çok az veri varsa sabit orta
        return pd.Series([3] * len(df_hr), index=df_hr.index), f"{col} az veri"

    # qcut aynı değerlerde hata verebilir -> rank ile stabilize
    ranked = vals.rank(method="first")
    try:
        bins = pd.qcut(ranked, 5, labels=[1, 2, 3, 4, 5])
        return bins.astype(int), f"quantile({col})"
    except Exception:
        # fallback: percentiles manual
        q = np.nanpercentile(vals, [20, 40, 60, 80])
        out = pd.Series(3, index=df_hr.index)
        out[vals <= q[0]] = 1
        out[(vals > q[0]) & (vals <= q[1])] = 2
        out[(vals > q[1]) & (vals <= q[2])] = 3
        out[(vals > q[2]) & (vals <= q[3])] = 4
        out[vals > q[3]] = 5
        return out.astype(int), f"percentile({col})"

def micro_ops_text(likert: int) -> str:
    # Kısa, doğrudan, “emir” gibi olmayan dil
    if likert >= 5:
        return "Öneri: Kritik yoğunluk görülebilir. Görünür devriye ve giriş–çıkış akslarında kısa süreli yoğunlaştırma değerlendirilebilir."
    if likert == 4:
        return "Öneri: Risk artışı olabilir. Transit/ana arter çevresinde kısa kontrollü tur planlanabilir."
    if likert == 3:
        return "Öneri: Rutin devriye yeterli; anomali gözlemi odaklı izleme yapılabilir."
    if likert == 2:
        return "Öneri: Düşük risk; standart devriye ve caydırıcılık odaklı dolaşım uygundur."
    return "Öneri: Çok düşük risk; rutin görünürlük korunabilir."

def render_legend_compact():
    # “hover” cümlesi yerine yüzde dilimleri
    with st.popover("🎨 Risk Ölçeği", use_container_width=False):
        st.markdown("**Bu saat dilimi içinde göreli sınıflandırma**")
        st.caption("Risk seviyeleri, seçili tarih+saat dilimindeki tüm hücrelerin risk skorları dağılımı %20’lik dilimlere bölünerek hesaplanır.")
        items = [
            (1, "Çok Düşük", "0–20"),
            (2, "Düşük", "20–40"),
            (3, "Orta", "40–60"),
            (4, "Yüksek", "60–80"),
            (5, "Çok Yüksek", "80–100"),
        ]
        for k, label, pct in items:
            rgb = LIKERT[k][1]
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; margin:8px 0;">
                  <div style="display:flex; align-items:center; gap:10px;">
                    <div style="width:14px;height:14px;border-radius:5px;background:rgb({rgb[0]},{rgb[1]},{rgb[2]});"></div>
                    <div><b>{k}</b> — {label}</div>
                  </div>
                  <div style="opacity:0.75;">%{pct}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.caption("Not: Bu ölçek mutlak eşik değildir; aynı saat dilimindeki hücrelerin birbirine göre konumunu gösterir.")

def enrich_geojson(gj: dict, df_hr: pd.DataFrame) -> dict:
    if not gj or df_hr.empty:
        return gj

    df = df_hr.copy()

    # GEOID
    geoid_col = _pick_col(df, ["GEOID", "geoid"])
    df["geoid"] = df[geoid_col].map(digits11) if geoid_col else ""

    # p & expected
    pe = _pick_col(df, ["p_event", "risk_prob"])
    ex = _pick_col(df, ["expected_count", "expected_crimes"])
    df["_p_event"] = pd.to_numeric(df[pe], errors="coerce") if pe else np.nan
    df["_expected"] = pd.to_numeric(df[ex], errors="coerce") if ex else np.nan
    df["p_event_txt"] = df["_p_event"].map(_fmt3)
    df["expected_txt"] = df["_expected"].map(_fmt_expected_band)

    # top cats
    t1 = _pick_col(df, ["top1_category", "top1_cat", "cat1"])
    t2 = _pick_col(df, ["top2_category", "top2_cat", "cat2"])
    t3 = _pick_col(df, ["top3_category", "top3_cat", "cat3"])

    def _clean(s: pd.Series) -> pd.Series:
        return s.astype(str).replace("nan", "").replace("None", "").fillna("")

    df["top1_category"] = _clean(df[t1]) if t1 else ""
    df["top2_category"] = _clean(df[t2]) if t2 else ""
    df["top3_category"] = _clean(df[t3]) if t3 else ""

    # ✅ SADECE BU SAAT DİLİMİ İÇİN: göreli likert
    df["risk_likert"], _method = compute_relative_likert(df)
    df["likert_label"] = df["risk_likert"].map(lambda k: LIKERT[int(k)][0] if int(k) in LIKERT else "Orta")
    df["fill_color"] = df["risk_likert"].map(lambda k: LIKERT[int(k)][1] if int(k) in LIKERT else DEFAULT_FILL)

    # mikro öneri tooltip içine
    df["ops_tip"] = df["risk_likert"].map(lambda k: micro_ops_text(int(k)))

    # tekilleştir
    df["_exp_num"] = pd.to_numeric(df["_expected"], errors="coerce").fillna(0.0)
    df = df.sort_values(["risk_likert", "_exp_num"], ascending=[False, False]).drop_duplicates("geoid", keep="first")
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

        key = digits11(raw)
        props["display_id"] = str(raw) if raw not in (None, "") else key

        # defaults
        props["likert_label"] = ""
        props["p_event_txt"] = "—"
        props["expected_txt"] = "—"
        props["top1_category"] = ""
        props["top2_category"] = ""
        props["top3_category"] = ""
        props["ops_tip"] = ""
        props["fill_color"] = DEFAULT_FILL

        if key and key in dmap.index:
            row = dmap.loc[key]
            props["likert_label"] = str(row.get("likert_label") or "")
            props["p_event_txt"] = str(row.get("p_event_txt") or "—")
            props["expected_txt"] = str(row.get("expected_txt") or "—")
            props["top1_category"] = str(row.get("top1_category") or "")
            props["top2_category"] = str(row.get("top2_category") or "")
            props["top3_category"] = str(row.get("top3_category") or "")
            props["ops_tip"] = str(row.get("ops_tip") or "")
            props["fill_color"] = row.get("fill_color", DEFAULT_FILL)

        feats_out.append({**feat, "properties": props})

    return {**gj, "features": feats_out}

def draw_map(gj: dict):
    layer = pdk.Layer(
        "GeoJsonLayer",
        gj,
        stroked=True,
        get_line_color=[80, 80, 80],
        line_width_min_pixels=0.6,
        filled=True,
        get_fill_color="properties.fill_color",
        pickable=True,
        opacity=0.65,
    )
    tooltip = {
        "html": (
            "<b>GEOID:</b> {display_id}"
            "<br/><b>Risk Seviyesi:</b> {likert_label}"
            "<br/><b>Suç olasılığı (p):</b> {p_event_txt}"
            "<br/><b>Beklenen suç sayısı:</b> {expected_txt}"
            "<hr style='opacity:0.28'/>"
            "<b>En olası 3 suç:</b>"
            "<br/>• {top1_category}"
            "<br/>• {top2_category}"
            "<br/>• {top3_category}"
            "<hr style='opacity:0.28'/>"
            "<b>Kolluk Notu:</b><br/>{ops_tip}"
        ),
        "style": {"backgroundColor": "#111827", "color": "white", "maxWidth": "360px"},
    }
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=37.7749, longitude=-122.4194, zoom=10),
        map_style="light",
        tooltip=tooltip,
    )
    st.pydeck_chart(deck, use_container_width=True)

def render_anlik_risk_haritasi():
    st.markdown("# Anlık Risk Haritası")
    st.caption("San Francisco yerel saatine göre mevcut saat dilimindeki risk düzeylerini 5’li ölçekte gösterir.")

    if _IMPORT_SRC_ERR is not None:
        st.error("`src.io_data` import edilemedi. `src/` klasörünü ve yolları kontrol edin.")
        st.code(repr(_IMPORT_SRC_ERR))
        return

    fc = load_forecast()
    if fc.empty:
        st.error("Forecast verisi bulunamadı/boş. `data/forecast_7d.parquet` veya `deploy/full_fc.parquet` gerekli.")
        return

    date_col = _pick_col(fc, ["date"])
    hr_col = _pick_col(fc, ["hour_range", "hour_bucket"])
    if not date_col or not hr_col:
        st.error("Forecast içinde `date` ve/veya `hour_range` yok.")
        return

    fc = fc.copy()
    fc[date_col] = pd.to_datetime(fc[date_col], errors="coerce")
    fc["date_norm"] = fc[date_col].dt.normalize()

    now_sf = datetime.now(ZoneInfo(TARGET_TZ))
    today = pd.Timestamp(now_sf.date())

    dates = sorted(fc["date_norm"].dropna().unique())
    if not dates:
        st.error("Forecast içinde geçerli tarih yok.")
        return

    sel_date = today if today in dates else max([d for d in dates if d <= today], default=dates[0])
    labels = sorted(fc[hr_col].dropna().astype(str).unique().tolist())
    hr_label = _hour_to_bucket(now_sf.hour, labels) or (labels[0] if labels else None)
    if not hr_label:
        st.error("Forecast içinde hour_range bulunamadı.")
        return

    st.caption(f"SF saati: **{now_sf:%Y-%m-%d %H:%M}** • Tarih: **{pd.Timestamp(sel_date).date()}** • Dilim: **{hr_label}**")

    # ✅ Senin istediğin cümle (daha düzgün)
    st.info("Risk ölçeği, bu tarih ve saat dilimindeki hücrelere ait risk skorlarının dağılımı temel alınarak göreli (%20’lik dilimler) şekilde hesaplanmıştır.")

    # Legend popover (yüzdeli)
    render_legend_compact()

    df_hr = fc[(fc["date_norm"] == sel_date) & (fc[hr_col].astype(str) == str(hr_label))].copy()
    if df_hr.empty:
        st.warning("Bu tarih/saat dilimi için kayıt bulunamadı.")
        return

    gj = load_geojson()
    if not gj:
        st.error(f"GeoJSON bulunamadı: `{GEOJSON_PATH}`")
        return

    gj_enriched = enrich_geojson(gj, df_hr)
    draw_map(gj_enriched)

    # İstersen burada alt kısma sadece kısa not bırak:
    st.caption("Not: Çıktılar karar destek amaçlıdır; saha bilgisi ve amir değerlendirmesi ile birlikte yorumlanmalıdır.")
