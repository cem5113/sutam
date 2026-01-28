# pages/anlik_risk_haritasi.py
# SUTAM — Anlık Risk Haritası (Kolluk için sade • seçim yok • 5’li Likert)
# - app.py router tarafından çağrılır: render_anlik_risk_haritasi()

from __future__ import annotations

import os, json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

from src.io_data import load_parquet_or_csv, prepare_forecast  # gp yok

# ----------------------------
# Settings / Paths
# ----------------------------
DATA_DIR = os.getenv("DATA_DIR", "data").rstrip("/")

FC_CANDIDATES = [
    f"{DATA_DIR}/forecast_7d.parquet",
    f"{DATA_DIR}/full_fc.parquet",
    "deploy/forecast_7d.parquet",
    "deploy/full_fc.parquet",
    "data/forecast_7d.parquet",
    "data/full_fc.parquet",
]

GEOJSON_LOCAL = os.getenv("GEOJSON_PATH", "data/sf_cells.geojson")
TARGET_TZ = "America/Los_Angeles"

LIKERT = {
    1: ("Çok Düşük", [46, 204, 113]),
    2: ("Düşük",     [88, 214, 141]),
    3: ("Orta",      [241, 196, 15]),
    4: ("Yüksek",    [230, 126, 34]),
    5: ("Çok Yüksek",[192, 57, 43]),
}
DEFAULT_FILL = [220, 220, 220]


def _is_url(p: str) -> bool:
    return str(p).startswith(("http://", "https://"))

def _first_existing(candidates: list[str]) -> str | None:
    for p in candidates:
        if _is_url(p):
            return p
        if os.path.exists(p):
            return p
    return None

def _digits11(x) -> str:
    s = "".join(ch for ch in str(x) if ch.isdigit())
    return s.zfill(11) if s else ""

def _safe_str(x) -> str:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)

def _format_expected(x) -> str:
    try:
        v = float(x)
        if v < 0:
            v = 0.0
        lo = int(np.floor(v))
        hi = int(np.ceil(v))
        return f"~{lo}" if lo == hi else f"~{lo}–{hi}"
    except Exception:
        return "—"

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
        rg = _parse_range(lab)
        if rg:
            parsed.append((lab, rg[0], rg[1]))

    for lab, s, e in parsed:
        if s <= h < e:
            return lab

    for lab, s, e in parsed:
        if s > e and (h >= s or h < e):
            return lab

    return parsed[0][0] if parsed else None

def _risk_to_likert(df_hr: pd.DataFrame) -> pd.Series:
    for c in ["risk_likert", "likert", "risk_level_5", "risk5"]:
        if c in df_hr.columns:
            s = pd.to_numeric(df_hr[c], errors="coerce").fillna(3).astype(int)
            return s.clip(1, 5)

    if "risk_level" in df_hr.columns:
        s = df_hr["risk_level"].astype(str).str.lower()
        mapping = {
            "very_low": 1, "vlow": 1, "low": 2,
            "medium": 3, "mid": 3,
            "high": 4,
            "critical": 5, "very_high": 5, "vhigh": 5
        }
        out = s.map(mapping)
        if out.notna().any():
            return out.fillna(3).astype(int).clip(1, 5)

    rs = pd.to_numeric(df_hr.get("risk_score", pd.Series([np.nan] * len(df_hr))), errors="coerce")
    if rs.notna().any():
        try:
            bins = pd.qcut(rs.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
            return bins.astype(int)
        except Exception:
            pass

    return pd.Series([3] * len(df_hr), index=df_hr.index)

@st.cache_data(show_spinner=False)
def _load_fc() -> pd.DataFrame:
    fc_path = _first_existing(FC_CANDIDATES)
    if not fc_path:
        return pd.DataFrame()

    try:
        if _is_url(fc_path):
            fc = pd.read_parquet(fc_path) if fc_path.lower().endswith(".parquet") else pd.read_csv(fc_path)
        else:
            fc = load_parquet_or_csv(fc_path)
    except Exception:
        return pd.DataFrame()

    try:
        fc = prepare_forecast(fc, gp=None) if not fc.empty else fc
    except TypeError:
        pass

    return fc

@st.cache_data(show_spinner=False)
def _load_geojson() -> dict:
    if os.path.exists(GEOJSON_LOCAL):
        with open(GEOJSON_LOCAL, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _enrich_geojson(gj: dict, df_hr: pd.DataFrame) -> dict:
    if not gj or df_hr.empty:
        return gj

    df = df_hr.copy()

    if "GEOID" in df.columns:
        df["geoid"] = df["GEOID"].map(_digits11)
    elif "geoid" in df.columns:
        df["geoid"] = df["geoid"].map(_digits11)
    else:
        df["geoid"] = ""

    df["risk_likert"] = _risk_to_likert(df)
    df["likert_label"] = df["risk_likert"].map(lambda k: LIKERT.get(int(k), ("Orta", DEFAULT_FILL))[0])
    df["fill_color"]   = df["risk_likert"].map(lambda k: LIKERT.get(int(k), ("Orta", DEFAULT_FILL))[1])

    if "expected_count" not in df.columns:
        df["expected_count"] = np.nan
    df["expected_txt"] = df["expected_count"].map(_format_expected)

    for i in (1, 2, 3):
        c = f"top{i}_category"
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].map(_safe_str)

    dmap = df.set_index("geoid")

    feats_out = []
    for feat in gj.get("features", []):
        props = dict(feat.get("properties") or {})

        raw = None
        for k in ("geoid", "GEOID", "cell_id", "id", "geoid11", "geoid_11"):
            if k in props:
                raw = props[k]; break
        if raw is None:
            for k, v in props.items():
                if "geoid" in str(k).lower():
                    raw = v; break

        key = _digits11(raw)
        props["geoid"] = key

        props["likert_label"] = ""
        props["expected_txt"] = "—"
        props["top1_category"] = ""
        props["top2_category"] = ""
        props["top3_category"] = ""
        props["fill_color"] = DEFAULT_FILL

        if key and key in dmap.index:
            row = dmap.loc[key]
            props["likert_label"] = _safe_str(row.get("likert_label", ""))
            props["expected_txt"] = _safe_str(row.get("expected_txt", "—"))
            props["top1_category"] = _safe_str(row.get("top1_category", ""))
            props["top2_category"] = _safe_str(row.get("top2_category", ""))
            props["top3_category"] = _safe_str(row.get("top3_category", ""))
            props["fill_color"] = row.get("fill_color", DEFAULT_FILL)

        feats_out.append({**feat, "properties": props})

    return {**gj, "features": feats_out}

def _draw_map(gj: dict):
    layer = pdk.Layer(
        "GeoJsonLayer",
        gj,
        stroked=True,
        get_line_color=[90, 90, 90],
        line_width_min_pixels=0.5,
        filled=True,
        get_fill_color="properties.fill_color",
        pickable=True,
        opacity=0.65,
    )

    tooltip = {
        "html": (
            "<b>Risk Seviyesi:</b> {likert_label}"
            "<br/><b>Beklenen olay (bu saat dilimi):</b> {expected_txt}"
            "<hr style='opacity:0.25'/>"
            "<b>En olası suç türleri:</b>"
            "<br/>• {top1_category}"
            "<br/>• {top2_category}"
            "<br/>• {top3_category}"
        ),
        "style": {"backgroundColor": "#111827", "color": "white"},
    }

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=37.7749, longitude=-122.4194, zoom=10),
        map_style="light",
        tooltip=tooltip,
    )
    st.pydeck_chart(deck, use_container_width=True)

def _make_ops_suggestions(df_hr: pd.DataFrame, top_n: int = 15) -> dict:
    if df_hr.empty:
        return {"title": "Kolluk Önerisi", "bullets": ["Veri bulunamadı."]}

    tmp = df_hr.copy()
    tmp["risk_likert"] = _risk_to_likert(tmp)

    if "expected_count" not in tmp.columns:
        tmp["expected_count"] = np.nan
    tmp["expected_count_num"] = pd.to_numeric(tmp["expected_count"], errors="coerce").fillna(0.0)

    top = tmp.sort_values(["risk_likert", "expected_count_num"], ascending=[False, False]).head(top_n)

    max_l = int(top["risk_likert"].max())
    max_label = LIKERT.get(max_l, ("Orta", DEFAULT_FILL))[0]

    cats = []
    if "top1_category" in top.columns:
        cats = [c for c in top["top1_category"].astype(str).tolist() if c and c.lower() != "nan"]
    top_cats = pd.Series(cats).value_counts().head(3).index.tolist() if cats else []

    bullets = []
    if max_l >= 4:
        bullets.append("Yüksek riskli bölgelerde görünür devriye yoğunluğu artırılabilir (sıcak noktalar öncelikli).")
        bullets.append("Transit/ana arter ve yoğun yaya akışlı alanlarda kısa süreli yoğunlaştırılmış devriye önerilir.")
    else:
        bullets.append("Rutin görünür devriye ve caydırıcılık odaklı dolaşım önerilir.")

    if top_cats:
        bullets.append(f"Bu saat diliminde öne çıkan suç türleri: {', '.join(top_cats)}.")

    bullets.append("Not: Öneriler bağlayıcı değildir; saha bilgisi ve amir değerlendirmesi ile birlikte yorumlanmalıdır.")
    return {"title": f"Kolluk Önerisi (Bu saat dilimi • en yüksek risk: {max_label})", "bullets": bullets}


def render_anlik_risk_haritasi():
    st.markdown("# 🗺️ Anlık Risk Haritası")
    st.caption("Harita, San Francisco yerel saatine göre mevcut saat dilimindeki göreli risk seviyelerini 5’li ölçekle gösterir.")

    fc = _load_fc()
    if fc.empty:
        st.error(
            "Forecast verisi bulunamadı/boş.\n\n"
            "Beklenen dosyalardan en az biri gerekli:\n"
            f"- {FC_CANDIDATES[0]}\n- {FC_CANDIDATES[1]}\n"
        )
        return

    if "date" not in fc.columns or "hour_range" not in fc.columns:
        st.error("Forecast içinde `date` ve/veya `hour_range` kolonu yok. `prepare_forecast` çıktısını kontrol edin.")
        return

    fc = fc.copy()
    fc["date"] = pd.to_datetime(fc["date"], errors="coerce")
    fc["date_norm"] = fc["date"].dt.normalize()

    now_sf = datetime.now(ZoneInfo(TARGET_TZ))
    today = pd.Timestamp(now_sf.date())

    dates = sorted(fc["date_norm"].dropna().unique())
    if not dates:
        st.error("Forecast içinde geçerli tarih bulunamadı.")
        return

    sel_date = today if today in dates else (max([d for d in dates if d <= today], default=dates[0]))

    labels = sorted(fc["hour_range"].dropna().astype(str).unique().tolist())
    hr_label = _hour_to_bucket(now_sf.hour, labels) or (labels[0] if labels else None)
    if not hr_label:
        st.error("Forecast içinde saat dilimi bulunamadı.")
        return

    st.caption(f"SF saati: **{now_sf:%Y-%m-%d %H:%M}**  •  Tarih: **{pd.Timestamp(sel_date).date()}**  •  Dilim: **{hr_label}**")

    df_hr = fc[(fc["date_norm"] == sel_date) & (fc["hour_range"].astype(str) == str(hr_label))].copy()
    if df_hr.empty:
        st.warning("Bu tarih/saat dilimi için kayıt bulunamadı.")
        return

    gj = _load_geojson()
    if not gj:
        st.error(f"GeoJSON bulunamadı: `{GEOJSON_LOCAL}` (polygonlar gerekli).")
        return

    with st.expander("🎨 Risk Ölçeği (5’li)", expanded=False):
        cols = st.columns(5)
        for i, c in enumerate(cols, start=1):
            label, rgb = LIKERT[i]
            c.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:10px;">
                  <div style="width:16px;height:16px;border-radius:4px;background:rgb({rgb[0]},{rgb[1]},{rgb[2]});"></div>
                  <div style="font-size:14px;"><b>{i}</b> — {label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    gj_enriched = _enrich_geojson(gj, df_hr)
    _draw_map(gj_enriched)

    st.divider()
    ops = _make_ops_suggestions(df_hr, top_n=15)
    st.subheader("👮 " + ops["title"])
    for b in ops["bullets"]:
        st.write("•", b)
