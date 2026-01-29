# pages/Anlik_Risk_Haritasi.py
# SUTAM — Anlık Risk Haritası (Kolluk için sade • seçim yok • SF saatine göre anlık hour_range)
# - Veri: data/forecast_7d.parquet (fallback: deploy/full_fc.parquet)
# - GeoJSON: data/sf_cells.geojson
# - Hover: beklenen suç sayısı, suç olasılığı, en olası 3 suç
# - Sağ üst: sabit açılır "Risk Ölçeği"
# - Alt panel: bu saat dilimi için genel kolluk önerisi (Top-N hücre üzerinden)
# - NOT: st.set_page_config burada yok (app.py zaten set ediyor)

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
# PATHS
# =============================================================================
DATA_DIR = os.getenv("DATA_DIR", "data").rstrip("/")
FC_CANDIDATES = [
    f"{DATA_DIR}/forecast_7d.parquet",
    f"{DATA_DIR}/full_fc.parquet",
    "data/forecast_7d.parquet",
    "deploy/full_fc.parquet",
]
GEOJSON_PATH = os.getenv("GEOJSON_PATH", "data/sf_cells.geojson")
TARGET_TZ = "America/Los_Angeles"

# 5'li Likert + renk
LIKERT = {
    1: ("Çok Düşük",  [46, 204, 113]),
    2: ("Düşük",      [88, 214, 141]),
    3: ("Orta",       [241, 196, 15]),
    4: ("Yüksek",     [230, 126, 34]),
    5: ("Çok Yüksek", [192, 57, 43]),
}
DEFAULT_FILL = [220, 220, 220]


# =============================================================================
# UI HELPERS
# =============================================================================
def render_fixed_legend():
    """Sağ-üst sabit açılır pencere (details/summary)"""
    items_html = ""
    for i in range(1, 6):
        label, rgb = LIKERT[i]
        items_html += f"""
        <div class="sutam-legend-row">
          <span class="sutam-legend-swatch" style="background: rgb({rgb[0]},{rgb[1]},{rgb[2]});"></span>
          <span><b>{i}</b> — {label}</span>
        </div>
        """

    st.markdown(
        f"""
        <style>
          /* Sağ üst sabit legend */
          .sutam-legend {{
            position: fixed;
            top: 78px;       /* üst bar + biraz boşluk */
            right: 22px;
            z-index: 9999;
            width: 240px;
            background: rgba(255,255,255,0.96);
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            box-shadow: 0 6px 18px rgba(15,23,42,0.10);
            padding: 10px 10px;
            backdrop-filter: blur(6px);
          }}
          .sutam-legend details {{
            width: 100%;
          }}
          .sutam-legend summary {{
            cursor: pointer;
            font-weight: 700;
            color: #0f172a;
            list-style: none;
            outline: none;
          }}
          /* summary ok işaretini güzelleştir */
          .sutam-legend summary::-webkit-details-marker {{
            display: none;
          }}
          .sutam-legend summary:before {{
            content: "▸";
            display: inline-block;
            margin-right: 8px;
            transform: rotate(0deg);
            transition: transform .15s ease;
            color: #334155;
          }}
          .sutam-legend details[open] summary:before {{
            transform: rotate(90deg);
          }}
          .sutam-legend-body {{
            margin-top: 8px;
          }}
          .sutam-legend-row {{
            display: flex;
            gap: 10px;
            align-items: center;
            font-size: 0.92rem;
            color: #0f172a;
            padding: 4px 0;
          }}
          .sutam-legend-swatch {{
            width: 14px;
            height: 14px;
            border-radius: 4px;
            border: 1px solid rgba(15,23,42,0.15);
          }}

          /* Küçük ekranlarda kaplasın diye genişliği azalt */
          @media (max-width: 900px) {{
            .sutam-legend {{
              width: 210px;
              top: 70px;
              right: 12px;
            }}
          }}
        </style>

        <div class="sutam-legend">
          <details>
            <summary>🎨 Risk Ölçeği (5’li)</summary>
            <div class="sutam-legend-body">
              {items_html}
              <div style="margin-top:8px; font-size:0.82rem; color:#64748b;">
                Not: Gösterim göreli risk seviyesidir.
              </div>
            </div>
          </details>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# DATA HELPERS
# =============================================================================
def _first_existing(paths: list[str]) -> str | None:
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def _digits11(x) -> str:
    s = "".join(ch for ch in str(x) if ch.isdigit())
    return s.zfill(11) if s else ""

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

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k.lower() in cols:
            return cols[k.lower()]
    return None

def _risk_to_likert(df_hr: pd.DataFrame) -> pd.Series:
    direct = _pick_col(df_hr, ["risk_likert", "likert", "risk5", "risk_level_5"])
    if direct:
        s = pd.to_numeric(df_hr[direct], errors="coerce").fillna(3).astype(int)
        return s.clip(1, 5)

    rlev = _pick_col(df_hr, ["risk_level", "level"])
    if rlev:
        s = df_hr[rlev].astype(str).str.lower()
        mapping = {
            "very_low": 1, "vlow": 1, "low": 2,
            "medium": 3, "mid": 3,
            "high": 4,
            "critical": 5, "very_high": 5, "vhigh": 5
        }
        out = s.map(mapping)
        if out.notna().any():
            return out.fillna(3).astype(int).clip(1, 5)

    rs_col = _pick_col(df_hr, ["risk_score", "risk", "p_event", "prob_event"])
    rs = pd.to_numeric(df_hr[rs_col], errors="coerce") if rs_col else pd.Series([np.nan] * len(df_hr), index=df_hr.index)
    if rs.notna().any():
        try:
            bins = pd.qcut(rs.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
            return bins.astype(int)
        except Exception:
            pass

    return pd.Series([3] * len(df_hr), index=df_hr.index)


# =============================================================================
# LOADERS
# =============================================================================
@st.cache_data(show_spinner=False)
def load_forecast() -> pd.DataFrame:
    p = _first_existing(FC_CANDIDATES)
    if not p:
        return pd.DataFrame()
    if load_parquet_or_csv is None:
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

def enrich_geojson(gj: dict, df_hr: pd.DataFrame) -> dict:
    if not gj or df_hr.empty:
        return gj

    df = df_hr.copy()

    geoid_col = _pick_col(df, ["GEOID", "geoid"])
    df["geoid11"] = df[geoid_col].map(_digits11) if geoid_col else ""

    exp_col = _pick_col(df, ["expected_count", "exp_count", "lambda", "mu"])
    pe_col  = _pick_col(df, ["p_event", "prob_event", "crime_prob", "risk_prob"])

    df["expected_txt"] = df[exp_col].map(_fmt_expected) if exp_col else "—"
    df["p_event_txt"]  = df[pe_col].map(_fmt3) if pe_col else "—"

    top1 = _pick_col(df, ["top1_category", "top1_cat", "cat1"])
    top2 = _pick_col(df, ["top2_category", "top2_cat", "cat2"])
    top3 = _pick_col(df, ["top3_category", "top3_cat", "cat3"])
    df["top1_category"] = df[top1].astype(str) if top1 else ""
    df["top2_category"] = df[top2].astype(str) if top2 else ""
    df["top3_category"] = df[top3].astype(str) if top3 else ""

    df["risk_likert"] = _risk_to_likert(df)
    df["likert_label"] = df["risk_likert"].map(lambda k: LIKERT.get(int(k), ("Orta", DEFAULT_FILL))[0])
    df["fill_color"]   = df["risk_likert"].map(lambda k: LIKERT.get(int(k), ("Orta", DEFAULT_FILL))[1])

    dmap = df.set_index("geoid11")

    feats_out = []
    for feat in gj.get("features", []):
        props = dict(feat.get("properties") or {})

        raw = None
        for k in ("geoid", "GEOID", "cell_id", "id", "geoid11", "geoid_11"):
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

        props["likert_label"] = ""
        props["expected_txt"] = "—"
        props["p_event_txt"]  = "—"
        props["top1_category"] = ""
        props["top2_category"] = ""
        props["top3_category"] = ""
        props["fill_color"] = DEFAULT_FILL

        if key and key in dmap.index:
            row = dmap.loc[key]
            props["likert_label"] = str(row.get("likert_label", ""))
            props["expected_txt"] = str(row.get("expected_txt", "—"))
            props["p_event_txt"]  = str(row.get("p_event_txt", "—"))
            props["top1_category"] = str(row.get("top1_category", "") or "")
            props["top2_category"] = str(row.get("top2_category", "") or "")
            props["top3_category"] = str(row.get("top3_category", "") or "")
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
            "<b>GEOID:</b> {properties.display_id}"
            "<br/><b>Risk Seviyesi:</b> {properties.likert_label}"
            "<br/><b>Suç olasılığı (p):</b> {properties.p_event_txt}"
            "<br/><b>Beklenen suç sayısı:</b> {properties.expected_txt}"
            "<hr style='opacity:0.30'/>"
            "<b>En olası 3 suç:</b>"
            "<br/>• {properties.top1_category}"
            "<br/>• {properties.top2_category}"
            "<br/>• {properties.top3_category}"
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

def make_ops_suggestions(df_hr: pd.DataFrame, top_n: int = 20) -> dict:
    if df_hr.empty:
        return {"title": "Kolluk Önerisi", "bullets": ["Veri bulunamadı."]}

    tmp = df_hr.copy()
    tmp["risk_likert"] = _risk_to_likert(tmp)

    exp_col = _pick_col(tmp, ["expected_count", "exp_count", "lambda", "mu"])
    tmp["_exp"] = pd.to_numeric(tmp[exp_col], errors="coerce").fillna(0.0) if exp_col else 0.0

    top = tmp.sort_values(["risk_likert", "_exp"], ascending=[False, False]).head(top_n)

    max_l = int(top["risk_likert"].max())
    max_label = LIKERT.get(max_l, ("Orta", DEFAULT_FILL))[0]

    top1 = _pick_col(top, ["top1_category", "top1_cat", "cat1"])
    cats = [c for c in top[top1].astype(str).tolist()] if top1 else []
    top_cats = pd.Series([c for c in cats if c and c.lower() != "nan"]).value_counts().head(3).index.tolist()

    bullets = []
    if max_l >= 4:
        bullets.append("Sıcak noktalarda görünür devriye yoğunluğu artırılabilir (Top-K hücreler öncelikli).")
        bullets.append("Transit/ana arter ve yoğun yaya akışlı bölgelerde kısa süreli yoğunlaştırılmış devriye önerilir.")
        bullets.append("Kamera kör noktaları ve giriş–çıkış akslarında çevrimli devriye değerlendirilebilir.")
    else:
        bullets.append("Rutin görünür devriye ve caydırıcılık odaklı dolaşım önerilir.")

    if top_cats:
        bullets.append(f"Bu saat diliminde öne çıkan suç türleri: {', '.join(top_cats)}.")

    bullets.append("Not: Çıktılar bağlayıcı değildir; saha bilgisi ve amir değerlendirmesi ile birlikte yorumlanmalıdır.")
    return {"title": f"Kolluk Önerisi (Bu saat dilimi • en yüksek risk: {max_label})", "bullets": bullets}


# =============================================================================
# PUBLIC ENTRYPOINT
# =============================================================================
def render_anlik_risk_haritasi():
    # Sabit açılır legend (sağ üst)
    render_fixed_legend()

    st.markdown("# 🗺️ Anlık Risk Haritası")
    st.caption("San Francisco yerel saatine göre mevcut saat dilimindeki risk düzeylerini 5’li ölçekte gösterir (seçim yok).")

    if _IMPORT_SRC_ERR is not None:
        st.error("`src.io_data` modülü import edilemedi. `src/` klasörünü ve dosya yollarını kontrol edin.")
        st.code(repr(_IMPORT_SRC_ERR))
        return

    fc = load_forecast()
    if fc.empty:
        st.error(
            "Forecast verisi bulunamadı/boş.\n\n"
            "Beklenen dosyalardan en az biri gerekli:\n"
            + "\n".join([f"- {p}" for p in FC_CANDIDATES])
        )
        return

    date_col = _pick_col(fc, ["date"])
    hr_col   = _pick_col(fc, ["hour_range", "hour_bucket"])
    if not date_col or not hr_col:
        st.error("Forecast içinde `date` ve/veya `hour_range` kolonu yok. `prepare_forecast` çıktısını kontrol edin.")
        return

    fc = fc.copy()
    fc[date_col] = pd.to_datetime(fc[date_col], errors="coerce")
    fc["date_norm"] = fc[date_col].dt.normalize()

    now_sf = datetime.now(ZoneInfo(TARGET_TZ))
    today = pd.Timestamp(now_sf.date())

    dates = sorted(fc["date_norm"].dropna().unique())
    if not dates:
        st.error("Forecast içinde geçerli tarih bulunamadı.")
        return

    sel_date = today if today in dates else max([d for d in dates if d <= today], default=dates[0])

    labels = sorted(fc[hr_col].dropna().astype(str).unique().tolist())
    hr_label = _hour_to_bucket(now_sf.hour, labels) or (labels[0] if labels else None)
    if not hr_label:
        st.error("Forecast içinde saat dilimi bulunamadı.")
        return

    st.caption(f"SF saati: **{now_sf:%Y-%m-%d %H:%M}**  •  Tarih: **{pd.Timestamp(sel_date).date()}**  •  Dilim: **{hr_label}**")

    df_hr = fc[(fc["date_norm"] == sel_date) & (fc[hr_col].astype(str) == str(hr_label))].copy()
    if df_hr.empty:
        st.warning("Bu tarih/saat dilimi için kayıt bulunamadı.")
        return

    gj = load_geojson()
    if not gj:
        st.error(f"GeoJSON bulunamadı: `{GEOJSON_PATH}` (polygonlar gerekli).")
        return

    gj_enriched = enrich_geojson(gj, df_hr)
    draw_map(gj_enriched)

    st.divider()
    ops = make_ops_suggestions(df_hr, top_n=20)
    st.subheader("👮 " + ops["title"])
    for b in ops["bullets"]:
        st.write("•", b)
