# pages/Anlik_Risk_Haritasi.py
# SUTAM — Anlık Risk Haritası (SF saatine göre, seçim yok)
# - Veri: data/forecast_7d.parquet (fallback: deploy/full_fc.parquet)
# - GeoJSON: data/sf_cells.geojson
# - Risk 1–5: SADECE ilgili tarih+saat dilimindeki risk skorlarının dağılımına göre (quintile)
# - Tooltip: GEOID + risk seviyesi + p + expected + top3 + kolluk notu (GEOID bazlı)
# - Ayrı "seçili hücre analizi" YOK

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
TARGET_TZ = "America/Los_Angeles"

# 5'li Likert + renk (kurumsal ve sabit)
LIKERT = {
    1: ("Çok Düşük",  [46, 204, 113]),
    2: ("Düşük",      [88, 214, 141]),
    3: ("Orta",       [241, 196, 15]),
    4: ("Yüksek",     [230, 126, 34]),
    5: ("Çok Yüksek", [192, 57, 43]),
}
DEFAULT_FILL = [220, 220, 220]


# =============================================================================
# GLOBAL UI FIXES (tooltip overflow + compact)
# =============================================================================
def _apply_tooltip_css():
    st.markdown(
        """
        <style>
          /* Deck.gl tooltip container */
          .deckgl-tooltip {
            max-width: 340px !important;
            max-height: 320px !important;
            overflow: auto !important;
            padding: 10px 12px !important;
            line-height: 1.25 !important;
            border-radius: 12px !important;
            box-shadow: 0 10px 30px rgba(0,0,0,.25) !important;
          }
          /* Tooltip content spacing */
          .deckgl-tooltip hr { margin: 8px 0 !important; opacity: .25 !important; }
          /* Cursor'dan biraz sağa-aşağı kaydır: hep aşağı açılıyor hissini azaltır */
          .deckgl-tooltip {
            transform: translate(12px, 12px) !important;
          }
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

    for lab, s, e in parsed:
        if s <= h < e:
            return lab

    # wrap-around e.g. "21-3"
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


# =============================================================================
# RISK (quintile) + LEGEND CUTS
# =============================================================================
def _compute_likert_quintiles(df_slice: pd.DataFrame) -> tuple[pd.Series, dict]:
    """
    Likert 1–5: Sadece ilgili tarih+saat dilimindeki hücrelerin risk skor dağılımına göre (quintile).
    Dönüş:
      - likert_series (index df_slice ile aynı)
      - legend_meta: {"cuts": [q20,q40,q60,q80], "source_col": "..."}
    """
    # Risk kaynağı öncelik: risk_score -> p_event -> risk_prob -> expected_count (en son)
    src = (
        _pick_col(df_slice, ["risk_score"]) or
        _pick_col(df_slice, ["p_event"]) or
        _pick_col(df_slice, ["risk_prob"]) or
        _pick_col(df_slice, ["expected_count"]) or
        _pick_col(df_slice, ["expected_crimes"]) or
        None
    )

    if not src:
        lik = pd.Series([3] * len(df_slice), index=df_slice.index)
        return lik, {"cuts": [np.nan, np.nan, np.nan, np.nan], "source_col": None}

    v = pd.to_numeric(df_slice[src], errors="coerce")
    # Tamamen NaN ise
    if v.notna().sum() < 10:
        lik = pd.Series([3] * len(df_slice), index=df_slice.index)
        return lik, {"cuts": [np.nan, np.nan, np.nan, np.nan], "source_col": src}

    # Quintile: rank(method="first") ile eşitliklerde sorun azalt
    try:
        bins = pd.qcut(v.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        lik = bins.astype(int)
    except Exception:
        # fallback: cut by quantiles manually
        qs = v.quantile([0.2, 0.4, 0.6, 0.8]).values.tolist()
        q20, q40, q60, q80 = qs
        lik = pd.Series(3, index=df_slice.index)
        lik[v <= q20] = 1
        lik[(v > q20) & (v <= q40)] = 2
        lik[(v > q40) & (v <= q60)] = 3
        lik[(v > q60) & (v <= q80)] = 4
        lik[v > q80] = 5

    cuts = v.quantile([0.2, 0.4, 0.6, 0.8]).values.tolist()
    return lik, {"cuts": cuts, "source_col": src}


# =============================================================================
# GEOJSON ENRICH
# =============================================================================
def _likert_advice(k: int) -> str:
    # Tooltip içinde tek satır, kısa ve “kural” gibi değil; karar destek dili
    if k >= 5:
        return "Öneri: Kritik yoğunluk — görünür devriye ve kısa kontrollü tur artırılabilir."
    if k == 4:
        return "Öneri: Risk artışı olası — transit/ana arter çevresinde kısa kontrollü tur planlanabilir."
    if k == 3:
        return "Öneri: Orta risk — rutin devriye + caydırıcılık odaklı dolaşım önerilir."
    if k == 2:
        return "Öneri: Düşük risk — rutin dolaşım sürdürülür; gözlemsel teyit önerilir."
    return "Öneri: Çok düşük risk — temel görünürlük ve izleme yeterli olabilir."

def enrich_geojson(gj: dict, df_hr: pd.DataFrame) -> dict:
    if not gj or df_hr.empty:
        return gj

    df = df_hr.copy()

    # GEOID normalize
    geoid_col = _pick_col(df, ["GEOID", "geoid"])
    df["geoid"] = df[geoid_col].map(_digits11) if geoid_col else ""

    # p_event & expected
    pe_col = _pick_col(df, ["p_event", "risk_prob", "prob_event"])
    ex_col = _pick_col(df, ["expected_count", "expected_crimes", "mu", "lambda"])

    df["p_event_txt"] = pd.to_numeric(df[pe_col], errors="coerce").map(_fmt3) if pe_col else "—"
    df["expected_txt"] = pd.to_numeric(df[ex_col], errors="coerce").map(_fmt_expected) if ex_col else "—"

    # top categories (senin parquet'te top1_share vs var; biz sadece category'yi gösteriyoruz)
    for i in (1, 2, 3):
        c = _pick_col(df, [f"top{i}_category", f"top{i}_cat", f"cat{i}"])
        df[f"top{i}_category"] = df[c].astype(str).replace("nan", "").fillna("") if c else ""

    # Likert = quintile
    lik, _legend_meta = _compute_likert_quintiles(df)
    df["risk_likert"] = lik.clip(1, 5)

    df["likert_label"] = df["risk_likert"].map(lambda k: LIKERT[int(k)][0])
    df["fill_color"] = df["risk_likert"].map(lambda k: LIKERT[int(k)][1])
    df["advice_txt"] = df["risk_likert"].map(lambda k: _likert_advice(int(k)))

    # Aynı GEOID birden fazla satırsa: (risk yüksek + expected yüksek) en öndeki kalsın
    df["_exp_num"] = pd.to_numeric(df[ex_col], errors="coerce").fillna(0.0) if ex_col else 0.0
    df = (
        df.sort_values(["risk_likert", "_exp_num"], ascending=[False, False])
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
        props["top1_category"] = ""
        props["top2_category"] = ""
        props["top3_category"] = ""
        props["advice_txt"] = ""
        props["fill_color"] = DEFAULT_FILL

        if key and key in dmap.index:
            row = dmap.loc[key]  # Series
            props["likert_label"] = str(row.get("likert_label", "") or "")
            props["p_event_txt"] = str(row.get("p_event_txt", "—") or "—")
            props["expected_txt"] = str(row.get("expected_txt", "—") or "—")
            props["top1_category"] = str(row.get("top1_category", "") or "")
            props["top2_category"] = str(row.get("top2_category", "") or "")
            props["top3_category"] = str(row.get("top3_category", "") or "")
            props["advice_txt"] = str(row.get("advice_txt", "") or "")
            props["fill_color"] = row.get("fill_color", DEFAULT_FILL)

        feats_out.append({**feat, "properties": props})

    return {**gj, "features": feats_out}


# =============================================================================
# MAP
# =============================================================================
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
        opacity=0.65,
    )

    # Tooltip alanları: properties'e yazdık, burada doğrudan {field} kullanıyoruz
    tooltip = {
        "html": (
            "<div class='tt-title' style='font-weight:800; font-size:14px;'>GEOID: {display_id}</div>"
            "<div><b>Risk Seviyesi:</b> {likert_label}</div>"
            "<div><b>Suç olasılığı (p):</b> {p_event_txt}</div>"
            "<div><b>Beklenen suç sayısı:</b> {expected_txt}</div>"
            "<div class='tt-sep'></div>"
            "<div class='tt-h' style='font-weight:800;'>En olası 3 suç</div>"
            "<div class='tt-li'>• {top1_category}</div>"
            "<div class='tt-li'>• {top2_category}</div>"
            "<div class='tt-li'>• {top3_category}</div>"
            "<div class='tt-sep'></div>"
            "<div class='tt-h' style='font-weight:800;'>Kolluk Notu</div>"
            "<div>{advice_txt}</div>"
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
def render_anlik_risk_haritasi():
    _apply_tooltip_css()

    st.markdown("# 🗺️ Anlık Risk Haritası")

    # Akademik/etik “kural” dili yerine karar-destek dili (hocanın eleştirisine uygun)
    st.caption(
        "San Francisco yerel saatine göre mevcut saat dilimindeki hücreler için göreli risk düzeylerini gösterir. "
        "Çıktılar karar destek amaçlıdır; saha bilgisi ve amir değerlendirmesi ile birlikte yorumlanmalıdır."
    )

    if _IMPORT_SRC_ERR is not None:
        st.error("`src.io_data` modülü import edilemedi. `src/` klasörünü ve dosya yollarını kontrol edin.")
        st.code(repr(_IMPORT_SRC_ERR))
        return

    fc = load_forecast()
    if fc.empty:
        st.error(
            "Forecast verisi bulunamadı/boş.\n\nBeklenen dosyalardan en az biri gerekli:\n"
            + "\n".join([f"- {p}" for p in FC_CANDIDATES[:2]])
        )
        return

    date_col = _pick_col(fc, ["date"])
    hr_col = _pick_col(fc, ["hour_range", "hour_bucket"])
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

    # Legend / açıklama: modern "popover"
    # (harita içi overlay Streamlit+pydeck’te zor; bu çözüm temiz ve kurumsal)
    lik, meta = _compute_likert_quintiles(df_hr)
    q20, q40, q60, q80 = meta["cuts"]
    src_col = meta["source_col"] or "risk_score"

    with st.popover("🎨 Risk Ölçeği", use_container_width=False):
        st.markdown(
            "Risk düzeyleri, **bu tarih ve saat dilimindeki hücre risk skorlarının dağılımına göre** "
            "(eşit dilimler / %20’lik dilimler) sınıflandırılmıştır."
        )
        st.caption(f"Kullanılan risk metriği: **{src_col}**")
        # Yüzdelik dilimler sabit (0–20, 20–40...), kesim değerlerini de ekleyelim:
        # (değerler NaN olabilir; o durumda çizgi göstermeyiz)
        def _qtxt(x):
            return "—" if not np.isfinite(_safe_float(x)) else f"{float(x):.3f}"

        rows = [
            (1, "Çok Düşük", "0–20",  None, _qtxt(q20)),
            (2, "Düşük",     "20–40", _qtxt(q20), _qtxt(q40)),
            (3, "Orta",      "40–60", _qtxt(q40), _qtxt(q60)),
            (4, "Yüksek",    "60–80", _qtxt(q60), _qtxt(q80)),
            (5, "Çok Yüksek","80–100",_qtxt(q80), None),
        ]
        for k, label, pct, lo, hi in rows:
            rgb = LIKERT[k][1]
            rng = f"{lo}–{hi}" if (lo is not None and hi is not None) else (f"≤ {hi}" if lo is None else f"> {lo}")
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; padding:8px 10px; border:1px solid #e2e8f0; border-radius:12px; margin-bottom:8px;">
                  <div style="display:flex; align-items:center; gap:10px;">
                    <div style="width:14px;height:14px;border-radius:4px;background:rgb({rgb[0]},{rgb[1]},{rgb[2]});"></div>
                    <div style="font-weight:700;">{k}</div>
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

    # Harita
    gj_enriched = enrich_geojson(gj, df_hr)
    draw_map(gj_enriched)

    st.caption(
        "İpucu: Hücrelerin üzerine gelerek (hover) detayları görüntüleyebilirsiniz."
    )
