# pages/Anlik_Risk_Haritasi.py
# SUTAM — Anlık Risk Haritası (Kolluk için sade • seçim yok • SF saatine göre anlık hour_range)
# - Veri: data/forecast_7d.parquet (fallback: deploy/full_fc.parquet)
# - GeoJSON: data/sf_cells.geojson
# - Hover: Risk (Likert 1-5), p_event, expected_count, top1-3 category
# - Alt panel: dinamik kolluk önerisi (Top-N hücre üzerinden) + sayısal kanıt
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
    "data/full_fc.parquet",
]
GEOJSON_PATH = os.getenv("GEOJSON_PATH", "data/sf_cells.geojson")
TARGET_TZ = "America/Los_Angeles"


# =============================================================================
# LIKERT SCALE (5'li) — daha kurumsal/soft tonlar
# =============================================================================
LIKERT = {
    1: ("Çok Düşük",  [56, 189, 137]),
    2: ("Düşük",      [104, 207, 162]),
    3: ("Orta",       [241, 196, 15]),
    4: ("Yüksek",     [235, 147, 80]),
    5: ("Çok Yüksek", [220, 88, 76]),
}
DEFAULT_FILL = [220, 220, 220]


# =============================================================================
# HELPERS
# =============================================================================
def _first_existing(paths: list[str]) -> str | None:
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def digits11(x) -> str:
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

def _fmt_expected_band(x) -> str:
    """Kolluk dili: ~0, ~1–2 gibi"""
    v = _safe_float(x, np.nan)
    if not np.isfinite(v):
        return "—"
    v = max(0.0, v)
    lo = int(np.floor(v))
    hi = int(np.ceil(v))
    return f"~{lo}" if lo == hi else f"~{lo}–{hi}"

def _parse_range(tok: str):
    # "18-21" -> (18,21) end exclusive
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

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k.lower() in cols:
            return cols[k.lower()]
    return None

def _risk_to_likert(df: pd.DataFrame) -> pd.Series:
    # 1) direkt 1-5 kolon varsa
    direct = _pick_col(df, ["risk_likert", "likert", "risk5", "risk_level_5"])
    if direct:
        s = pd.to_numeric(df[direct], errors="coerce").fillna(3).astype(int)
        return s.clip(1, 5)

    # 2) risk_level string varsa
    rlev = _pick_col(df, ["risk_level", "level"])
    if rlev:
        s = df[rlev].astype(str).str.lower()
        mapping = {
            "very_low": 1, "vlow": 1, "low": 2,
            "medium": 3, "mid": 3,
            "high": 4,
            "critical": 5, "very_high": 5, "vhigh": 5,
        }
        out = s.map(mapping)
        if out.notna().any():
            return out.fillna(3).astype(int).clip(1, 5)

    # 3) risk_score / p_event ile qcut(5)
    rs_col = _pick_col(df, ["risk_score", "risk", "p_event", "prob_event", "crime_prob"])
    rs = pd.to_numeric(df[rs_col], errors="coerce") if rs_col else pd.Series([np.nan] * len(df), index=df.index)
    if rs.notna().any():
        try:
            bins = pd.qcut(rs.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
            return bins.astype(int)
        except Exception:
            pass

    return pd.Series([3] * len(df), index=df.index)


# =============================================================================
# UI helpers (modern legend)
# =============================================================================
def render_likert_legend_inline():
    st.markdown(
        """
        <style>
          .likert-wrap{display:flex; gap:10px; flex-wrap:wrap; margin: 10px 0 14px 0;}
          .likert-chip{
            display:flex; align-items:center; gap:10px;
            padding:8px 10px; border:1px solid #e2e8f0; border-radius:999px;
            background:#fff; box-shadow:0 1px 0 rgba(15,23,42,.03);
            font-size:14px;
          }
          .likert-dot{width:14px;height:14px;border-radius:5px;}
          .likert-num{font-weight:700; color:#0f172a;}
          .likert-lab{color:#334155;}
        </style>
        <div class="likert-wrap">
        """,
        unsafe_allow_html=True,
    )

    chips = []
    for i in range(1, 6):
        label, rgb = LIKERT[i]
        chips.append(
            f"""
            <div class="likert-chip">
              <div class="likert-dot" style="background: rgb({rgb[0]},{rgb[1]},{rgb[2]});"></div>
              <div class="likert-num">{i}</div>
              <div class="likert-lab">{label}</div>
            </div>
            """
        )
    st.markdown("".join(chips) + "</div>", unsafe_allow_html=True)


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

    # prepare_forecast gp=None ile destekliyorsa hız için öyle
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


# =============================================================================
# CORE: enrich geojson
# =============================================================================
def enrich_geojson(gj: dict, df_hr: pd.DataFrame) -> dict:
    if not gj or df_hr.empty:
        return gj

    df = df_hr.copy()

    # GEOID normalize
    geoid_col = _pick_col(df, ["GEOID", "geoid"])
    if geoid_col:
        df["geoid"] = df[geoid_col].map(digits11)
    else:
        df["geoid"] = ""

    # p_event: varsa onu, yoksa risk_score, o da yoksa top1_prob
    pe_col = _pick_col(df, ["p_event", "prob_event", "crime_prob"])
    rs_col = _pick_col(df, ["risk_score", "risk"])
    t1p_col = _pick_col(df, ["top1_prob", "top1_probability", "top1_p"])

    if pe_col:
        df["_p_event"] = pd.to_numeric(df[pe_col], errors="coerce")
    elif rs_col:
        df["_p_event"] = pd.to_numeric(df[rs_col], errors="coerce")
    elif t1p_col:
        df["_p_event"] = pd.to_numeric(df[t1p_col], errors="coerce")
    else:
        df["_p_event"] = np.nan

    df["p_event_txt"] = df["_p_event"].map(_fmt3)

    # expected_count: varsa onu, yoksa boş
    exp_col = _pick_col(df, ["expected_count", "exp_count", "lambda", "mu"])
    if exp_col:
        df["_expected"] = pd.to_numeric(df[exp_col], errors="coerce")
    else:
        df["_expected"] = np.nan
    df["expected_txt"] = df["_expected"].map(_fmt_expected_band)

    # top categories
    top1 = _pick_col(df, ["top1_category", "top1_cat", "cat1"])
    top2 = _pick_col(df, ["top2_category", "top2_cat", "cat2"])
    top3 = _pick_col(df, ["top3_category", "top3_cat", "cat3"])

    def _clean_str_series(s: pd.Series) -> pd.Series:
        return s.astype(str).replace("nan", "").replace("None", "").fillna("")

    df["top1_category"] = _clean_str_series(df[top1]) if top1 else ""
    df["top2_category"] = _clean_str_series(df[top2]) if top2 else ""
    df["top3_category"] = _clean_str_series(df[top3]) if top3 else ""

    # risk -> likert
    df["risk_likert"] = _risk_to_likert(df)
    df["likert_label"] = df["risk_likert"].map(lambda k: LIKERT.get(int(k), ("Orta", DEFAULT_FILL))[0])
    df["fill_color"] = df["risk_likert"].map(lambda k: LIKERT.get(int(k), ("Orta", DEFAULT_FILL))[1])

    # ✅ Tekilleştir: aynı GEOID birden çok satır varsa “en riskli + en yüksek expected” seç
    df["_expected_num"] = pd.to_numeric(df["_expected"], errors="coerce").fillna(0.0)
    df = (
        df.sort_values(["risk_likert", "_expected_num"], ascending=[False, False])
          .drop_duplicates("geoid", keep="first")
    )

    dmap = df.set_index("geoid")

    feats_out = []
    for feat in gj.get("features", []):
        props = dict(feat.get("properties") or {})

        # feature GEOID adayları
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
        props["fill_color"] = DEFAULT_FILL

        if key and key in dmap.index:
            row = dmap.loc[key]  # Series
            props["likert_label"] = str(row.get("likert_label", "") or "")
            props["p_event_txt"] = str(row.get("p_event_txt", "—") or "—")
            props["expected_txt"] = str(row.get("expected_txt", "—") or "—")
            props["top1_category"] = str(row.get("top1_category", "") or "")
            props["top2_category"] = str(row.get("top2_category", "") or "")
            props["top3_category"] = str(row.get("top3_category", "") or "")
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
            "<hr style='opacity:0.30'/>"
            "<b>En olası 3 suç:</b>"
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


# =============================================================================
# OPS SUGGESTIONS (dinamik + kanıtlı)
# =============================================================================
def make_ops_suggestions(df_hr: pd.DataFrame, top_n: int = 20) -> dict:
    if df_hr.empty:
        return {"title": "Kolluk Önerisi", "bullets": ["Veri bulunamadı."], "metrics": {}}

    tmp = df_hr.copy()
    tmp["risk_likert"] = _risk_to_likert(tmp)

    # expected / p_event kolonları
    exp_col = _pick_col(tmp, ["expected_count", "exp_count", "lambda", "mu"])
    pe_col  = _pick_col(tmp, ["p_event", "prob_event", "crime_prob"])
    rs_col  = _pick_col(tmp, ["risk_score", "risk"])
    t1p_col = _pick_col(tmp, ["top1_prob", "top1_probability", "top1_p"])

    tmp["_exp"] = pd.to_numeric(tmp[exp_col], errors="coerce").fillna(0.0) if exp_col else 0.0

    if pe_col:
        tmp["_p"] = pd.to_numeric(tmp[pe_col], errors="coerce")
    elif rs_col:
        tmp["_p"] = pd.to_numeric(tmp[rs_col], errors="coerce")
    elif t1p_col:
        tmp["_p"] = pd.to_numeric(tmp[t1p_col], errors="coerce")
    else:
        tmp["_p"] = np.nan

    # Top-N
    top = tmp.sort_values(["risk_likert", "_exp"], ascending=[False, False]).head(top_n)

    max_l = int(top["risk_likert"].max())
    max_label = LIKERT.get(max_l, ("Orta", DEFAULT_FILL))[0]

    # suç dağılımı (top1)
    top1 = _pick_col(top, ["top1_category", "top1_cat", "cat1"])
    cats = top[top1].astype(str).replace("nan", "").replace("None", "").tolist() if top1 else []
    top_cats = pd.Series([c for c in cats if c]).value_counts().head(3).index.tolist()

    # metrik kanıt
    n_cells = int(top["risk_likert"].shape[0])
    n_crit = int((top["risk_likert"] >= 5).sum())
    n_high = int((top["risk_likert"] >= 4).sum())
    exp_sum = float(top["_exp"].sum()) if "_exp" in top.columns else 0.0
    p_med = float(np.nanmedian(top["_p"])) if "_p" in top.columns else np.nan

    metrics = {
        "Top hücre sayısı": f"{n_cells}",
        "Yüksek+ (≥4) hücre": f"{n_high}",
        "Kritik (5) hücre": f"{n_crit}",
        "Top-N beklenen toplam": _fmt_expected_band(exp_sum),
        "Top-N p medyan": _fmt3(p_med),
    }

    bullets = []
    if max_l >= 4:
        bullets.append("Sıcak noktalarda görünür devriye yoğunluğu artırılabilir (Top-K hücreler öncelikli).")
        bullets.append("Transit/ana arter ve yoğun yaya akışlı bölgelerde kısa süreli yoğunlaştırılmış devriye önerilir.")
        bullets.append("Giriş–çıkış aksları ve kamera kör noktalarında çevrimli devriye değerlendirilebilir.")
        if n_crit >= max(3, top_n // 6):
            bullets.append("Kritik hücre yoğunluğu yüksek: kritik bölgelerde devriye süresi artırılabilir.")
    else:
        bullets.append("Rutin görünür devriye ve caydırıcılık odaklı dolaşım önerilir.")
        bullets.append("Risk artışı görülen mikro-bölgelerde kısa süreli kontrol turu planlanabilir.")

    if top_cats:
        bullets.append(f"Bu saat diliminde öne çıkan suç türleri: {', '.join(top_cats)}.")

    bullets.append("Not: Çıktılar bağlayıcı değildir; saha bilgisi ve amir değerlendirmesi ile birlikte yorumlanmalıdır.")
    return {"title": f"Kolluk Önerisi (Bu saat dilimi • en yüksek risk: {max_label})", "bullets": bullets, "metrics": metrics}


# =============================================================================
# PUBLIC ENTRYPOINT (app.py bunu import edip çağıracak)
# =============================================================================
def render_anlik_risk_haritasi():
    st.markdown("# Anlık Risk Haritası")
    st.caption("San Francisco yerel saatine göre mevcut saat dilimindeki risk düzeylerini 5’li ölçekte gösterir (seçim yok).")

    if _IMPORT_SRC_ERR is not None:
        st.error("`src.io_data` modülü import edilemedi. `src/` klasörünü ve dosya yollarını kontrol edin.")
        st.code(repr(_IMPORT_SRC_ERR))
        return

    fc = load_forecast()
    if fc.empty:
        st.error(
            "Forecast verisi bulunamadı/boş.\n\nBeklenen dosyalardan en az biri gerekli:\n"
            + "\n".join([f"- {p}" for p in FC_CANDIDATES[:3]])
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
        st.error("Forecast içinde hour_range bulunamadı.")
        return

    st.caption(
        f"SF saati: **{now_sf:%Y-%m-%d %H:%M}**  •  Tarih: **{pd.Timestamp(sel_date).date()}**  •  Dilim: **{hr_label}**"
    )

    # anlık filtre
    df_hr = fc[(fc["date_norm"] == sel_date) & (fc[hr_col].astype(str) == str(hr_label))].copy()
    if df_hr.empty:
        st.warning("Bu tarih/saat dilimi için kayıt bulunamadı.")
        return

    # risk ölçeği — modern
    render_likert_legend_inline()

    gj = load_geojson()
    if not gj:
        st.error(f"GeoJSON bulunamadı: `{GEOJSON_PATH}` (polygonlar gerekli).")
        return

    gj_enriched = enrich_geojson(gj, df_hr)
    draw_map(gj_enriched)

    st.divider()

    # dinamik kolluk önerisi + metrikler
    ops = make_ops_suggestions(df_hr, top_n=20)
    st.subheader("👮 " + ops["title"])

    m = ops.get("metrics", {}) or {}
    if m:
        cols = st.columns(5)
        keys = list(m.keys())[:5]
        for i, k in enumerate(keys):
            cols[i].metric(k, m[k])

    for b in ops["bullets"]:
        st.write("•", b)

    # İsteğe bağlı debug (gerekirse aç)
    with st.expander("🧪 Debug (ops/dağılım)", expanded=False):
        st.write("risk_likert dağılımı:", _risk_to_likert(df_hr).value_counts(dropna=False).head(10))
        t1 = _pick_col(df_hr, ["top1_category", "top1_cat", "cat1"])
        if t1:
            st.write("top1_category dağılımı:", df_hr[t1].astype(str).value_counts().head(10))
        st.write("Kolonlar:", list(df_hr.columns))
