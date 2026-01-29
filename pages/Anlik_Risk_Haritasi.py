# pages/Anlik_Risk_Haritasi.py
# SUTAM — Anlık Risk Haritası (ANLIK • seçim yok)
# - Harita hover: GEOID / risk / p_event / expected / top1-3
# - Ops panel: (1) Saat dilimi genel öneri (Top-N) + (2) GEOID seçilince hücreye özel öneri
# - Legend: harita üstünde popover (dikey değil)

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
# LIKERT (soft, kurumsal)
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
    # "21-24" -> (21,24)
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
# RISK -> 1..5 (önce risk_level/risk_bin varsa onu kullan)
# =============================================================================
def risk_to_likert(df: pd.DataFrame) -> pd.Series:
    # 0) zaten 1-5 varsa
    direct = _pick_col(df, ["risk_likert", "likert", "risk5", "risk_level_5"])
    if direct:
        s = pd.to_numeric(df[direct], errors="coerce").fillna(3).astype(int)
        return s.clip(1, 5)

    # 1) risk_bin varsa (bazı pipeline'larda 1..10 decile gibi olabilir)
    rb = _pick_col(df, ["risk_bin", "risk_decile"])
    if rb:
        s = pd.to_numeric(df[rb], errors="coerce")
        if s.notna().any():
            # 1..10 -> 1..5
            if s.max() > 5:
                # 1-10'u 5'e sıkıştır
                out = np.ceil(s / 2.0)
                return pd.Series(out, index=df.index).fillna(3).astype(int).clip(1, 5)
            return s.fillna(3).astype(int).clip(1, 5)

    # 2) risk_level string varsa
    rl = _pick_col(df, ["risk_level", "level"])
    if rl:
        s = df[rl].astype(str).str.lower()
        mapping = {
            "very_low": 1, "vlow": 1, "low": 2,
            "medium": 3, "mid": 3,
            "high": 4,
            "critical": 5, "very_high": 5, "vhigh": 5
        }
        out = s.map(mapping)
        if out.notna().any():
            return out.fillna(3).astype(int).clip(1, 5)

    # 3) risk_score / p_event ile qcut
    rs = _pick_col(df, ["risk_score", "risk", "p_event", "risk_prob"])
    if rs:
        vals = pd.to_numeric(df[rs], errors="coerce")
        if vals.notna().any():
            try:
                bins = pd.qcut(vals.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
                return bins.astype(int)
            except Exception:
                pass

    return pd.Series([3] * len(df), index=df.index)


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


# =============================================================================
# UI: legend popover (harita üstünde)
# =============================================================================
def render_legend_popover():
    left, right = st.columns([1, 5], vertical_alignment="center")
    with left:
        with st.popover("🎨 Risk Ölçeği", use_container_width=True):
            st.markdown("**5’li Risk Ölçeği**")
            for i in range(1, 6):
                label, rgb = LIKERT[i]
                st.markdown(
                    f"""
                    <div style="display:flex; align-items:center; gap:10px; margin:6px 0;">
                      <div style="width:14px;height:14px;border-radius:5px;background:rgb({rgb[0]},{rgb[1]},{rgb[2]});"></div>
                      <div><b>{i}</b> — {label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.caption("Not: Renkler karar destek amaçlıdır; saha bilgisiyle birlikte yorumlanmalıdır.")
    with right:
        st.caption("Harita üzerinde hücrelerin üstüne gelerek (hover) detayları görebilirsiniz.")


# =============================================================================
# enrich geojson
# =============================================================================
def enrich_geojson(gj: dict, df_hr: pd.DataFrame) -> dict:
    if not gj or df_hr.empty:
        return gj

    df = df_hr.copy()

    # GEOID normalize
    geoid_col = _pick_col(df, ["GEOID", "geoid"])
    df["geoid"] = df[geoid_col].map(digits11) if geoid_col else ""

    # p_event
    pe = _pick_col(df, ["p_event", "risk_prob"])
    df["_p_event"] = pd.to_numeric(df[pe], errors="coerce") if pe else np.nan
    df["p_event_txt"] = df["_p_event"].map(_fmt3)

    # expected
    exp = _pick_col(df, ["expected_count", "expected_crimes"])
    df["_expected"] = pd.to_numeric(df[exp], errors="coerce") if exp else np.nan
    df["expected_txt"] = df["_expected"].map(_fmt_expected_band)

    # top categories
    t1 = _pick_col(df, ["top1_category", "top1_cat", "cat1"])
    t2 = _pick_col(df, ["top2_category", "top2_cat", "cat2"])
    t3 = _pick_col(df, ["top3_category", "top3_cat", "cat3"])

    def _clean(s: pd.Series) -> pd.Series:
        return s.astype(str).replace("nan", "").replace("None", "").fillna("")

    df["top1_category"] = _clean(df[t1]) if t1 else ""
    df["top2_category"] = _clean(df[t2]) if t2 else ""
    df["top3_category"] = _clean(df[t3]) if t3 else ""

    # likert + color
    df["risk_likert"] = risk_to_likert(df)
    df["likert_label"] = df["risk_likert"].map(lambda k: LIKERT.get(int(k), ("Orta", DEFAULT_FILL))[0])
    df["fill_color"] = df["risk_likert"].map(lambda k: LIKERT.get(int(k), ("Orta", DEFAULT_FILL))[1])

    # ✅ tekilleştir
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
        props["fill_color"] = DEFAULT_FILL

        if key and key in dmap.index:
            row = dmap.loc[key]
            props["likert_label"] = str(row.get("likert_label") or "")
            props["p_event_txt"] = str(row.get("p_event_txt") or "—")
            props["expected_txt"] = str(row.get("expected_txt") or "—")
            props["top1_category"] = str(row.get("top1_category") or "")
            props["top2_category"] = str(row.get("top2_category") or "")
            props["top3_category"] = str(row.get("top3_category") or "")
            props["fill_color"] = row.get("fill_color", DEFAULT_FILL)

        feats_out.append({**feat, "properties": props})

    return {**gj, "features": feats_out}


# =============================================================================
# map
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
# OPS: genel + hücre özel
# =============================================================================
def make_ops_general(df_hr: pd.DataFrame, top_n: int = 20) -> dict:
    tmp = df_hr.copy()
    tmp["risk_likert"] = risk_to_likert(tmp)

    exp = _pick_col(tmp, ["expected_count", "expected_crimes"])
    tmp["_exp"] = pd.to_numeric(tmp[exp], errors="coerce").fillna(0.0) if exp else 0.0

    top = tmp.sort_values(["risk_likert", "_exp"], ascending=[False, False]).head(top_n)
    max_l = int(top["risk_likert"].max())
    max_label = LIKERT.get(max_l, ("Orta", DEFAULT_FILL))[0]

    t1 = _pick_col(top, ["top1_category", "top1_cat", "cat1"])
    cats = top[t1].astype(str).replace("nan", "").replace("None", "").tolist() if t1 else []
    top_cats = pd.Series([c for c in cats if c]).value_counts().head(3).index.tolist()

    metrics = {
        "Top hücre": f"{len(top)}",
        "≥4 hücre": f"{int((top['risk_likert']>=4).sum())}",
        "5 (kritik)": f"{int((top['risk_likert']>=5).sum())}",
        "Beklenen toplam": _fmt_expected_band(float(top["_exp"].sum())),
        "Risk medyan": f"{float(np.median(top['risk_likert'])):.1f}",
    }

    bullets = []
    if max_l >= 4:
        bullets += [
            "Sıcak noktalarda görünür devriye yoğunluğu artırılabilir (Top-K hücreler öncelikli).",
            "Transit/ana arter ve yoğun yaya akışlı bölgelerde kısa süreli yoğunlaştırılmış devriye önerilir.",
            "Giriş–çıkış aksları ve kamera kör noktalarında çevrimli devriye değerlendirilebilir.",
        ]
    else:
        bullets += [
            "Rutin görünür devriye ve caydırıcılık odaklı dolaşım önerilir.",
            "Risk artışı görülen mikro-bölgelerde kısa süreli kontrol turu planlanabilir.",
        ]

    if top_cats:
        bullets.append(f"Bu saat diliminde öne çıkan suç türleri: {', '.join(top_cats)}.")
    bullets.append("Not: Çıktılar bağlayıcı değildir; saha bilgisi ve amir değerlendirmesi ile birlikte yorumlanmalıdır.")

    return {"title": f"Kolluk Önerisi (Saat dilimi özeti • en yüksek risk: {max_label})", "metrics": metrics, "bullets": bullets, "top": top}

def make_ops_cell(cell_row: pd.Series) -> dict:
    # Hücreye özel mikro öneri: risk + beklenen + top suçlara göre
    likert = int(cell_row.get("risk_likert", 3))
    label = LIKERT.get(likert, ("Orta", DEFAULT_FILL))[0]

    exp = _safe_float(cell_row.get("_expected", np.nan), np.nan)
    p = _safe_float(cell_row.get("_p_event", np.nan), np.nan)

    t1 = str(cell_row.get("top1_category", "") or "")
    t2 = str(cell_row.get("top2_category", "") or "")
    t3 = str(cell_row.get("top3_category", "") or "")
    cats = [c for c in [t1, t2, t3] if c]

    bullets = []
    if likert >= 5:
        bullets.append("Kritik hücre: kısa süreli yoğunlaştırılmış devriye ve görünür caydırıcılık önerilir.")
        bullets.append("Giriş–çıkış aksları, transit duraklar ve yaya yoğunluğu yüksek noktalara odaklanın.")
    elif likert == 4:
        bullets.append("Yüksek risk: devriye sıklığını artırın; kısa kontrollü turlar planlanabilir.")
    else:
        bullets.append("Orta/düşük risk: rutin devriye yeterli; anomali gözlemi odaklı izleme önerilir.")

    if np.isfinite(exp) and exp >= 2:
        bullets.append("Beklenen olay sayısı görece yüksek: aynı bölgede tekrar eden kontrol turu uygulanabilir.")
    if np.isfinite(p) and p >= 0.9:
        bullets.append("Olay olasılığı yüksek: görünürlük ve hızlı müdahale hazır bulundurulabilir.")

    if cats:
        bullets.append(f"Bu hücrede öne çıkan suç türleri: {', '.join(cats[:3])}.")

    return {"title": f"Seçili Hücre Önerisi (Risk: {label})", "bullets": bullets}


# =============================================================================
# ENTRYPOINT
# =============================================================================
def render_anlik_risk_haritasi():
    st.markdown("# Anlık Risk Haritası")
    st.caption("San Francisco yerel saatine göre mevcut saat dilimindeki risk düzeylerini 5’li ölçekte gösterir (seçim yok).")

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
    sel_date = today if today in dates else max([d for d in dates if d <= today], default=dates[0])

    labels = sorted(fc[hr_col].dropna().astype(str).unique().tolist())
    hr_label = _hour_to_bucket(now_sf.hour, labels) or (labels[0] if labels else None)
    if not hr_label:
        st.error("Forecast içinde hour_range bulunamadı.")
        return

    st.caption(f"SF saati: **{now_sf:%Y-%m-%d %H:%M}** • Tarih: **{pd.Timestamp(sel_date).date()}** • Dilim: **{hr_label}**")

    df_hr = fc[(fc["date_norm"] == sel_date) & (fc[hr_col].astype(str) == str(hr_label))].copy()
    if df_hr.empty:
        st.warning("Bu tarih/saat dilimi için kayıt bulunamadı.")
        return

    # Legend: harita üstünde popover
    render_legend_popover()

    # Map
    gj = load_geojson()
    if not gj:
        st.error(f"GeoJSON bulunamadı: `{GEOJSON_PATH}`")
        return

    # enrich için df_hr içine risk_likert/p/expected gibi alanları da hazırlayalım
    df_hr = df_hr.copy()
    df_hr["risk_likert"] = risk_to_likert(df_hr)

    pe = _pick_col(df_hr, ["p_event", "risk_prob"])
    ex = _pick_col(df_hr, ["expected_count", "expected_crimes"])
    df_hr["_p_event"] = pd.to_numeric(df_hr[pe], errors="coerce") if pe else np.nan
    df_hr["_expected"] = pd.to_numeric(df_hr[ex], errors="coerce") if ex else np.nan

    gj_enriched = enrich_geojson(gj, df_hr)
    draw_map(gj_enriched)

    st.divider()

    # GENEL öneri (Top-N)
    ops = make_ops_general(df_hr, top_n=20)
    st.subheader("👮 " + ops["title"])
    cols = st.columns(5)
    met_keys = list(ops["metrics"].keys())
    for i in range(min(5, len(met_keys))):
        cols[i].metric(met_keys[i], ops["metrics"][met_keys[i]])
    for b in ops["bullets"]:
        st.write("•", b)

    st.divider()

    # HÜCRE ÖZEL (dropdown ile)
    st.subheader("🎯 Seçili Hücre Analizi")
    geoid_col = _pick_col(df_hr, ["GEOID", "geoid"])
    if not geoid_col:
        st.info("Forecast içinde GEOID bulunamadı; hücre seçimi devre dışı.")
        return

    df_hr["geoid"] = df_hr[geoid_col].map(digits11)
    # tekilleştir
    df_uni = df_hr.sort_values(["risk_likert"], ascending=False).drop_duplicates("geoid", keep="first")

    options = df_uni["geoid"].dropna().astype(str).tolist()
    # default: Top-N içinden ilk
    default_geoid = options[0] if options else None

    sel_geoid = st.selectbox(
        "GEOID seç (haritada hover ile gördüğünü buradan seçebilirsin):",
        options=options,
        index=0 if default_geoid else 0,
    )

    row = df_uni[df_uni["geoid"].astype(str) == str(sel_geoid)].iloc[0]

    # Hücre metrikleri
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GEOID", str(sel_geoid))
    c2.metric("Risk (Likert)", str(int(row.get("risk_likert", 3))))
    c3.metric("p_event", _fmt3(row.get("_p_event", np.nan)))
    c4.metric("Beklenen", _fmt_expected_band(row.get("_expected", np.nan)))

    cell_ops = make_ops_cell(row)
    st.markdown("**" + cell_ops["title"] + "**")
    for b in cell_ops["bullets"]:
        st.write("•", b)

    with st.expander("🧪 Debug", expanded=False):
        st.write("Kolonlar:", list(df_hr.columns))
        st.write("risk_likert dağılımı:", df_hr["risk_likert"].value_counts(dropna=False))
        t1 = _pick_col(df_hr, ["top1_category", "top1_cat", "cat1"])
        if t1:
            st.write("top1_category top10:", df_hr[t1].astype(str).value_counts().head(10))
