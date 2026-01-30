# pages/Suc_Zarar_Tahmini.py
# SUTAM — Suç + Zarar (HARM) Tahmini | Operasyonel Karar Destek (Kolluk-Dostu)
# - Veri: data/forecast_7d_ops_harm_ready.csv  (fallback: /mnt/data/forecast_7d_ops_harm_ready.csv)
# - GeoJSON: data/sf_cells.geojson
# - Üstte tarih + saat aralığı seçimi
# - Harita: mevcut GeoJsonLayer mantığı korunur
# - Altta: Kolluğa Öneriler (datasetten ops_* + driver bayrakları + harm/risk özetleri)

from __future__ import annotations

import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk


# =============================================================================
# PATHS / CONSTANTS
# =============================================================================
DATA_DIR = os.getenv("DATA_DIR", "data").rstrip("/")
TARGET_TZ = "America/Los_Angeles"
GEOJSON_PATH = os.getenv("GEOJSON_PATH", f"{DATA_DIR}/sf_cells.geojson")

OPS_HARM_CANDIDATES = [
    "deploy/forecast_7d_ops_ready.parquet",
]

LIKERT = {
    1: ("Çok Düşük",  [46, 204, 113]),
    2: ("Düşük",      [88, 214, 141]),
    3: ("Orta",       [241, 196, 15]),
    4: ("Yüksek",     [230, 126, 34]),
    5: ("Çok Yüksek", [192, 57, 43]),
}
DEFAULT_FILL = [220, 220, 220]


# =============================================================================
# UI CSS (tooltip overflow + compact)
# =============================================================================
def _apply_tooltip_css():
    st.markdown(
        """
        <style>
          .deckgl-tooltip {
            max-width: 360px !important;
            max-height: 360px !important;
            overflow: auto !important;
            padding: 10px 12px !important;
            line-height: 1.25 !important;
            border-radius: 12px !important;
            box-shadow: 0 10px 30px rgba(0,0,0,.25) !important;
            transform: translate(12px, 12px) !important;
          }
          .deckgl-tooltip hr { margin: 8px 0 !important; opacity: .25 !important; }
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

def _fmt_moneyish(x) -> str:
    v = _safe_float(x, np.nan)
    if not np.isfinite(v):
        return "—"
    # harm sayısı para olmayabilir; ama okunabilir kalsın:
    if v >= 1000:
        return f"{v:,.0f}"
    if v >= 100:
        return f"{v:.0f}"
    return f"{v:.1f}"

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

def _hour_range_sort_key(label: str) -> int:
    rg = _parse_range(str(label))
    return rg[0] if rg else 0

def _compute_likert_quintiles(values: pd.Series) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    if v.notna().sum() < 10:
        return pd.Series([3] * len(v), index=v.index)
    try:
        bins = pd.qcut(v.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        return bins.astype(int)
    except Exception:
        qs = v.quantile([0.2, 0.4, 0.6, 0.8]).values.tolist()
        q20, q40, q60, q80 = qs
        lik = pd.Series(3, index=v.index)
        lik[v <= q20] = 1
        lik[(v > q20) & (v <= q40)] = 2
        lik[(v > q40) & (v <= q60)] = 3
        lik[(v > q60) & (v <= q80)] = 4
        lik[v > q80] = 5
        return lik.astype(int)


# =============================================================================
# LOADERS
# =============================================================================
@st.cache_data(show_spinner=False)
def load_ops_harm_forecast() -> pd.DataFrame:
    p = _first_existing(OPS_HARM_CANDIDATES)
    if not p:
        return pd.DataFrame()

    try:
        # parquet mi?
        if str(p).lower().endswith(".parquet"):
            df = pd.read_parquet(p)   # pyarrow gerektirir (requirements'te var)
        else:
            df = pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # --- normalize date ---
    dc = _pick_col(df, ["date"])
    if dc:
        df[dc] = pd.to_datetime(df[dc], errors="coerce")
        df["date_norm"] = df[dc].dt.normalize()
    else:
        df["date_norm"] = pd.NaT

    # --- normalize hour_range ---
    hc = _pick_col(df, ["hour_range", "hour_bucket"])
    df["hour_range_norm"] = df[hc].astype(str) if hc else ""

    # --- normalize geoid ---
    gc = _pick_col(df, ["GEOID", "geoid"])
    df["geoid_norm"] = df[gc].map(_digits11) if gc else ""

    return df

@st.cache_data(show_spinner=False)
def load_geojson() -> dict:
    if os.path.exists(GEOJSON_PATH):
        with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# =============================================================================
# AGGREGATION (date+hour_range aralığı seçilirse)
# =============================================================================
def _aggregate_for_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aralık seçildiğinde (birden fazla slot), GEOID bazında özet üretir.
    - expected_count & expected_harm: SUM (operasyon planlama)
    - p_event / risk_score: MEAN
    - top categories: en çok görülen (mode) (pratik ve hızlı)
    - ops_*: en sık görülen kısa aksiyon (mode) + uzun metinlerden ilk dolu
    """
    if df.empty:
        return df

    exp_col = _pick_col(df, ["expected_count", "expected_crimes"])
    p_col   = _pick_col(df, ["p_event", "risk_prob", "prob_event"])
    rs_col  = _pick_col(df, ["risk_score"])
    harm_col = _pick_col(df, ["expected_harm", "harm_expected", "harm"])

    def _mode(s: pd.Series):
        s = s.dropna().astype(str)
        s = s[s != "nan"]
        if s.empty:
            return ""
        return s.value_counts().index[0]

    group_cols = ["geoid_norm"]

    agg = pd.DataFrame(index=df["geoid_norm"].unique())
    agg.index.name = "geoid_norm"
    agg = agg.reset_index()

    if exp_col:
        agg["expected_count"] = df.groupby(group_cols)[exp_col].apply(lambda x: pd.to_numeric(x, errors="coerce").fillna(0).sum()).values
    else:
        agg["expected_count"] = 0.0

    if harm_col:
        agg["expected_harm"] = df.groupby(group_cols)[harm_col].apply(lambda x: pd.to_numeric(x, errors="coerce").fillna(0).sum()).values
    else:
        agg["expected_harm"] = 0.0

    if p_col:
        agg["p_event"] = df.groupby(group_cols)[p_col].apply(lambda x: pd.to_numeric(x, errors="coerce").mean()).values
    else:
        agg["p_event"] = np.nan

    if rs_col:
        agg["risk_score"] = df.groupby(group_cols)[rs_col].apply(lambda x: pd.to_numeric(x, errors="coerce").mean()).values
    else:
        agg["risk_score"] = np.nan

    # top categories
    for i in (1, 2, 3):
        c = _pick_col(df, [f"top{i}_category", f"top{i}_cat"])
        agg[f"top{i}_category"] = df.groupby(group_cols)[c].apply(_mode).values if c else ""

    # ops text
    act_short = _pick_col(df, ["ops_actions_short"])
    rea_long  = _pick_col(df, ["ops_reasons_long", "ops_reasons"])
    act_long  = _pick_col(df, ["ops_actions_long", "ops_actions"])

    agg["ops_actions_short"] = df.groupby(group_cols)[act_short].apply(_mode).values if act_short else ""
    # uzunlar: ilk dolu
    def _first_nonempty(s: pd.Series):
        s = s.dropna().astype(str)
        s = s[(s != "nan") & (s.str.strip() != "")]
        return s.iloc[0] if len(s) else ""

    agg["ops_reasons_long"] = df.groupby(group_cols)[rea_long].apply(_first_nonempty).values if rea_long else ""
    agg["ops_actions_long"] = df.groupby(group_cols)[act_long].apply(_first_nonempty).values if act_long else ""

    return agg


# =============================================================================
# GEOJSON ENRICH (harita aynı mantık)
# =============================================================================
def enrich_geojson(gj: dict, df_slot: pd.DataFrame, metric_for_color: str) -> dict:
    if not gj or df_slot.empty:
        return gj

    df = df_slot.copy()

    # geoid
    if "geoid_norm" not in df.columns:
        gc = _pick_col(df, ["GEOID", "geoid"])
        df["geoid_norm"] = df[gc].map(_digits11) if gc else ""

    # columns
    p_col    = _pick_col(df, ["p_event", "risk_prob", "prob_event"])
    exp_col  = _pick_col(df, ["expected_count", "expected_crimes"])
    harm_col = _pick_col(df, ["expected_harm", "harm_expected", "harm"])
    ops_col  = _pick_col(df, ["ops_actions_short", "ops_actions"])

    # text fields
    df["p_event_txt"]   = pd.to_numeric(df[p_col], errors="coerce").map(_fmt3) if p_col else "—"
    df["expected_txt"]  = pd.to_numeric(df[exp_col], errors="coerce").map(_fmt_expected) if exp_col else "—"
    df["harm_txt"]      = pd.to_numeric(df[harm_col], errors="coerce").map(_fmt_moneyish) if harm_col else "—"
    df["ops_note_txt"]  = df[ops_col].astype(str).replace("nan", "").fillna("") if ops_col else ""

    for i in (1, 2, 3):
        c = _pick_col(df, [f"top{i}_category", f"top{i}_cat"])
        df[f"top{i}_category"] = df[c].astype(str).replace("nan", "").fillna("") if c else ""

    # color metric -> likert
    if metric_for_color == "harm":
        src = harm_col or "expected_harm"
        values = pd.to_numeric(df.get(src, np.nan), errors="coerce")
    else:
        # risk
        rs_col = _pick_col(df, ["risk_score"]) or p_col or exp_col
        values = pd.to_numeric(df.get(rs_col, np.nan), errors="coerce")

    df["risk_likert"] = _compute_likert_quintiles(values).clip(1, 5)
    df["likert_label"] = df["risk_likert"].map(lambda k: LIKERT[int(k)][0])
    df["fill_color"] = df["risk_likert"].map(lambda k: LIKERT[int(k)][1])

    # duplicates: keep highest likert then highest expected_count
    df["_exp_num"] = pd.to_numeric(df[exp_col], errors="coerce").fillna(0.0) if exp_col else 0.0
    df = (
        df.sort_values(["risk_likert", "_exp_num"], ascending=[False, False])
          .drop_duplicates("geoid_norm", keep="first")
    )
    dmap = df.set_index("geoid_norm")

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
        props["harm_txt"] = "—"
        props["top1_category"] = ""
        props["top2_category"] = ""
        props["top3_category"] = ""
        props["ops_note_txt"] = ""
        props["fill_color"] = DEFAULT_FILL

        if key and key in dmap.index:
            row = dmap.loc[key]
            props["likert_label"] = str(row.get("likert_label", "") or "")
            props["p_event_txt"] = str(row.get("p_event_txt", "—") or "—")
            props["expected_txt"] = str(row.get("expected_txt", "—") or "—")
            props["harm_txt"] = str(row.get("harm_txt", "—") or "—")
            props["top1_category"] = str(row.get("top1_category", "") or "")
            props["top2_category"] = str(row.get("top2_category", "") or "")
            props["top3_category"] = str(row.get("top3_category", "") or "")
            props["ops_note_txt"] = str(row.get("ops_note_txt", "") or "")
            props["fill_color"] = row.get("fill_color", DEFAULT_FILL)

        feats_out.append({**feat, "properties": props})

    return {**gj, "features": feats_out}


# =============================================================================
# MAP (aynı yapı)
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

    tooltip = {
        "html": (
            "<div style='font-weight:900; font-size:14px;'>GEOID: {display_id}</div>"
            "<div><b>Risk Seviyesi:</b> {likert_label}</div>"
            "<div><b>Suç olasılığı (p):</b> {p_event_txt}</div>"
            "<div><b>Beklenen suç:</b> {expected_txt}</div>"
            "<div><b>Beklenen zarar (HARM):</b> {harm_txt}</div>"
            "<hr/>"
            "<div style='font-weight:900;'>En olası 3 suç</div>"
            "<div>• {top1_category}</div>"
            "<div>• {top2_category}</div>"
            "<div>• {top3_category}</div>"
            "<hr/>"
            "<div style='font-weight:900;'>Kolluk Notu</div>"
            "<div>{ops_note_txt}</div>"
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
# KOLLUĞA ÖNERİLER PANELİ
# =============================================================================
def _render_ops_panel(df_slot: pd.DataFrame, title_suffix: str = ""):
    if df_slot.empty:
        st.info("Öneri üretmek için veri yok.")
        return

    exp_col = _pick_col(df_slot, ["expected_count", "expected_crimes"]) or "expected_count"
    harm_col = _pick_col(df_slot, ["expected_harm", "harm_expected", "harm"]) or "expected_harm"
    p_col = _pick_col(df_slot, ["p_event", "risk_prob", "prob_event"]) or "p_event"

    # Top-N riskli hücreler: expected_harm (yoksa expected_count)
    metric = harm_col if harm_col in df_slot.columns else exp_col
    tmp = df_slot.copy()
    tmp["_metric"] = pd.to_numeric(tmp.get(metric, 0), errors="coerce").fillna(0.0)
    tmp["_exp"] = pd.to_numeric(tmp.get(exp_col, 0), errors="coerce").fillna(0.0)
    tmp["_p"] = pd.to_numeric(tmp.get(p_col, np.nan), errors="coerce")

    topn = (
        tmp.sort_values(["_metric", "_exp"], ascending=[False, False])
           .head(12)
           .copy()
    )

    ops_short = _pick_col(tmp, ["ops_actions_short"]) or _pick_col(tmp, ["ops_actions"]) or None
    rea_long = _pick_col(tmp, ["ops_reasons_long", "ops_reasons"]) or None
    act_long = _pick_col(tmp, ["ops_actions_long", "ops_actions"]) or None

    st.markdown(f"### 👮 Kolluğa Öneriler{title_suffix}")

    # 1) Genel özet
    total_exp = float(tmp["_exp"].sum())
    total_harm = float(pd.to_numeric(tmp.get(harm_col, 0), errors="coerce").fillna(0.0).sum()) if harm_col in tmp.columns else np.nan
    mean_p = float(tmp["_p"].mean()) if np.isfinite(tmp["_p"].mean()) else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("Toplam beklenen suç", f"{total_exp:,.1f}")
    c2.metric("Toplam beklenen zarar (HARM)", "—" if not np.isfinite(total_harm) else f"{total_harm:,.0f}")
    c3.metric("Ortalama p", "—" if not np.isfinite(mean_p) else f"{mean_p:.3f}")

    # 2) Top-N listesi (kısa)
    show_cols = ["geoid_norm", "_metric", "_exp", "_p"]
    view = topn[show_cols].rename(
        columns={
            "geoid_norm": "GEOID",
            "_metric": "Öncelik (HARM/Count)",
            "_exp": "Beklenen suç",
            "_p": "p",
        }
    )

    st.caption("Öncelik listesi: seçilen zaman aralığında en yüksek beklenen zarar (yoksa beklenen suç) üreten hücreler.")
    st.dataframe(view, use_container_width=True, hide_index=True)

    # 3) Hücre bazlı detay (kolluk metinleri)
    geoids = topn["geoid_norm"].astype(str).unique().tolist()
    if geoids:
        pick = st.selectbox("Detay görmek için GEOID seç", geoids, index=0)
        row = tmp[tmp["geoid_norm"].astype(str) == str(pick)].head(1)
        if not row.empty:
            r = row.iloc[0]
            st.markdown("#### Seçili hücre — operasyon özeti")
            st.write(
                f"- **GEOID:** {pick}\n"
                f"- **p:** {_fmt3(r.get(p_col))}\n"
                f"- **Beklenen suç:** {_fmt_expected(r.get(exp_col))}\n"
                f"- **Beklenen HARM:** {_fmt_moneyish(r.get(harm_col)) if harm_col in tmp.columns else '—'}"
            )

            if ops_short:
                txt = str(r.get(ops_short, "") or "").strip()
                if txt and txt.lower() != "nan":
                    st.info(txt)

            with st.expander("Gerekçe ve aksiyonlar (detay)", expanded=True):
                if rea_long:
                    rl = str(r.get(rea_long, "") or "").strip()
                    if rl and rl.lower() != "nan":
                        st.markdown("**Neden (dataset):**")
                        st.write(rl)
                if act_long:
                    al = str(r.get(act_long, "") or "").strip()
                    if al and al.lower() != "nan":
                        st.markdown("**Önerilen aksiyon (dataset):**")
                        st.write(al)

            # Driver bayrakları (varsa)
            flags = []
            for col in ["calls_flag", "neighbor_flag", "transit_flag", "poi_flag", "weather_flag", "time_flag", "impact_flag"]:
                c = _pick_col(tmp, [col])
                if c and str(r.get(c, "")).strip() not in ("", "nan", "None"):
                    v = str(r.get(c))
                    if v.lower() in ("1", "true", "yes"):
                        flags.append(col.replace("_flag", ""))
            if flags:
                st.caption("Baskın sinyal alanları: " + ", ".join(flags))


# =============================================================================
# PAGE ENTRYPOINT
# =============================================================================
def render_suc_zarar_tahmini():
    _apply_tooltip_css()

    st.markdown("# 🧭 Suç + Zarar (HARM) Tahmini")
    st.caption(
        "Operasyonel karar destek amaçlıdır. Zaman aralığı seçerek haritayı ve kolluk önerilerini aynı ekranda görürsünüz."
    )

    df = load_ops_harm_forecast()
    if df.empty:
        st.error(
            "Ops+Harm forecast CSV bulunamadı/boş.\n\nBeklenen dosyalardan biri:\n"
            + "\n".join([f"- {p}" for p in OPS_HARM_CANDIDATES])
        )
        return

    gj = load_geojson()
    if not gj:
        st.error(f"GeoJSON bulunamadı: `{GEOJSON_PATH}`")
        return

    # --- seçimler ---
    now_sf = datetime.now(ZoneInfo(TARGET_TZ))

    dates = sorted([d for d in df["date_norm"].dropna().unique()])
    if not dates:
        st.error("Veride geçerli `date` bulunamadı.")
        return

    hr_labels = sorted(df["hour_range_norm"].dropna().unique().tolist(), key=_hour_range_sort_key)
    if not hr_labels:
        st.error("Veride geçerli `hour_range` bulunamadı.")
        return

    # varsayılanlar: SF now'a en yakın
    today = pd.Timestamp(now_sf.date()).normalize()
    default_date = today if today in dates else dates[0]
    default_hr = None
    for lab in hr_labels:
        rg = _parse_range(lab)
        if rg and (rg[0] <= now_sf.hour < rg[1]):
            default_hr = lab
            break
    default_hr = default_hr or hr_labels[0]

    st.markdown("### ⏱️ Zaman seçimi")

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        start_date = st.selectbox("Başlangıç tarih", dates, index=dates.index(default_date))
    with c2:
        end_date = st.selectbox("Bitiş tarih", dates, index=dates.index(default_date))
    # sıralama düzelt
    if end_date < start_date:
        start_date, end_date = end_date, start_date

    with c3:
        start_hr = st.selectbox("Başlangıç dilim", hr_labels, index=hr_labels.index(default_hr))
    with c4:
        end_hr = st.selectbox("Bitiş dilim", hr_labels, index=hr_labels.index(default_hr))

    # dilim aralığını set'e çevir
    hr_labels_sorted = hr_labels[:]  # already sorted
    i1 = hr_labels_sorted.index(start_hr)
    i2 = hr_labels_sorted.index(end_hr)
    if i2 < i1:
        i1, i2 = i2, i1
    hr_pick = set(hr_labels_sorted[i1:i2 + 1])

    # metrik seçimi (renk)
    metric_for_color = st.radio(
        "Harita renklendirme metriği",
        options=["Risk (göreli)", "Zarar/HARM (göreli)"],
        horizontal=True,
        index=0,
    )
    metric_key = "harm" if "HARM" in metric_for_color else "risk"

    st.caption(
        f"SF saati: **{now_sf:%Y-%m-%d %H:%M}**  •  Seçim: **{pd.Timestamp(start_date).date()} → {pd.Timestamp(end_date).date()}** "
        f"ve **{hr_labels_sorted[i1]} → {hr_labels_sorted[i2]}**"
    )

    # --- filtre ---
    mask = (
        (df["date_norm"] >= pd.Timestamp(start_date)) &
        (df["date_norm"] <= pd.Timestamp(end_date)) &
        (df["hour_range_norm"].astype(str).isin(hr_pick))
    )
    df_sel = df[mask].copy()
    if df_sel.empty:
        st.warning("Bu tarih/saat aralığı için kayıt bulunamadı.")
        return

    # tek slot mu? (tek gün ve tek dilim)
    is_single_slot = (pd.Timestamp(start_date) == pd.Timestamp(end_date)) and (len(hr_pick) == 1)

    # slot -> harita dataframe
    if is_single_slot:
        df_map = df_sel.copy()
        title_suffix = " (tek zaman dilimi)"
    else:
        df_map = _aggregate_for_range(df_sel)
        title_suffix = " (seçilen aralığın özeti)"

    # --- harita ---
    st.markdown("### 🗺️ Harita")
    gj_enriched = enrich_geojson(gj, df_map, metric_for_color=metric_key)
    draw_map(gj_enriched)

    st.caption("İpucu: Hücrelerin üzerine gelerek (hover) hem suç hem HARM hem de kolluk notunu görebilirsiniz.")

    # --- kolluğa öneriler ---
    with st.expander("👮 Kolluğa Öneriler", expanded=True):
        _render_ops_panel(df_map, title_suffix=title_suffix)
