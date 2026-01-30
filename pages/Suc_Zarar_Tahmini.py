# pages/3_🧭_Suc_Tahmini_ve_Zarar.py
# SUTAM — Suç Tahmini & Suç Zarar Tahmini (TEK SAYFA • TEK HARİTA • Katman seçimi)
# Kaynak (yerel): deploy/full_fc.parquet + deploy/geoid_profile.parquet + data/sf_cells.geojson
# - Katman: "Suç Riski" (risk_prob / p_event)  veya  "Suç Zarar" (expected_harm)
# - Harita renklendirme: seçili katmana göre 5 seviye (Likert)
#   * Risk: 0–1 sabit eşikler (0.2/0.4/0.6/0.8)
#   * Zarar: seçili pencere içinde quintile (göreli) eşikler
# - Kolluk önerileri: seçili katmana göre otomatik değişir
# Not: fr_crime_10.. yok; SF dosyalarıyla çalışır.

from __future__ import annotations

import os, json
from datetime import datetime, timedelta

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

# -----------------------------
# 0) SAYFA AYAR
# -----------------------------
st.title("🌀 Suç Tahmini & Suç Zarar Tahmini — Haritalı GEOID görünüm")
st.caption("Tek harita • Katman seçimi (Risk/Zarar) • Aynı filtreler (tarih/saat/GEOID)")

# -----------------------------
# 1) DOSYA YOLLARI
# -----------------------------
FULL_FC_CANDIDATES = [
    "deploy/full_fc.parquet",
    "full_fc.parquet",
    "data/full_fc.parquet",
]

GEOID_PROFILE_CANDIDATES = [
    "deploy/geoid_profile.parquet",
    "geoid_profile.parquet",
    "data/geoid_profile.parquet",
]

GEOJSON_LOCAL = "data/sf_cells.geojson"

# -----------------------------
# 2) HARM WEIGHTS (opsiyonel, tooltip/öneri metni için)
#    full_fc zaten expected_harm üretiyor; bu sözlük sadece yardımcı.
# -----------------------------
HARM_W = {
    "Arson": 70.0, "Assault": 70.0, "Burglary": 45.0, "Case Closure": 0.0,
    "Civil Sidewalks": 5.0, "Courtesy Report": 0.0, "Disorderly Conduct": 10.0,
    "Drug Offense": 55.0, "Drug Violation": 50.0, "Embezzlement": 35.0,
    "Fire Report": 5.0, "Forgery And Counterfeiting": 30.0, "Fraud": 30.0,
    "Gambling": 15.0, "Homicide": 100.0,
    "Human Trafficking (A), Commercial Sex Acts": 90.0,
    "Human Trafficking, Commercial Sex Acts": 90.0,
    "Human Trafficking, Involuntary Servitude": 90.0,
    "Larceny Theft": 30.0, "Liquor Laws": 10.0, "Lost Property": 5.0,
    "Malicious Mischief": 20.0, "Miscellaneous Investigation": 5.0,
    "Missing Person": 15.0, "Motor Vehicle Theft": 40.0, "Non-Criminal": 0.0,
    "Offences Against The Family And Children": 65.0,
    "Other": 10.0, "Other Miscellaneous": 10.0, "Other Offenses": 10.0,
    "Prostitution": 40.0, "Rape": 95.0, "Recovered Vehicle": 0.0,
    "Robbery": 80.0, "Sex Offense": 80.0, "Stolen Property": 35.0,
    "Suicide": 60.0, "Suspicious": 10.0, "Suspicious Occ": 10.0,
    "Traffic Collision": 15.0, "Traffic Violation Arrest": 20.0,
    "Vandalism": 20.0, "Vehicle Impounded": 5.0, "Vehicle Misplaced": 5.0,
    "Warrant": 10.0, "Weapons Carrying Etc": 60.0, "Weapons Offence": 60.0,
    "Weapons Offense": 60.0,
}
UNK_W = 10.0

# -----------------------------
# 3) HELPERS
# -----------------------------
def _pick_existing(paths: list[str]) -> str | None:
    return next((p for p in paths if os.path.exists(p)), None)

def _sf_now() -> datetime:
    if ZoneInfo is not None:
        return datetime.now(ZoneInfo("America/Los_Angeles"))
    return datetime.utcnow()

def _digits11(x) -> str:
    s = "".join(ch for ch in str(x) if ch.isdigit())
    return s.zfill(11) if s else ""

def normalize_geoid_for_map(df: pd.DataFrame, col="GEOID") -> pd.DataFrame:
    df = df.copy()
    if col not in df.columns:
        return df
    df[col] = df[col].astype(str).str.replace(".0", "", regex=False).str.zfill(11)
    return df

def rgba_to_hex(rgba):
    try:
        r, g, b, _ = rgba
        return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))
    except Exception:
        return "#dddddd"

# Risk Likert (sabit eşikler)
RISK_BUCKETS = [
    (0.00, 0.20, "Çok Düşük", [220, 220, 220, 160]),
    (0.20, 0.40, "Düşük",     [56, 168, 0, 200]),
    (0.40, 0.60, "Orta",      [255, 221, 0, 210]),
    (0.60, 0.80, "Yüksek",    [255, 140, 0, 220]),
    (0.80, 1.01, "Çok Yüksek",[160, 0, 0, 240]),
]
COLOR_MAP = {name: rgba for _, _, name, rgba in RISK_BUCKETS}

def bucket_of_fixed01(v: float) -> str:
    x = 0.0 if pd.isna(v) else float(v)
    x = max(0.0, min(1.0, x))
    for lo, hi, name, _ in RISK_BUCKETS:
        if lo <= x < hi:
            return name
    return "Çok Düşük"

def make_quintile_bucketizer(vals: pd.Series):
    # vals: numeric series
    v = pd.to_numeric(vals, errors="coerce").dropna()
    if len(v) < 20:
        # veri azsa sabit 0–1 gibi davranmayalım; lineer 5 böl
        def _b(x):
            if pd.isna(x): return "Çok Düşük"
            x = float(x)
            if x <= 0: return "Çok Düşük"
            # kaba log ölçek benzeri
            return "Orta"
        return _b, {"q20": np.nan, "q40": np.nan, "q60": np.nan, "q80": np.nan}

    q20, q40, q60, q80 = v.quantile([0.2, 0.4, 0.6, 0.8]).tolist()

    def _bucket(x):
        if pd.isna(x): return "Çok Düşük"
        x = float(x)
        if x <= q20: return "Çok Düşük"
        if x <= q40: return "Düşük"
        if x <= q60: return "Orta"
        if x <= q80: return "Yüksek"
        return "Çok Yüksek"

    return _bucket, {"q20": q20, "q40": q40, "q60": q60, "q80": q80}

@st.cache_data(show_spinner=False)
def load_full_fc() -> pd.DataFrame:
    path = _pick_existing(FULL_FC_CANDIDATES)
    if path is None:
        raise FileNotFoundError("full_fc.parquet bulunamadı. deploy/full_fc.parquet bekleniyor.")
    df = pd.read_parquet(path)

    # normalize
    df = df.copy()
    # tarih
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")

    # GEOID
    if "GEOID" in df.columns:
        df = normalize_geoid_for_map(df, "GEOID")
    else:
        raise ValueError("full_fc içinde GEOID kolonu yok.")

    # hour_range -> hour_start
    if "hour_range" in df.columns:
        def parse_h0(s):
            if pd.isna(s): return np.nan
            t = str(s).strip().replace("–", "-").replace("—", "-")
            if "-" not in t: return np.nan
            a = t.split("-", 1)[0].strip()
            try: return int(a)
            except: return np.nan
        df["hour_start"] = pd.to_numeric(df["hour_range"].map(parse_h0), errors="coerce")
        df["hour_start"] = df["hour_start"].clip(0, 23).astype("Int64")
    else:
        df["hour_start"] = pd.NA

    # timestamp (3h slot başlangıcı)
    df["timestamp"] = df["date"] + pd.to_timedelta(df["hour_start"].fillna(0).astype(int), unit="h")

    # risk_prob yoksa p_event dene
    if "risk_prob" not in df.columns and "p_event" in df.columns:
        df["risk_prob"] = pd.to_numeric(df["p_event"], errors="coerce")

    # expected_harm yoksa, expected_count * avg_harm_per_crime (varsa)
    if "expected_harm" not in df.columns:
        if "expected_count" in df.columns and "avg_harm_per_crime" in df.columns:
            ec = pd.to_numeric(df["expected_count"], errors="coerce")
            ah = pd.to_numeric(df["avg_harm_per_crime"], errors="coerce")
            df["expected_harm"] = ec * ah
        else:
            df["expected_harm"] = np.nan

    # expected_count yoksa expected_crimes dene
    if "expected_count" not in df.columns and "expected_crimes" in df.columns:
        df["expected_count"] = pd.to_numeric(df["expected_crimes"], errors="coerce")

    return df

@st.cache_data(show_spinner=False)
def load_geoid_profile() -> pd.DataFrame:
    path = _pick_existing(GEOID_PROFILE_CANDIDATES)
    if path is None:
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "GEOID" in df.columns:
        df = normalize_geoid_for_map(df, "GEOID")
    return df

@st.cache_data(show_spinner=False)
def load_geojson() -> dict:
    if os.path.exists(GEOJSON_LOCAL):
        with open(GEOJSON_LOCAL, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def enrich_geojson_with_metric(gj: dict, agg_df: pd.DataFrame, metric_key: str) -> dict:
    """
    agg_df: GEOID + metric_value + bucket + tooltip fields
    metric_key: 'metric_value' label
    """
    if not gj or agg_df is None or agg_df.empty:
        return gj

    m = agg_df.set_index("GEOID")

    feats = []
    for feat in gj.get("features", []):
        props = dict(feat.get("properties") or {})

        raw = None
        for k in ("GEOID", "geoid", "id", "cell_id", "geoid11", "geoid_norm"):
            if k in props:
                raw = props[k]
                break
        if raw is None:
            for k, v in props.items():
                if "geoid" in str(k).lower():
                    raw = v
                    break

        key = _digits11(raw)
        props["geoid_norm"] = key
        props["display_id"] = str(raw) if raw not in (None, "") else key

        # defaults
        props.setdefault("bucket", "")
        props.setdefault("metric_txt", "")
        props.setdefault("risk_txt", "")
        props.setdefault("harm_txt", "")
        props.setdefault("top1_category", "")
        props.setdefault("expected_txt", "")
        props.setdefault("fill_color", [220, 220, 220, 160])

        if key and key in m.index:
            row = m.loc[key]

            props["bucket"] = str(row.get("bucket", ""))

            rgba = COLOR_MAP.get(props["bucket"], [220, 220, 220, 160])
            props["fill_color"] = rgba

            def f3(x):
                try:
                    return f"{float(x):.3f}"
                except Exception:
                    return ""

            props["metric_txt"] = f3(row.get("metric_value", np.nan))
            props["risk_txt"] = f3(row.get("risk_mean", np.nan))
            props["harm_txt"] = f3(row.get("harm_mean", np.nan))
            props["expected_txt"] = f3(row.get("expected_mean", np.nan))
            props["top1_category"] = str(row.get("top1_category", "") or "")

        feats.append({**feat, "properties": props})

    return {**gj, "features": feats}

def build_ops_reco(selected_layer: str, top_row: pd.Series) -> list[str]:
    """
    selected_layer: "Risk (Suç Olasılığı)" or "Zarar (Expected Harm)"
    top_row: Top-1 GEOID özet satırı (agg)
    """
    out = []
    geoid = str(top_row.get("GEOID", ""))
    bucket = str(top_row.get("bucket", ""))
    risk = top_row.get("risk_mean", np.nan)
    harm = top_row.get("harm_mean", np.nan)
    expected = top_row.get("expected_mean", np.nan)
    topcat = str(top_row.get("top1_category", "") or "")

    if selected_layer.startswith("Risk"):
        out.append(f"**Öncelik:** Seçili pencerede **suç olasılığı** en yüksek hücrelerden biri (**{bucket}**).")
        if pd.notna(risk):
            out.append(f"Bu GEOID için ortalama suç olasılığı ≈ **{float(risk):.3f}**.")
        if pd.notna(expected):
            out.append(f"Beklenen olay sayısı ≈ **{float(expected):.3f}**.")
        if topcat:
            out.append(f"En olası suç türü: **{topcat}**.")
        out.append("**Uygulama:** Görünür devriye, hızlı müdahale rotası, kısa süreli hotspot kontrolü.")
    else:
        out.append(f"**Öncelik:** Seçili pencerede **beklenen zarar (O-CHF)** en yüksek hücrelerden biri (**{bucket}**).")
        if pd.notna(harm):
            out.append(f"Bu GEOID için ortalama beklenen zarar ≈ **{float(harm):.3f}**.")
        if pd.notna(expected):
            out.append(f"Beklenen olay sayısı ≈ **{float(expected):.3f}** (zarar yüksekliği öncelik sıralamasını değiştirir).")
        if topcat:
            out.append(f"Zararı yükselten baskın suç türü: **{topcat}**.")
        out.append("**Uygulama:** Caydırıcılık odaklı görünürlük, kritik noktalara yönlendirme, yüksek-zarar türlerine karşı hedefli önlem.")
    return out

# -----------------------------
# 4) SIDEBAR — AYARLAR
# -----------------------------
st.sidebar.header("⚙️ Ayarlar")

layer = st.sidebar.radio(
    "Harita katmanı (tek seçim)",
    ["Risk (Suç Olasılığı)", "Zarar (Expected Harm)"],
    index=0,
)

# 3 saatlik bloklar (≤7 gün) ve günlük (≤365 gün) — tek sayfada ikisi de
mode = st.sidebar.radio(
    "Zaman çözünürlüğü",
    ["3 Saatlik Bloklar (≤7 gün)", "Günlük (≤365 gün)"],
    index=0,
)

# Saatlik modda saat aralığı seçimi
def default_hour_block_label(hour_blocks: dict) -> str:
    fallback = "18–21"
    try:
        now_sf = _sf_now()
        h = now_sf.hour
        for label, (h0, h1) in hour_blocks.items():
            if h0 <= h <= h1:
                return label
        return fallback
    except Exception:
        return fallback

selected_hours = []
if mode.startswith("3 Saatlik"):
    st.sidebar.subheader("Saat Aralığı (3 saatlik)")
    hour_blocks = {
        "00–03": (0, 2),
        "03–06": (3, 5),
        "06–09": (6, 8),
        "09–12": (9, 11),
        "12–15": (12, 14),
        "15–18": (15, 17),
        "18–21": (18, 20),
        "21–24": (21, 23),
    }
    selected_label = st.sidebar.select_slider(
        "Saat aralığı",
        options=list(hour_blocks.keys()),
        value=default_hour_block_label(hour_blocks),
    )
    h0, h1 = hour_blocks[selected_label]
    selected_hours = list(range(h0, h1 + 1))

# Tarih aralığı (SF local referans)
now_sf = _sf_now()
max_days = 7 if mode.startswith("3 Saatlik") else 365
st.sidebar.caption(
    f"{'3 Saatlik' if max_days == 7 else 'Günlük'} görünümde en fazla {max_days} gün seçebilirsiniz."
)

d_start_default = now_sf.date()
d_end_default = now_sf.date()
d_start = st.sidebar.date_input("Başlangıç tarihi", value=d_start_default)
d_end = st.sidebar.date_input("Bitiş tarihi", value=d_end_default)

if (pd.to_datetime(d_end) - pd.to_datetime(d_start)).days > max_days:
    d_end = (pd.to_datetime(d_start) + pd.Timedelta(days=max_days)).date()
    st.sidebar.warning(f"Seçim {max_days} günü aşamaz; bitiş {d_end} olarak güncellendi.")

geof_txt = st.sidebar.text_input("GEOID filtre (virgülle ayır)", value="")
geoids_sel = [g.strip().zfill(11) for g in geof_txt.split(",") if g.strip()]
top_k = st.sidebar.slider("Top-K (tablo)", 10, 200, 50, step=10)

# -----------------------------
# 5) VERİ YÜKLE / FİLTRELE
# -----------------------------
with st.spinner("Veriler yükleniyor…"):
    fc = load_full_fc()
    prof = load_geoid_profile()  # opsiyonel
    geojson = load_geojson()

# filtre penceresi
t0 = pd.to_datetime(d_start).floor("D")
t1 = pd.to_datetime(d_end).floor("D") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

view_df = fc[(fc["date"] >= t0) & (fc["date"] <= t1)].copy()
if geoids_sel:
    view_df = view_df[view_df["GEOID"].isin(geoids_sel)].copy()

if mode.startswith("3 Saatlik"):
    if len(selected_hours):
        view_df = view_df[view_df["hour_start"].isin(selected_hours)].copy()
else:
    # Günlük: aynı gün içinde slotları günlük agregede toplayacağız
    pass

if view_df.empty:
    st.info("Seçilen aralık için kayıt bulunamadı; en güncel çıktıdan göstereceğim.")
    view_df = fc.copy()

# -----------------------------
# 6) AGG — GEOID bazlı özet metrikler (katmana göre)
# -----------------------------
# risk_mean: risk_prob ortalaması (0–1)
# harm_mean: expected_harm ortalaması (ölçek run'a göre)
# expected_mean: expected_count ortalaması
# top1_category: ilk görülen (pencerede)
def first_nonnull(s):
    s2 = s.dropna()
    return s2.iloc[0] if len(s2) else None

# Günlük modda: önce (GEOID, date) bazında topla sonra GEOID mean (daha stabil)
if mode.startswith("Günlük"):
    day_agg = (
        view_df.groupby(["GEOID", "date"], as_index=False)
        .agg(
            risk_day=("risk_prob", "mean"),
            harm_day=("expected_harm", "mean"),
            expected_day=("expected_count", "mean"),
            top1_category=("top1_category", first_nonnull),
        )
    )
    agg = (
        day_agg.groupby("GEOID", as_index=False)
        .agg(
            risk_mean=("risk_day", "mean"),
            harm_mean=("harm_day", "mean"),
            expected_mean=("expected_day", "mean"),
            top1_category=("top1_category", first_nonnull),
        )
    )
else:
    agg = (
        view_df.groupby("GEOID", as_index=False)
        .agg(
            risk_mean=("risk_prob", "mean"),
            harm_mean=("expected_harm", "mean"),
            expected_mean=("expected_count", "mean"),
            top1_category=("top1_category", first_nonnull),
        )
    )

# GEOID profile ile zenginleştir (varsa)
if len(prof) and "GEOID" in prof.columns:
    keep_cols = [
        "GEOID", "population", "poi_total_count", "poi_risk_score",
        "bus_stop_count", "train_stop_count",
        "distance_to_police", "is_near_police",
        "distance_to_government_building", "is_near_government",
        "neighbor_crime_1d", "neighbor_crime_3d", "neighbor_crime_7d",
        "avg_911_daily", "avg_911_hourly", "avg_311_daily",
    ]
    keep_cols = [c for c in keep_cols if c in prof.columns]
    agg = agg.merge(prof[keep_cols], on="GEOID", how="left")

# katmana göre metric_value seç
if layer.startswith("Risk"):
    agg["metric_value"] = pd.to_numeric(agg["risk_mean"], errors="coerce")
    agg["bucket"] = agg["metric_value"].map(bucket_of_fixed01)
else:
    agg["metric_value"] = pd.to_numeric(agg["harm_mean"], errors="coerce")
    harm_bucketer, harm_q = make_quintile_bucketizer(agg["metric_value"])
    agg["bucket"] = agg["metric_value"].map(harm_bucketer)

# Top-K sıralama
agg_sorted = agg.sort_values("metric_value", ascending=False).reset_index(drop=True)
topk = agg_sorted.head(top_k).copy() if len(agg_sorted) else pd.DataFrame()

# -----------------------------
# 7) HARİTA (Folium)
# -----------------------------
st.subheader("🗺️ Harita — 5 seviye Likert (katmana göre)")

if geojson and len(agg_sorted):
    gj_enriched = enrich_geojson_with_metric(geojson, agg_sorted, metric_key="metric_value")

    # Lejand
    st.markdown(
        "**Lejand:** "
        "<span style='background:#dcdcdc;padding:2px 6px;border-radius:4px;'>Çok Düşük</span> "
        "<span style='background:#38a800;padding:2px 6px;border-radius:4px;'>Düşük</span> "
        "<span style='background:#ffdd00;padding:2px 6px;border-radius:4px;'>Orta</span> "
        "<span style='background:#ff8c00;padding:2px 6px;border-radius:4px;'>Yüksek</span> "
        "<span style='background:#a00000;padding:2px 6px;border-radius:4px;'>Çok Yüksek</span> ",
        unsafe_allow_html=True,
    )

    m = folium.Map(
        location=[37.7749, -122.4194],
        zoom_start=11,
        tiles="cartodbpositron",
        control_scale=True,
    )

    def style_fn(feature):
        props = feature.get("properties", {})
        rgba = props.get("fill_color", [220, 220, 220, 160])
        return {
            "fillColor": rgba_to_hex(rgba),
            "color": "#505050",
            "weight": 0.5,
            "fillOpacity": float(rgba[3]) / 255.0 if len(rgba) == 4 else 0.6,
        }

    def highlight_fn(feature):
        return {"weight": 2, "color": "#000000"}

    # Tooltip: her zaman hem risk hem zarar göster; ama "metric" seçilen katman
    metric_alias = "Risk (0–1)" if layer.startswith("Risk") else "Zarar (expected_harm)"
    tooltip = folium.GeoJsonTooltip(
        fields=["display_id", "bucket", "metric_txt", "risk_txt", "harm_txt", "expected_txt", "top1_category"],
        aliases=[
            "GEOID:",
            "Likert:",
            f"{metric_alias}:",
            "Ortalama risk (0–1):",
            "Ortalama zarar:",
            "Beklenen olay:",
            "Top1 suç türü:",
        ],
        sticky=True,
    )

    folium.GeoJson(
        gj_enriched,
        name="Layer",
        style_function=style_fn,
        highlight_function=highlight_fn,
        tooltip=tooltip,
    ).add_to(m)

    clicked_geoid = None
    folium_ret = st_folium(
        m,
        width=None,
        height=520,
        returned_objects=["last_active_drawing"],
        key="sutam_one_map",
    )

    if folium_ret and folium_ret.get("last_active_drawing"):
        props = folium_ret["last_active_drawing"].get("properties", {}) or {}
        clicked_geoid = str(props.get("geoid_norm") or props.get("display_id") or "").strip()
        if clicked_geoid:
            st.session_state["clicked_geoid_onepage"] = clicked_geoid
else:
    st.info("GeoJSON veya GEOID bazlı özet veri yok; harita devre dışı.")

# -----------------------------
# 8) GEOID SEÇİMİ + SEKME YAPISI
# -----------------------------
st.markdown("---")

options = sorted(agg_sorted["GEOID"].astype(str).unique().tolist()) if len(agg_sorted) else []
clicked = st.session_state.get("clicked_geoid_onepage", None)

selected_geoid = None
if options:
    default_index = options.index(clicked) if clicked in options else 0
    selected_geoid = st.selectbox("Detay göstermek için GEOID seç:", options, index=default_index)
else:
    st.info("Detay için listelenecek GEOID bulunamadı.")

tab1, tab2, tab3 = st.tabs(["Özet & Kolluk Önerileri", "Zaman Serisi", "Isı Haritası / Top-K"])

# -----------------------------
# TAB 1
# -----------------------------
with tab1:
    st.subheader("📌 Özet & Kolluk önerileri (katmana göre)")

    if selected_geoid is None:
        st.info("Lütfen bir GEOID seçin.")
    else:
        df_sel = view_df[view_df["GEOID"] == selected_geoid].copy()
        df_sel = df_sel.sort_values("timestamp" if mode.startswith("3 Saatlik") else "date")

        row_agg = agg_sorted[agg_sorted["GEOID"] == selected_geoid].iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("GEOID", selected_geoid)
        c2.metric("Likert", str(row_agg.get("bucket", "—")))
        if layer.startswith("Risk"):
            c3.metric("Risk (0–1)", f"{float(row_agg.get('risk_mean', np.nan)):.3f}" if pd.notna(row_agg.get("risk_mean", np.nan)) else "—")
        else:
            c3.metric("Zarar (expected_harm)", f"{float(row_agg.get('harm_mean', np.nan)):.3f}" if pd.notna(row_agg.get("harm_mean", np.nan)) else "—")

        # İkinci satır metrikler
        c4, c5, c6 = st.columns(3)
        c4.metric("Beklenen olay", f"{float(row_agg.get('expected_mean', np.nan)):.3f}" if pd.notna(row_agg.get("expected_mean", np.nan)) else "—")
        c5.metric("Top1 suç türü", str(row_agg.get("top1_category", "—") or "—"))

        # Komşu suç + 911/311 (profile'dan)
        n7 = row_agg.get("neighbor_crime_7d", np.nan)
        c6.metric("Komşu suç (7 gün)", f"{float(n7):.1f}" if pd.notna(n7) else "—")

        st.markdown("### 🛡️ Kolluk önerileri (otomatik)")
        if len(topk):
            top1 = topk.iloc[0]
            reco = build_ops_reco(layer, top1)
            for r in reco:
                st.markdown(f"- {r}")
        else:
            st.caption("Top-K üretilmediği için öneri üretilemedi.")

        st.markdown("### 🧠 Seçili GEOID — kısa açıklama")
        # Katmana göre kısa gerekçe
        parts = []
        if layer.startswith("Risk"):
            rv = row_agg.get("risk_mean", np.nan)
            if pd.notna(rv):
                parts.append(f"Bu pencerede **risk ortalaması** ≈ **{float(rv):.3f}**.")
            ev = row_agg.get("expected_mean", np.nan)
            if pd.notna(ev):
                parts.append(f"**Beklenen olay** ≈ **{float(ev):.3f}**.")
        else:
            hv = row_agg.get("harm_mean", np.nan)
            if pd.notna(hv):
                parts.append(f"Bu pencerede **beklenen zarar** ≈ **{float(hv):.3f}**.")
            ev = row_agg.get("expected_mean", np.nan)
            if pd.notna(ev):
                parts.append(f"**Beklenen olay** ≈ **{float(ev):.3f}** (zarar sıralaması olay sayısından farklı olabilir).")

        if str(row_agg.get("top1_category", "") or "").strip():
            parts.append(f"Baskın tür: **{row_agg.get('top1_category')}**.")

        if pd.notna(row_agg.get("poi_total_count", np.nan)):
            parts.append(f"POI sayısı ≈ **{float(row_agg.get('poi_total_count', 0)):.0f}**.")

        if parts:
            st.markdown(" ".join(parts))
        else:
            st.caption("Bu GEOID için özet açıklama üretilemedi.")

        with st.expander("🔎 Debug: Seçili GEOID ham kayıtlar", expanded=False):
            cols_show = ["date", "hour_range", "risk_prob", "expected_count", "expected_harm", "top1_category", "risk_level", "risk_decile", "risk_score"]
            cols_show = [c for c in cols_show if c in df_sel.columns]
            st.write(df_sel[cols_show].tail(20))

# -----------------------------
# TAB 2 — Zaman Serisi
# -----------------------------
with tab2:
    st.subheader("📈 Zaman serisi (katmana göre)")

    if view_df.empty:
        st.info("Seçilen aralık için veri yok.")
    else:
        # default: Top-K ilk 3
        default_geoids = topk["GEOID"].head(3).tolist() if len(topk) else []
        if selected_geoid and selected_geoid not in default_geoids:
            default_geoids = [selected_geoid] + default_geoids
        default_geoids = default_geoids[:4]

        options_geoids = sorted(view_df["GEOID"].astype(str).unique().tolist())

        chosen = st.multiselect(
            "Grafikte gösterilecek GEOID'ler",
            options=options_geoids,
            default=[g for g in default_geoids if g in options_geoids],
        )

        y_col = "risk_prob" if layer.startswith("Risk") else "expected_harm"
        if y_col not in view_df.columns:
            st.warning(f"{y_col} kolonu veride yok.")
        elif chosen:
            x_col = "timestamp" if mode.startswith("3 Saatlik") else "date"
            piv = (
                view_df[view_df["GEOID"].isin(chosen)]
                .pivot_table(index=x_col, columns="GEOID", values=y_col, aggfunc="mean")
                .sort_index()
            )
            st.line_chart(piv, height=360)
        else:
            st.caption("Grafik için en az bir GEOID seçin.")

# -----------------------------
# TAB 3 — Isı Haritası / Top-K
# -----------------------------
with tab3:
    st.subheader("🔥 Isı haritası (GEOID × Zaman)")

    if view_df.empty:
        st.info("Seçilen aralık için veri yok.")
    else:
        x_col = "hour_range" if mode.startswith("3 Saatlik") else "date"
        y_col = "risk_prob" if layer.startswith("Risk") else "expected_harm"
        if y_col not in view_df.columns:
            st.warning(f"{y_col} kolonu veride yok.")
        else:
            heat = (
                view_df.groupby([x_col, "GEOID"], as_index=False)[y_col]
                .mean()
                .pivot(index=x_col, columns="GEOID", values=y_col)
            )
            st.dataframe(
                heat.style.format("{:.3f}"),
                use_container_width=True,
                height=420,
            )

    st.markdown("---")
    st.subheader("🔝 Top-K GEOID (katmana göre)")

    if len(topk):
        show_cols = ["GEOID", "bucket", "metric_value", "risk_mean", "harm_mean", "expected_mean", "top1_category"]
        show_cols = [c for c in show_cols if c in topk.columns]
        st.dataframe(topk[show_cols], use_container_width=True, height=320)
    else:
        st.caption("Top-K tablosu için yeterli veri yok.")

# -----------------------------
# 9) DİPNOT
# -----------------------------
st.caption(
    "Kaynak: deploy/full_fc.parquet (risk_prob, expected_count, expected_harm, top categories) + "
    "deploy/geoid_profile.parquet (GEOID profili/komşu suç/altyapı) + data/sf_cells.geojson (geometri). "
    "Katman seçimi yalnızca renklendirme, Top-K ve kolluk önerilerini değiştirir."
)
