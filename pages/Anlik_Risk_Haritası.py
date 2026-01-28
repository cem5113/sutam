# pages/Anlik_Risk_Haritasi.py
# SUTAM — 🗺️ Anlık Suç Haritası (MODÜL)
# ✅ Router uyumlu: import sırasında çalışmaz, sadece render() çağrılınca çalışır.
# ✅ st.set_page_config YOK (app.py'de tek kez var)

from __future__ import annotations

import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional, Iterable, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from src.io_data import load_parquet_or_csv, prepare_forecast


# ----------------------------
# CONFIG
# ----------------------------
DATA_DIR = os.getenv("DATA_DIR", "data").rstrip("/")

FC_CANDIDATES = [
    f"{DATA_DIR}/forecast_7d.parquet",
    f"{DATA_DIR}/full_fc.parquet",
    "data/forecast_7d.parquet",
    "data/full_fc.parquet",
    "deploy/forecast_7d.parquet",
    "deploy/full_fc.parquet",
]

GEOJSON_LOCAL = os.getenv("GEOJSON_PATH", "data/sf_cells.geojson")
TARGET_TZ = "America/Los_Angeles"
TOPK_DEFAULT = 25

LIKERT = {
    1: ("Çok Düşük", [46, 204, 113]),
    2: ("Düşük",     [88, 214, 141]),
    3: ("Orta",      [241, 196, 15]),
    4: ("Yüksek",    [230, 126, 34]),
    5: ("Çok Yüksek",[192, 57, 43]),
}
DEFAULT_FILL = [220, 220, 220]


# ----------------------------
# HELPERS
# ----------------------------
def _is_url(p: str) -> bool:
    return str(p).startswith(("http://", "https://"))

def _first_existing(cands: list[str]) -> Optional[str]:
    for p in cands:
        if _is_url(p):
            return p
        if os.path.exists(p):
            return p
    return None

def _digits11(x) -> str:
    s = "".join(ch for ch in str(x) if ch.isdigit())
    return s.zfill(11) if s else ""

def _parse_range(tok: str) -> Optional[Tuple[int, int]]:
    if not isinstance(tok, str) or "-" not in tok:
        return None
    a, b = tok.split("-", 1)
    try:
        s = int(a.strip()); e = int(b.strip())
    except Exception:
        return None
    s = max(0, min(23, s))
    e = max(1, min(24, e))
    return (s, e)  # end exclusive

def _hour_to_bucket(h: int, labels: Iterable[str]) -> Optional[str]:
    parsed = []
    for lab in labels:
        rg = _parse_range(str(lab))
        if rg:
            parsed.append((str(lab), rg[0], rg[1]))

    for lab, s, e in parsed:
        if s <= h < e:
            return lab

    for lab, s, e in parsed:  # wrap-around
        if s > e and (h >= s or h < e):
            return lab

    return parsed[0][0] if parsed else None

def _col(df: pd.DataFrame, *names: str) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n in df.columns:
            return n
        if n.lower() in cols_lower:
            return cols_lower[n.lower()]
    return None

def _to_num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _format_expected(x) -> str:
    try:
        v = float(x)
        if v < 0: v = 0.0
        lo = int(np.floor(v))
        hi = int(np.ceil(v))
        return f"~{lo}" if lo == hi else f"~{lo}–{hi}"
    except Exception:
        return "—"

def _risk_to_likert(df: pd.DataFrame) -> pd.Series:
    for c in ["risk_likert", "likert", "risk_level_5", "risk5"]:
        cc = _col(df, c)
        if cc:
            s = _to_num(df[cc]).fillna(3).astype(int)
            return s.clip(1, 5)

    rl = _col(df, "risk_level")
    if rl:
        s = df[rl].astype(str).str.lower()
        mapping = {"very_low":1,"low":2,"medium":3,"high":4,"critical":5,"very_high":5}
        out = s.map(mapping)
        if out.notna().any():
            return out.fillna(3).astype(int).clip(1, 5)

    rs = _col(df, "risk_score")
    if rs:
        v = _to_num(df[rs])
        if v.notna().any():
            try:
                bins = pd.qcut(v.rank(method="first"), 5, labels=[1,2,3,4,5])
                return bins.astype(int)
            except Exception:
                pass

    return pd.Series([3]*len(df), index=df.index)

def _pick_rank_score(df: pd.DataFrame) -> pd.Series:
    for c in ["expected_harm", "expected_count", "p_event", "risk_score"]:
        cc = _col(df, c)
        if cc:
            return _to_num(df[cc]).fillna(0.0)
    return pd.Series(np.zeros(len(df)), index=df.index)


# ----------------------------
# LOADERS (cache)
# ----------------------------
@st.cache_data(show_spinner=False)
def load_fc_fast() -> pd.DataFrame:
    p = _first_existing(FC_CANDIDATES)
    if not p:
        return pd.DataFrame()

    try:
        if _is_url(p):
            df = pd.read_parquet(p) if p.lower().endswith(".parquet") else pd.read_csv(p)
        else:
            df = load_parquet_or_csv(p)
    except Exception:
        return pd.DataFrame()

    try:
        df = prepare_forecast(df, gp=None) if not df.empty else df
    except TypeError:
        pass

    return df

@st.cache_data(show_spinner=False)
def load_geojson() -> dict:
    if os.path.exists(GEOJSON_LOCAL):
        with open(GEOJSON_LOCAL, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def enrich_geojson(gj: dict, df_window: pd.DataFrame) -> dict:
    if not gj or df_window.empty:
        return gj

    df = df_window.copy()
    geoid_c = _col(df, "GEOID", "geoid")
    if not geoid_c:
        return gj

    df["geoid11"] = df[geoid_c].map(_digits11)

    df["risk_likert"] = _risk_to_likert(df)
    df["likert_label"] = df["risk_likert"].map(lambda k: LIKERT.get(int(k), ("Orta", DEFAULT_FILL))[0])
    df["fill_color"]   = df["risk_likert"].map(lambda k: LIKERT.get(int(k), ("Orta", DEFAULT_FILL))[1])

    expc = _col(df, "expected_count")
    df["expected_txt"] = df[expc].map(_format_expected) if expc else "—"

    pe = _col(df, "p_event")
    df["p_event_txt"] = _to_num(df[pe]).map(lambda v: f"{v:.3f}" if pd.notna(v) else "—") if pe else "—"

    for i in (1,2,3):
        c = _col(df, f"top{i}_category")
        df[f"top{i}_category"] = df[c].astype(str) if c else ""

    dmap = df.set_index("geoid11")

    feats_out = []
    for feat in gj.get("features", []):
        props = dict(feat.get("properties") or {})

        raw = None
        for k in ("geoid","GEOID","cell_id","id","geoid11","geoid_11"):
            if k in props:
                raw = props[k]; break
        if raw is None:
            for k,v in props.items():
                if "geoid" in str(k).lower():
                    raw = v; break

        key = _digits11(raw)
        props["geoid11"] = key

        props.setdefault("likert_label", "")
        props.setdefault("expected_txt", "—")
        props.setdefault("p_event_txt", "—")
        props.setdefault("top1_category", "")
        props.setdefault("top2_category", "")
        props.setdefault("top3_category", "")
        props.setdefault("fill_color", DEFAULT_FILL)

        if key and key in dmap.index:
            row = dmap.loc[key]
            props["likert_label"]  = str(row.get("likert_label", ""))
            props["expected_txt"]  = str(row.get("expected_txt", "—"))
            props["p_event_txt"]   = str(row.get("p_event_txt", "—"))
            props["top1_category"] = str(row.get("top1_category", ""))
            props["top2_category"] = str(row.get("top2_category", ""))
            props["top3_category"] = str(row.get("top3_category", ""))
            props["fill_color"]    = row.get("fill_color", DEFAULT_FILL)

        feats_out.append({**feat, "properties": props})

    return {**gj, "features": feats_out}

def draw_map(gj: dict):
    try:
        import pydeck as pdk
    except Exception as e:
        st.error("pydeck yüklenemedi. requirements.txt içine `pydeck` ekleyin.")
        st.code(str(e))
        return

    layer = pdk.Layer(
        "GeoJsonLayer",
        gj,
        stroked=True,
        get_line_color=[90,90,90],
        line_width_min_pixels=0.5,
        filled=True,
        get_fill_color="properties.fill_color",
        pickable=True,
        opacity=0.65,
    )

    tooltip = {
        "html": (
            "<b>GEOID:</b> {geoid11}"
            "<br/><b>Risk Seviyesi (5’li):</b> {likert_label}"
            "<br/><b>Beklenen olay:</b> {expected_txt}"
            "<br/><b>Suç olasılığı:</b> {p_event_txt}"
            "<hr style='opacity:0.25'/>"
            "<b>En olası suç türleri</b>"
            "<br/>• {top1_category}"
            "<br/>• {top2_category}"
            "<br/>• {top3_category}"
        ),
        "style": {"backgroundColor":"#111827","color":"white"},
    }

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=37.7749, longitude=-122.4194, zoom=10),
        map_style="light",
        tooltip=tooltip,
    )
    st.pydeck_chart(deck, use_container_width=True)

def make_ops_suggestion(df_window: pd.DataFrame, top_n: int = 15) -> dict:
    if df_window.empty:
        return {"title": "Kolluk Önerisi", "bullets": ["Veri bulunamadı."]}

    tmp = df_window.copy()
    tmp["risk_likert"] = _risk_to_likert(tmp)
    tmp["_rank"] = _pick_rank_score(tmp)

    top = tmp.sort_values(["risk_likert","_rank"], ascending=[False, False]).head(top_n)

    max_l = int(top["risk_likert"].max())
    max_label = LIKERT.get(max_l, ("Orta", DEFAULT_FILL))[0]

    t1 = _col(top, "top1_category")
    cats = [c for c in top[t1].astype(str).tolist() if c and c.lower() != "nan"] if t1 else []
    top_cats = pd.Series(cats).value_counts().head(3).index.tolist() if cats else []

    bullets = []
    if max_l >= 4:
        bullets.append("Yüksek riskli hücrelerde görünür devriye yoğunluğu artırılabilir (sıcak noktalar öncelikli).")
        bullets.append("Transit/ana arter ve yoğun yaya akışlı alanlarda kısa süreli yoğunlaştırılmış devriye önerilir.")
        bullets.append("Kritik bölgelerde hızlı müdahale hattı ve caydırıcılık odaklı konumlanma değerlendirilebilir.")
    else:
        bullets.append("Rutin görünür devriye ve caydırıcılık odaklı dolaşım önerilir.")

    if top_cats:
        bullets.append(f"Bu saat diliminde öne çıkan suç türleri: {', '.join(top_cats)}.")

    bullets.append("Not: Çıktılar bağlayıcı değildir; saha bilgisi ve amir değerlendirmesi ile birlikte yorumlanmalıdır.")
    return {"title": f"Kolluk Önerisi (Bu saat dilimi • en yüksek risk: {max_label})", "bullets": bullets}


# ----------------------------
# PUBLIC ENTRYPOINT (router burayı çağıracak)
# ----------------------------
def render_anlik_risk_haritasi():
    st.markdown("# 🗺️ Anlık Suç Haritası")
    st.caption("Harita, **San Francisco yerel saatine** göre içinde bulunulan saat dilimi için sıcak bölgeleri gösterir (seçim yok).")

    fc = load_fc_fast()
    if fc.empty:
        st.error("Forecast verisi bulunamadı/boş. `data/forecast_7d.parquet` veya `deploy/full_fc.parquet` kontrol edin.")
        return

    date_c  = _col(fc, "date")
    hr_c    = _col(fc, "hour_range")
    geoid_c = _col(fc, "GEOID", "geoid")
    if not date_c or not hr_c or not geoid_c:
        st.error("Forecast içinde `date`, `hour_range`, `GEOID` kolonları bulunamadı.")
        return

    df = fc.copy()
    df[date_c] = pd.to_datetime(df[date_c], errors="coerce")
    df["date_norm"] = df[date_c].dt.normalize()
    df["hr"] = df[hr_c].astype(str)

    now_sf = datetime.now(ZoneInfo(TARGET_TZ))
    today = pd.Timestamp(now_sf.date())

    dates = sorted(df["date_norm"].dropna().unique())
    sel_date = today if today in dates else max([d for d in dates if d <= today], default=dates[0])

    labels = sorted(df["hr"].dropna().unique().tolist())
    sel_hr = _hour_to_bucket(now_sf.hour, labels) or (labels[0] if labels else None)
    if not sel_hr:
        st.error("hour_range etiketleri bulunamadı.")
        return

    st.caption(f"SF saati: **{now_sf:%Y-%m-%d %H:%M}**  •  Tarih: **{pd.Timestamp(sel_date).date()}**  •  Dilim: **{sel_hr}**")

    dfw = df[(df["date_norm"] == sel_date) & (df["hr"] == str(sel_hr))].copy()
    if dfw.empty:
        st.warning("Bu tarih/saat dilimi için kayıt bulunamadı.")
        return

    gj = load_geojson()
    if not gj:
        st.error(f"GeoJSON bulunamadı: `{GEOJSON_LOCAL}`")
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

    gj2 = enrich_geojson(gj, dfw)
    draw_map(gj2)

    st.divider()

    ops = make_ops_suggestion(dfw, top_n=15)
    st.subheader("👮 " + ops["title"])
    for b in ops["bullets"]:
        st.write("•", b)

    st.write("")
    st.subheader("🔥 Top-K Sıcak Bölgeler (Özet Tablo)")
    topk = st.slider("Top-K", 10, 100, TOPK_DEFAULT, 5)

    dfw2 = dfw.copy()
    dfw2["_rank"] = _pick_rank_score(dfw2)

    geoid_c2 = _col(dfw2, "GEOID", "geoid")
    dfw2["GEOID"] = dfw2[geoid_c2].map(_digits11)

    pe = _col(dfw2, "p_event")
    ex = _col(dfw2, "expected_count")
    eh = _col(dfw2, "expected_harm")
    t1 = _col(dfw2, "top1_category")
    t2 = _col(dfw2, "top2_category")
    t3 = _col(dfw2, "top3_category")

    show = pd.DataFrame({"GEOID": dfw2["GEOID"]})
    if pe: show["p_event"] = _to_num(dfw2[pe]).round(3)
    if ex: show["expected_count"] = _to_num(dfw2[ex]).round(2)
    if eh: show["expected_harm"] = _to_num(dfw2[eh]).round(2)
    if t1: show["top1"] = dfw2[t1].astype(str)
    if t2: show["top2"] = dfw2[t2].astype(str)
    if t3: show["top3"] = dfw2[t3].astype(str)

    show["_rank"] = dfw2["_rank"].values
    show = show.sort_values("_rank", ascending=False).head(int(topk)).drop(columns=["_rank"])

    st.dataframe(show, use_container_width=True, hide_index=True)
