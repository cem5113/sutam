# pages/Anlik_Risk_Haritasi.py
# SUTAM — Anlık Risk Haritası (Operasyonel • PATCH’li)
# Amaç: Ne zaman, nerede, neye dikkat etmeli?

def render_suc_zarar_tahmini():
    import os, json
    import numpy as np
    import pandas as pd
    import streamlit as st
    import pydeck as pdk
    from datetime import datetime
    from zoneinfo import ZoneInfo

    # -----------------------------
    # PATHS
    # -----------------------------
    DATA_DIR = os.getenv("DATA_DIR", "data").rstrip("/")
    OPS_CANDIDATES = [
        f"{DATA_DIR}/forecast_7d_ops_ready.parquet",
        "deploy/forecast_7d_ops_ready.parquet",
        "data/forecast_7d_ops_ready.parquet",
    ]
    GEOJSON_PATH = os.getenv("GEOJSON_PATH", "data/sf_cells.geojson")
    TARGET_TZ = "America/Los_Angeles"

    # -----------------------------
    # helpers
    # -----------------------------
    def _first_existing(paths):
        for p in paths:
            if os.path.exists(p):
                return p
        return None

    def _digits11(x):
        s = "".join(ch for ch in str(x) if ch.isdigit())
        return s.zfill(11) if s else ""

    def _pick(df, names):
        m = {c.lower(): c for c in df.columns}
        for n in names:
            if n.lower() in m:
                return m[n.lower()]
        return None

    def _safe_float(x):
        try:
            return float(x)
        except Exception:
            return np.nan

    def _fmt_expected(x):
        v = _safe_float(x)
        if not np.isfinite(v):
            return "—"
        lo, hi = int(np.floor(v)), int(np.ceil(v))
        return f"~{lo}" if lo == hi else f"~{lo}–{hi}"

    def _compute_quintile(series):
        v = pd.to_numeric(series, errors="coerce")
        if v.notna().sum() < 10:
            return pd.Series([3] * len(v), index=v.index)
        return pd.qcut(v.rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)

    # -----------------------------
    # load ops forecast
    # -----------------------------
    ops_path = _first_existing(OPS_CANDIDATES)
    if not ops_path:
        st.error("OPS forecast bulunamadı: forecast_7d_ops_ready.parquet")
        return

    if ops_path.endswith(".parquet"):
        df = pd.read_parquet(ops_path)
    else:
        df = pd.read_csv(ops_path)

    if df.empty:
        st.error("OPS forecast boş geldi.")
        return

    # -----------------------------
    # load geojson
    # -----------------------------
    if not os.path.exists(GEOJSON_PATH):
        st.error("GeoJSON bulunamadı: sf_cells.geojson")
        return
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        gj = json.load(f)

    # -----------------------------
    # normalize base cols
    # -----------------------------
    geo_col = _pick(df, ["GEOID", "geoid"])
    date_col = _pick(df, ["date"])
    hr_col = _pick(df, ["hour", "hour_of_day", "hour_range"])

    if not geo_col or not date_col or not hr_col:
        st.error("Gerekli kolonlar eksik: GEOID/date/hour_range(hour)")
        return

    df["geoid"] = df[geo_col].map(_digits11)
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df["date_norm"] = df["date"].dt.normalize()

    # hour normalize (ops_ready genelde hour_range string; yoksa hour int)
    if df[hr_col].dtype == object:
        # hour_range: "00-03" gibi → başlangıç saatini hour olarak al
        def _hr_start(s):
            try:
                a = str(s).split("-", 1)[0].strip()
                return int(a)
            except Exception:
                return np.nan
        df["hour"] = df[hr_col].map(_hr_start).fillna(0).astype(int)
        df["hour_range"] = df[hr_col].astype(str)
    else:
        df["hour"] = pd.to_numeric(df[hr_col], errors="coerce").fillna(0).astype(int)
        df["hour_range"] = df["hour"].map(lambda h: f"{h:02d}-{min(h+3,24):02d}")

    # -----------------------------
    # UI header
    # -----------------------------
    st.markdown("# 📌 Suç & Zarar Tahmini")
    st.caption("Risk (beklenen suç/olasılık) + Etki (beklenen zarar) aynı ekranda. Operasyon için sadeleştirilmiştir.")

    # -----------------------------
    # MODE toggle
    # -----------------------------
    mode = st.radio(
        "Gösterim modu",
        ["Risk (Suç olasılığı/volume)", "Zarar (Etki/Harm)"],
        horizontal=True
    )
    is_harm = (mode.startswith("Zarar"))

    # -----------------------------
    # time filter
    # -----------------------------
    now = datetime.now(ZoneInfo(TARGET_TZ))
    c1, c2 = st.columns([2,3])
    with c1:
        sel_date = st.date_input("📅 Tarih", value=now.date())
    with c2:
        hr = st.slider("⏰ Saat (başlangıç)", 0, 23, value=int(now.hour))

    dff = df[(df["date_norm"] == pd.Timestamp(sel_date)) & (df["hour"] == hr)].copy()
    if dff.empty:
        st.warning("Bu tarih/saat için kayıt yok.")
        return

    # -----------------------------
    # choose value columns
    # -----------------------------
    risk_col = _pick(dff, ["risk_prob", "p_event", "risk_score"])
    exp_col  = _pick(dff, ["expected_crimes", "expected_count"])
    harm_col = _pick(dff, ["expected_harm", "harm_expected"])

    if not risk_col:
        dff["risk_prob"] = np.nan
        risk_col = "risk_prob"
    if not exp_col:
        dff["expected_crimes"] = np.nan
        exp_col = "expected_crimes"
    if not harm_col:
        dff["expected_harm"] = np.nan
        harm_col = "expected_harm"

    # quintiles for coloring
    if is_harm:
        dff["_lik"] = _compute_quintile(dff[harm_col])
    else:
        # risk modunda p_event/risk_prob ile
        dff["_lik"] = _compute_quintile(dff[risk_col])

    # colors (1..5)
    LIKERT = {
        1: [46, 204, 113],
        2: [88, 214, 141],
        3: [241, 196, 15],
        4: [230, 126, 34],
        5: [192, 57, 43],
    }
    dff["fill_color"] = dff["_lik"].map(lambda k: LIKERT.get(int(k), [220,220,220]))
    dff["expected_txt"] = dff[exp_col].map(_fmt_expected)
    dff["harm_txt"] = dff[harm_col].map(lambda x: f"{_safe_float(x):.1f}" if np.isfinite(_safe_float(x)) else "—")

    # top categories
    for i in (1,2,3):
        if f"top{i}_category" not in dff.columns:
            dff[f"top{i}_category"] = ""
        if f"top{i}_share" not in dff.columns:
            dff[f"top{i}_share"] = 0.0

    # ops action (short)
    if "ops_actions_short" not in dff.columns:
        dff["ops_actions_short"] = ""

    # -----------------------------
    # attach to geojson properties
    # -----------------------------
    g = dff.set_index("geoid")
    feats = []
    for f in gj.get("features", []):
        p = dict(f.get("properties") or {})
        gid = _digits11(p.get("geoid") or p.get("GEOID"))

        # defaults
        p.update({
            "display_id": gid,
            "fill_color": [220,220,220],
            "risk_prob": "—",
            "expected_txt": "—",
            "harm_txt": "—",
            "top": "",
            "action": "",
        })

        if gid in g.index:
            r = g.loc[gid]
            p["fill_color"] = r["fill_color"]
            p["risk_prob"] = f"{_safe_float(r[risk_col]):.2f}" if np.isfinite(_safe_float(r[risk_col])) else "—"
            p["expected_txt"] = r["expected_txt"]
            p["harm_txt"] = r["harm_txt"]
            p["top"] = (
                f"{r['top1_category']} ({float(r['top1_share']):.0%}), "
                f"{r['top2_category']} ({float(r['top2_share']):.0%}), "
                f"{r['top3_category']} ({float(r['top3_share']):.0%})"
            )
            p["action"] = str(r.get("ops_actions_short", ""))

        feats.append({**f, "properties": p})

    gj2 = {**gj, "features": feats}

    # -----------------------------
    # map
    # -----------------------------
    layer = pdk.Layer(
        "GeoJsonLayer",
        gj2,
        filled=True,
        stroked=True,
        get_fill_color="properties.fill_color",
        get_line_color=[80, 80, 80],
        pickable=True,
        opacity=0.65,
    )

    title_line = "Beklenen zarar" if is_harm else "Beklenen suç"
    tooltip = {
        "html": (
            "<b>GEOID:</b> {display_id}<br/>"
            "<b>Risk (p):</b> {risk_prob}<br/>"
            f"<b>{title_line}:</b> " + ("{harm_txt}" if is_harm else "{expected_txt}") + "<br/>"
            "<b>Olası suçlar:</b> {top}<br/>"
            "<b>Öneri:</b> {action}"
        ),
        "style": {"backgroundColor": "#0b1220", "color": "white"},
    }

    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(latitude=37.7749, longitude=-122.4194, zoom=10),
            tooltip=tooltip,
            map_style="light",
        ),
        use_container_width=True,
    )

    # -----------------------------
    # sidebar: Top-10 operasyon listesi
    # -----------------------------
    st.sidebar.markdown("### 🎯 Bu saat için öncelikler (Top 10)")
    rank_col = _pick(dff, ["ops_rank_score", "expected_harm", "expected_crimes", "risk_prob"])
    dff["_rank"] = pd.to_numeric(dff[rank_col], errors="coerce").fillna(-np.inf)
    top10 = dff.sort_values("_rank", ascending=False).head(10)

    show_cols = ["geoid", "hour_range", risk_col, exp_col, harm_col, "ops_actions_short"]
    show_cols = [c for c in show_cols if c in top10.columns]
    st.sidebar.dataframe(top10[show_cols], use_container_width=True, height=360)
