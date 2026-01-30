# pages/Suc_Zarar_Tahmini.py
# SUTAM — 📊 Suç & Suç Zararı Tahmini (TEK HARİTA + KATMAN)
# - Veri: data/forecast_7d.parquet (fallback: deploy/full_fc.parquet)
# - GeoJSON: data/sf_cells.geojson
# - Katman: "Suç Riski" (risk_prob / p_event)  veya  "Zarar Riski" (expected_harm)
# - Likert Q1–Q5: seçili tarih+saat dilimindeki GEOID dağılımına göre quintile
# - Kolluğa: sade özet + 3 maddelik öneri (teknik debug yok)

from __future__ import annotations

import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

import folium
from streamlit_folium import st_folium


# -----------------------------
# 0) DOSYA YOLLARI (SENDE VAR)
# -----------------------------
FC_CANDIDATES = [
    "data/forecast_7d.parquet",
    "deploy/full_fc.parquet",
    "data/full_fc.parquet",
]

GEOID_PROFILE_CANDIDATES = [
    "data/geoid_profile.parquet",
    "deploy/geoid_profile.parquet",
    "data/geoid_profile.csv",
]

GEOJSON_LOCAL = "data/sf_cells.geojson"


# -----------------------------
# 1) HARM WEIGHTS (O-CHF)
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
# 2) RENK / LİKERT (Q1–Q5)
# -----------------------------
LIKERT5 = [
    ("Q1 (Çok Düşük)", "#dcdcdc"),
    ("Q2 (Düşük)",     "#38a800"),
    ("Q3 (Orta)",      "#ffdd00"),
    ("Q4 (Yüksek)",    "#ff8c00"),
    ("Q5 (Çok Yüksek)","#a00000"),
]
LIKERT_LABELS = [x[0] for x in LIKERT5]
LIKERT_COLORS = {lab: col for lab, col in LIKERT5}


def _digits11(x) -> str:
    s = "".join(ch for ch in str(x) if ch.isdigit())
    return s.zfill(11) if s else ""


def normalize_geoid_11(s: str) -> str:
    s = str(s).replace(".0", "").strip()
    if s == "0":
        return "0"
    return _digits11(s)


@st.cache_data(show_spinner=False)
def load_forecast() -> pd.DataFrame:
    path = next((p for p in FC_CANDIDATES if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError("Forecast parquet bulunamadı. Beklenen: data/forecast_7d.parquet veya deploy/full_fc.parquet")

    df = pd.read_parquet(path)

    # Normalizasyon
    if "GEOID" in df.columns and "geoid" not in df.columns:
        df["geoid"] = df["GEOID"]
    df["geoid"] = df["geoid"].astype(str).map(normalize_geoid_11)

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")
    if "hour_range" not in df.columns:
        raise ValueError("Forecast içinde 'hour_range' yok. full_fc şemasında olmalı.")

    # Suç olasılığı metriği (önce risk_prob, yoksa p_event)
    if "risk_prob" in df.columns:
        df["crime_prob"] = pd.to_numeric(df["risk_prob"], errors="coerce")
    elif "p_event" in df.columns:
        df["crime_prob"] = pd.to_numeric(df["p_event"], errors="coerce")
    else:
        df["crime_prob"] = np.nan

    # Beklenen suç sayısı
    if "expected_crimes" in df.columns:
        df["expected_cnt"] = pd.to_numeric(df["expected_crimes"], errors="coerce")
    elif "expected_count" in df.columns:
        df["expected_cnt"] = pd.to_numeric(df["expected_count"], errors="coerce")
    else:
        df["expected_cnt"] = np.nan

    # Zarar metriği: önce expected_harm varsa onu kullan, yoksa türet
    if "expected_harm" in df.columns:
        df["harm_expected"] = pd.to_numeric(df["expected_harm"], errors="coerce")
    else:
        # 1) expected_cnt * avg_harm_per_crime varsa
        if "avg_harm_per_crime" in df.columns:
            ah = pd.to_numeric(df["avg_harm_per_crime"], errors="coerce")
            df["harm_expected"] = df["expected_cnt"] * ah
        else:
            # 2) top1_category + top1_share vb ile yaklaşık zarar
            # harm ≈ expected_cnt * Σ(share_k * harm_w(cat_k))
            harm_sum = 0.0
            any_share = False
            for k in [1, 2, 3]:
                ccat = f"top{k}_category"
                csh  = f"top{k}_share"
                if ccat in df.columns and csh in df.columns:
                    cat = df[ccat].astype(str)
                    sh  = pd.to_numeric(df[csh], errors="coerce").fillna(0.0)
                    w   = cat.map(lambda x: HARM_W.get(x, UNK_W)).astype(float)
                    harm_sum = harm_sum + (sh * w)
                    any_share = True
            if any_share:
                df["harm_expected"] = df["expected_cnt"] * harm_sum
            else:
                df["harm_expected"] = np.nan

    # Güvenlik: negatifleri temizle
    df["crime_prob"] = df["crime_prob"].clip(lower=0.0)
    df["expected_cnt"] = df["expected_cnt"].clip(lower=0.0)
    df["harm_expected"] = df["harm_expected"].clip(lower=0.0)

    return df


@st.cache_data(show_spinner=False)
def load_geoid_profile() -> pd.DataFrame:
    path = next((p for p in GEOID_PROFILE_CANDIDATES if os.path.exists(p)), None)
    if path is None:
        return pd.DataFrame()

    if path.endswith(".csv"):
        prof = pd.read_csv(path)
    else:
        prof = pd.read_parquet(path)

    if "GEOID" in prof.columns and "geoid" not in prof.columns:
        prof["geoid"] = prof["GEOID"]
    prof["geoid"] = prof["geoid"].astype(str).map(normalize_geoid_11)
    return prof


@st.cache_data(show_spinner=False)
def load_geojson() -> dict:
    if os.path.exists(GEOJSON_LOCAL):
        with open(GEOJSON_LOCAL, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def q5_bucket(series: pd.Series) -> pd.Series:
    """Quintile Q1–Q5: dağılıma göre 5'li Likert."""
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() < 5:
        # veri azsa hepsini Q3'e çek
        return pd.Series(["Q3 (Orta)"] * len(series), index=series.index)

    qs = x.quantile([0.2, 0.4, 0.6, 0.8]).values

    def lab(v):
        if pd.isna(v):
            return "Q1 (Çok Düşük)"
        if v <= qs[0]:
            return "Q1 (Çok Düşük)"
        if v <= qs[1]:
            return "Q2 (Düşük)"
        if v <= qs[2]:
            return "Q3 (Orta)"
        if v <= qs[3]:
            return "Q4 (Yüksek)"
        return "Q5 (Çok Yüksek)"

    return x.map(lab)


def pick_default_geoid(df_slice: pd.DataFrame, metric: str) -> str | None:
    if df_slice.empty:
        return None
    # şehir geneli 0'ı varsayılan yapma; hücrelerden seç
    cells = df_slice[df_slice["geoid"] != "0"].copy()
    if cells.empty:
        return None
    best = cells.sort_values(metric, ascending=False).iloc[0]["geoid"]
    return str(best)


def make_simple_recommendations(layer: str, row: pd.Series, prof_row: pd.Series | None) -> list[str]:
    """
    Kolluk dili: 3 madde, basit.
    layer: 'crime' veya 'harm'
    """
    rec = []

    top_cat = str(row.get("top1_category", "Unknown") or "Unknown")
    hr = str(row.get("hour_range", "—") or "—")
    exp_cnt = float(row.get("expected_cnt", 0) or 0)
    prob = float(row.get("crime_prob", 0) or 0)
    harm = float(row.get("harm_expected", 0) or 0)

    # 1) Katmana göre ana mesaj
    if layer == "crime":
        if prob >= 0.65 or exp_cnt >= 1.0:
            rec.append(f"Bu saat diliminde **devriye görünürlüğünü artırın** (önleyici varlık). Öncelik: {top_cat}.")
        else:
            rec.append(f"Bu slotta risk düşük/orta. **Rutin devriye** yeterli; odak {top_cat} olabilir.")
    else:
        # harm
        if harm >= np.nanquantile([harm], 0.5):  # tek değer; yine de mesaj üretelim
            rec.append(f"Bu slotta **zarar etkisi yüksek** olabilir. Müdahale kapasitesini kritik noktalara kaydırın.")
        else:
            rec.append("Bu slotta zarar etkisi sınırlı görünüyor. Rutin devriye + hızlı müdahale hazırlığı yeterli.")

    # 2) Yakın çevre (profil varsa)
    if prof_row is not None and len(prof_row):
        n7 = prof_row.get("neighbor_crime_7d", np.nan)
        try:
            n7 = float(n7)
        except Exception:
            n7 = np.nan

        if pd.notna(n7) and n7 >= 150:
            rec.append("Yakın çevrede son 7 günde hareketlilik yüksek: **komşu hücre geçişlerinde** devriye turu planlayın.")
        elif pd.notna(n7) and n7 >= 60:
            rec.append("Komşu hücrelerde orta yoğunluk: **kısa aralıklarla kontrol** önerilir.")
        else:
            rec.append("Komşu hücrelerde belirgin yoğunluk yok: **hedefli kısa kontrol** yeterli.")

    # 3) Polis yakınlığı (profil varsa) — sade
    if prof_row is not None and len(prof_row):
        near_pol = prof_row.get("is_near_police", np.nan)
        try:
            near_pol = float(near_pol)
        except Exception:
            near_pol = np.nan

        if pd.notna(near_pol) and near_pol >= 1:
            rec.append("Bölge polis noktasına yakın: **hızlı reaksiyon** avantajını kullanın, görünür devriye ile caydırıcılık sağlayın.")
        else:
            rec.append("Bölge polis noktasına uzak olabilir: **telsiz/ekip koordinasyonunu** güçlü tutun ve müdahale süresini azaltın.")

    # 3 maddeyi aşırı uzatmayalım
    return rec[:3]


def render_suc_zarar_tahmini():
    # -----------------------------
    # ÜST BAŞLIK / SADE AÇIKLAMA
    # -----------------------------
    st.markdown("# 📊 Suç & Suç Zararı Tahmini")
    st.caption("Tek harita üzerinde katman seçimi: **Suç riski** veya **Zarar (O-CHF) riski**. Likert Q1–Q5 dağılıma göre otomatik.")

    # -----------------------------
    # VERİ
    # -----------------------------
    with st.spinner("Tahmin verisi yükleniyor…"):
        df = load_forecast()
        prof = load_geoid_profile()
        geojson = load_geojson()

    if df.empty:
        st.error("Tahmin verisi boş görünüyor.")
        return

    # SF zamanı (gösterim için)
    try:
        now_sf = datetime.now(ZoneInfo("America/Los_Angeles"))
    except Exception:
        now_sf = datetime.utcnow()

    # -----------------------------
    # UI — SADE FİLTRELER
    # -----------------------------
    left, right = st.columns([1.2, 1], gap="large")

    with left:
        # Katman seçimi (harita + öneriler buna göre)
        layer = st.radio(
            "Harita katmanı",
            ["Suç Riski", "Zarar Riski (O-CHF)"],
            index=0,
            horizontal=True,
        )
        layer_key = "crime" if layer.startswith("Suç") else "harm"

        # Tarih + Saat aralığı (full_fc zaten hour_range ile gelir)
        available_dates = sorted(df["date"].dropna().dt.date.unique().tolist())
        if not available_dates:
            st.error("Tarih alanı üretilemedi.")
            return

        default_date = now_sf.date() if now_sf.date() in available_dates else available_dates[0]
        sel_date = st.date_input("Tarih (SF)", value=default_date, min_value=available_dates[0], max_value=available_dates[-1])

        hr_options = sorted(df["hour_range"].dropna().astype(str).unique().tolist())
        # Şu anki saat aralığını yakalamaya çalış
        def _guess_hr():
            h = now_sf.hour
            # hour_range "00-03" gibi
            for opt in hr_options:
                s = opt.replace("–","-").replace("—","-")
                if "-" in s:
                    a, b = s.split("-", 1)
                    try:
                        h0 = int(a.strip())
                        h1 = int(b.strip())
                        # 24 için 23 kabul
                        if h1 == 24:
                            h1 = 23
                        if h0 <= h <= h1:
                            return opt
                    except Exception:
                        continue
            return hr_options[0] if hr_options else "00-03"

        sel_hr = st.selectbox("Saat aralığı", options=hr_options, index=hr_options.index(_guess_hr()) if hr_options else 0)

        # Dilim
        d0 = pd.to_datetime(sel_date).floor("D")
        df_slice = df[(df["date"] == d0) & (df["hour_range"].astype(str) == str(sel_hr))].copy()

        if df_slice.empty:
            st.warning("Seçilen tarih+saat diliminde kayıt yok. En yakın dilim gösteriliyor.")
            # fallback: en güncel dilim
            latest_date = df["date"].max()
            latest_hr = df[df["date"] == latest_date]["hour_range"].astype(str).mode().iloc[0]
            df_slice = df[(df["date"] == latest_date) & (df["hour_range"].astype(str) == str(latest_hr))].copy()
            sel_date = latest_date.date()
            sel_hr = str(latest_hr)

        # Metrik seçimi
        metric = "crime_prob" if layer_key == "crime" else "harm_expected"
        df_slice[metric] = pd.to_numeric(df_slice[metric], errors="coerce").fillna(0.0)

        # Q1–Q5 (quintile) bu dilim için
        df_slice["likert"] = q5_bucket(df_slice[metric])
        df_slice["fillColor"] = df_slice["likert"].map(LIKERT_COLORS)

        # Varsayılan GEOID: o dilimde en yüksek metrik
        default_geoid = pick_default_geoid(df_slice, metric)
        if "selected_geoid" not in st.session_state:
            st.session_state["selected_geoid"] = default_geoid

        # GEOID seçim listesi (0 hariç + isteğe bağlı 0)
        geoids = sorted([g for g in df_slice["geoid"].astype(str).unique().tolist() if g != "0"])
        if not geoids:
            st.error("Bu dilimde hücre (GEOID) verisi yok.")
            return

        # Eğer haritadan tık geldiyse onu al
        clicked = st.session_state.get("clicked_geoid_forecast")
        if clicked and clicked in geoids:
            st.session_state["selected_geoid"] = clicked

        # Seçili geoid geçersizse default’a çek
        if st.session_state["selected_geoid"] not in geoids:
            st.session_state["selected_geoid"] = default_geoid or geoids[0]

        selected_geoid = st.selectbox(
            "GEOID seç (detay & öneriler)",
            options=geoids,
            index=geoids.index(st.session_state["selected_geoid"]),
        )
        st.session_state["selected_geoid"] = selected_geoid

    # -----------------------------
    # HARİTA
    # -----------------------------
    with left:
        st.markdown("### 🗺️ Harita (Likert Q1–Q5)")

        if not geojson:
            st.info("GeoJSON (data/sf_cells.geojson) bulunamadı. Harita devre dışı.")
        else:
            # GeoJSON’daki özelliklere geoid_norm yazıp renk ata
            # GeoJSON props içinden GEOID adayını bul
            risk_map = df_slice.set_index("geoid")

            feats = []
            for feat in geojson.get("features", []):
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

                g11 = _digits11(raw)
                props["geoid_norm"] = g11
                props["likert"] = ""
                props["metric_txt"] = ""
                props["fillColor"] = "#dcdcdc"

                if g11 and g11 in risk_map.index:
                    rr = risk_map.loc[g11]
                    props["likert"] = str(rr.get("likert", ""))
                    props["fillColor"] = str(rr.get("fillColor", "#dcdcdc"))
                    try:
                        props["metric_txt"] = f"{float(rr.get(metric, 0.0)):.3f}"
                    except Exception:
                        props["metric_txt"] = ""

                feats.append({**feat, "properties": props})

            gj = {**geojson, "features": feats}

            # Folium
            m = folium.Map(location=[37.7749, -122.4194], zoom_start=11, tiles="cartodbpositron", control_scale=True)

            def style_fn(feature):
                c = (feature.get("properties") or {}).get("fillColor", "#dcdcdc")
                return {"fillColor": c, "color": "#505050", "weight": 0.6, "fillOpacity": 0.72}

            tooltip = folium.GeoJsonTooltip(
                fields=["geoid_norm", "likert", "metric_txt", "top1_category"],
                aliases=[
                    "GEOID:",
                    "Seviye:",
                    "Skor:",
                    "En olası tür:",
                ],
                sticky=True,
            )

            folium.GeoJson(gj, style_function=style_fn, tooltip=tooltip, name="risk").add_to(m)

            # Lejand (mini)
            legend_html = """
            <div style="position: fixed; bottom: 22px; left: 22px; z-index: 9999;
                        background: white; padding: 10px 12px; border-radius: 10px;
                        border: 1px solid #e2e8f0; font-size: 12px;">
              <div style="font-weight:700; margin-bottom:6px;">Likert (Q1–Q5)</div>
            """
            for lab, col in LIKERT5:
                legend_html += f"""
                <div style="display:flex; align-items:center; gap:8px; margin:3px 0;">
                  <span style="width:14px; height:10px; display:inline-block; background:{col}; border:1px solid #999;"></span>
                  <span>{lab}</span>
                </div>
                """
            legend_html += "</div>"
            m.get_root().html.add_child(folium.Element(legend_html))

            folium_ret = st_folium(
                m,
                width=None,
                height=520,
                returned_objects=["last_active_drawing"],
                key="sutam_forecast_map",
            )

            # Tıkla → GEOID seç
            if folium_ret and folium_ret.get("last_active_drawing"):
                props = folium_ret["last_active_drawing"].get("properties", {}) or {}
                cg = str(props.get("geoid_norm") or "").strip()
                if cg:
                    st.session_state["clicked_geoid_forecast"] = cg

    # -----------------------------
    # SAĞ PANEL: SADE KOLLUK ÖZETİ
    # -----------------------------
    with right:
        st.markdown("### 🧩 Kolluğa Özet (sade)")

        row = df_slice[df_slice["geoid"] == st.session_state["selected_geoid"]]
        if row.empty:
            st.info("Seçili GEOID için bu dilimde kayıt yok.")
            return
        row = row.iloc[0]

        # Profile satırı (komşu suç, polis yakınlığı, POI vb.)
        prof_row = None
        if not prof.empty:
            pr = prof[prof["geoid"] == st.session_state["selected_geoid"]]
            if len(pr):
                prof_row = pr.iloc[0]

        # Başlık kartı
        likert = str(row.get("likert", "Q3 (Orta)"))
        top_cat = str(row.get("top1_category", "Unknown") or "Unknown")

        crime_prob = float(row.get("crime_prob", 0) or 0)
        exp_cnt = float(row.get("expected_cnt", 0) or 0)
        harm_exp = float(row.get("harm_expected", 0) or 0)

        # Metinleri katmana göre sadeleştir
        if layer_key == "crime":
            st.markdown(
                f"""
                <div style="border:1px solid #e2e8f0;border-radius:14px;padding:12px;background:#fff;">
                  <div style="font-weight:800;font-size:14px;margin-bottom:4px;">GEOID: {st.session_state["selected_geoid"]}</div>
                  <div style="color:#475569;font-size:13px;">Tarih/Saat (SF): <b>{sel_date}</b> • <b>{sel_hr}</b></div>
                  <div style="margin-top:8px;font-size:13px;">
                    Seviye: <b>{likert}</b><br/>
                    Suç olasılığı: <b>%{crime_prob*100:.1f}</b> • Beklenen olay: <b>{exp_cnt:.2f}</b><br/>
                    En olası tür: <b>{top_cat}</b>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="border:1px solid #e2e8f0;border-radius:14px;padding:12px;background:#fff;">
                  <div style="font-weight:800;font-size:14px;margin-bottom:4px;">GEOID: {st.session_state["selected_geoid"]}</div>
                  <div style="color:#475569;font-size:13px;">Tarih/Saat (SF): <b>{sel_date}</b> • <b>{sel_hr}</b></div>
                  <div style="margin-top:8px;font-size:13px;">
                    Seviye: <b>{likert}</b><br/>
                    Beklenen zarar (O-CHF): <b>{harm_exp:.2f}</b> • Beklenen olay: <b>{exp_cnt:.2f}</b><br/>
                    En olası tür: <b>{top_cat}</b>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        # Öneriler (3 madde)
        st.markdown("#### ✅ Kolluk önerileri")
        recs = make_simple_recommendations(layer_key, row, prof_row)
        for r in recs:
            st.markdown(f"- {r}")

        # Çok kısa “neden/bağlam” (teknik olmayan)
        st.write("")
        st.markdown("#### ℹ️ Kısa bağlam")
        chips = []

        # Komşu yoğunluğu
        if prof_row is not None:
            n7 = prof_row.get("neighbor_crime_7d", np.nan)
            try:
                n7 = float(n7)
                if n7 >= 150:
                    chips.append("Çevre yoğunluğu: Yüksek")
                elif n7 >= 60:
                    chips.append("Çevre yoğunluğu: Orta")
                else:
                    chips.append("Çevre yoğunluğu: Düşük")
            except Exception:
                pass

            # POI
            poi = prof_row.get("poi_total_count", np.nan)
            try:
                poi = float(poi)
                if poi >= 50:
                    chips.append("Aktivite (POI): Yüksek")
                elif poi >= 10:
                    chips.append("Aktivite (POI): Orta")
                else:
                    chips.append("Aktivite (POI): Düşük")
            except Exception:
                pass

        # Saat bilgisi (gece/gündüz)
        # hour_range "18-21" -> gece yorumu
        hr = str(sel_hr).replace("–","-").replace("—","-")
        try:
            h0 = int(hr.split("-", 1)[0].strip())
            if h0 >= 21 or h0 < 6:
                chips.append("Zaman: Gece")
            else:
                chips.append("Zaman: Gündüz")
        except Exception:
            pass

        if chips:
            st.markdown(" • ".join([f"**{c}**" for c in chips]))
        else:
            st.caption("Bağlam bilgisi üretilemedi (profil dosyası yok veya alanlar eksik).")

        st.write("")
        st.caption("Not: Likert Q1–Q5 seviyeleri, seçili tarih+saat dilimindeki GEOID dağılımına göre otomatik hesaplanır.")


# Streamlit multi-page import için gerekli:
# app.py içinden çağrılacak fonksiyon adı:
# render_suc_zarar_tahmini()
