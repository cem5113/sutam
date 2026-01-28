# app.py — SUTAM (FINAL • kurumsal sidebar • 60sn saat • hızlı açılış • page_link/switch_page yok)
from __future__ import annotations

import os
import json
import runpy
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
import pandas as pd

# ---------------------------
# 0) Page config (FIRST)
# ---------------------------
st.set_page_config(
    page_title="SUTAM — Operasyon Paneli",
    page_icon="🛰️",
    layout="wide",
)

# ---------------------------
# 1) Autorefresh (60s) — sadece saat için
# ---------------------------
def enable_autorefresh_60s():
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=60_000, key="sutam_clock_refresh")
    except Exception:
        pass

enable_autorefresh_60s()

# ---------------------------
# 2) Corporate CSS + default nav hide
# ---------------------------
def apply_corporate_style():
    st.markdown(
        """
        <style>
          html, body, [class*="css"]  {
            font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
            color: #0f172a;
          }
          .block-container {
            padding-top: 1.15rem;
            padding-bottom: 2.5rem;
            max-width: 1200px;
          }
          h1, h2, h3 { letter-spacing: -0.02em; }
          h1 { font-size: 1.65rem; margin-bottom: .25rem; }
          h2 { font-size: 1.15rem; margin-top: 1.1rem; }
          p, li { font-size: 0.95rem; line-height: 1.5; }

          .sutam-caption { color: #475569; font-size: 0.90rem; margin-top: 0.15rem; }
          .sutam-muted { color: #64748b; font-size: 0.88rem; }

          .sutam-card {
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 14px 14px;
            background: #ffffff;
            box-shadow: 0 1px 0 rgba(15, 23, 42, 0.03);
          }
          .sutam-card-title { font-weight: 700; font-size: 0.98rem; margin-bottom: 0.25rem; color: #0f172a; }
          .sutam-card-text { color: #334155; font-size: 0.92rem; margin: 0; }

          .sutam-callout {
            border-left: 4px solid #2563eb;
            background: #eff6ff;
            padding: 12px 14px;
            border-radius: 10px;
            color: #0f172a;
          }
          .sutam-ethics {
            border-left: 4px solid #64748b;
            background: #f8fafc;
            padding: 12px 14px;
            border-radius: 10px;
            color: #0f172a;
          }

          section[data-testid="stSidebar"] { border-right: 1px solid #e2e8f0; }

          /* default Pages nav gizle (app + otomatik liste) */
          [data-testid="stSidebarNav"] { display: none !important; }
          section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_corporate_style()

# ---------------------------
# 3) Lightweight "last update" badge
# ---------------------------
DATA_DIR = os.getenv("DATA_DIR", "data").rstrip("/")
AUDIT_CAND = [
    f"{DATA_DIR}/deploy_audit.json",
    "deploy/deploy_audit.json",
    "data/deploy_audit.json",
]

def load_deploy_time_utc() -> str:
    for p in AUDIT_CAND:
        try:
            if p.startswith(("http://", "https://")):
                obj = pd.read_json(p, typ="series").to_dict()
            elif os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    obj = json.load(f)
            else:
                continue
            if isinstance(obj, dict) and obj.get("deploy_time_utc"):
                return str(obj["deploy_time_utc"])
        except Exception:
            continue
    return "-"

@st.cache_data(show_spinner=False)
def _cached_deploy_time() -> str:
    return load_deploy_time_utc()

DEPLOY_TIME = _cached_deploy_time()

# ---------------------------
# 4) Router (query param)
# ---------------------------
PAGES = {
    "home": "🏠 Ana Sayfa",
    "map": "🗺️ Anlık Risk Haritası",
    "forecast": "📊 Suç & Suç Zararı Tahmini",
    "patrol": "👮 Devriye Planlama",
    "reports": "📄 Raporlar & Kolluğa Öneriler",
}

PAGE_FILES = {
    "map": "pages/1_Anlik_Risk_Haritasi.py",
    "forecast": "pages/2_Suc_Zarar_Tahmini.py",
    "patrol": "pages/3_Devriye_Planlama.py",
    "reports": "pages/4_Raporlar_Oneriler.py",
}

def get_current_page() -> str:
    p = st.query_params.get("p", "home")
    return p if p in PAGES else "home"

def set_page(p: str):
    st.query_params["p"] = p
    st.rerun()

def render_corporate_sidebar(active_key: str):
    st.sidebar.markdown("## Kurumsal Menü")
    try:
        sf_now = datetime.now(ZoneInfo("America/Los_Angeles"))
        st.sidebar.caption(f"🕒 {sf_now:%Y-%m-%d %H:%M:%S} (SF)")
    except Exception:
        st.sidebar.caption(f"🕒 {datetime.now():%Y-%m-%d %H:%M:%S}")

    st.sidebar.caption(f"Son güncelleme: {DEPLOY_TIME}")
    st.sidebar.divider()

    for key, label in PAGES.items():
        if key == active_key:
            st.sidebar.button(label, use_container_width=True, disabled=True)
        else:
            if st.sidebar.button(label, use_container_width=True):
                set_page(key)

def run_page_file(path: str):
    if not os.path.exists(path):
        st.error(f"Sayfa dosyası bulunamadı: `{path}`")
        return
    # streamlit page registry’ye ihtiyaç yok: dosyayı direkt çalıştır
    runpy.run_path(path, run_name="__main__")

# ---------------------------
# 5) Home (FAST)
# ---------------------------
def render_home():
    st.markdown("# SUTAM — Operasyon Paneli")
    st.markdown(
        f'<div class="sutam-caption">Zamansal–Mekânsal Suç Tahmini: Risk Analizi, Zarar Etkisi ve Devriye Önerisi • Son güncelleme: <b>{DEPLOY_TIME}</b></div>',
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown(
        """
        <div class="sutam-callout">
          <b>Bu uygulama ne yapar?</b><br/>
          Geçmiş suç olayları ve bağlamsal göstergelerden yararlanarak şehir genelinde <b>göreli risk düzeylerini</b> üretir ve
          devriye planlama süreçlerine <b>karar destek</b> sağlar.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        st.markdown(
            """
            <div class="sutam-card">
              <div class="sutam-card-title">🗺️ Anlık Risk Haritası</div>
              <p class="sutam-card-text">5’li risk seviyesi ile sıcak bölgeleri hızlıca görselleştirir.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="sutam-card">
              <div class="sutam-card-title">📊 Suç & Suç Zarar Tahmini</div>
              <p class="sutam-card-text">Olasılık ve beklenen etkiyi birlikte değerlendirir.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="sutam-card">
              <div class="sutam-card-title">👮 Devriye Planlama</div>
              <p class="sutam-card-text">Top-K kapasiteye göre öncelik önerir.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """
            <div class="sutam-card">
              <div class="sutam-card-title">📄 Raporlar & Öneriler</div>
              <p class="sutam-card-text">Özet çıktı ve saha önerilerini indirilebilir sunar.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # İstersen burada kurumsal bir görsel göster:
    img = "assets/risk_map_preview.png"
    if os.path.exists(img):
        st.write("")
        st.image(img, caption="Anlık Risk Haritası — Ön izleme", use_container_width=True)

    st.write("")
    st.divider()

    st.subheader("⚖️ Etik ve Sorumlu Kullanım Notları")
    st.markdown(
        """
        <div class="sutam-ethics">
          <ul style="margin: 0 0 0 1.15rem;">
            <li>Çıktılar <b>bağlayıcı değildir</b>; nihai karar her zaman <b>insan değerlendirmesine</b> aittir.</li>
            <li>Sistem <b>bireyleri hedeflemez</b>; yalnızca mekânsal-zamansal örüntüler üzerinden risk farkındalığı sağlar.</li>
            <li>Risk seviyeleri <b>olasılıksal</b> göstergelerdir; saha bilgisiyle birlikte yorumlanmalıdır.</li>
          </ul>
          <div class="sutam-muted" style="margin-top: 8px;">
            Not: Teknik ayrıntılar analist odaklı raporlamada sunulur.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------
# 6) Render
# ---------------------------
current_page = get_current_page()
render_corporate_sidebar(current_page)

if current_page == "home":
    render_home()
else:
    run_page_file(PAGE_FILES[current_page])
