# app.py — SUTAM (FULL REVIZE • kurumsal sidebar • 60sn saat • hızlı açılış • page_link yok)
# ✅ Bu sürüm: senin mevcut dosya adlarınla uyumlu
# pages/
#   Anlik_Risk_Haritasi.py
#   Suc_Zarar_Tahmini.py
#   Devriye_Planlama.py
#   Raporlar_Oneriler.py

from __future__ import annotations

import os
import json
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
import pandas as pd

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------
# 0) Page config (FIRST)
# ---------------------------
st.set_page_config(
    page_title="SUTAM — Operasyon Paneli",
    page_icon="🛰️",
    layout="wide",
)

# ---------------------------
# 1) Optional autorefresh (60s)
# ---------------------------
def enable_autorefresh_60s():
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=60_000, key="sutam_clock_refresh")
    except Exception:
        # paket yoksa sessiz geç
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

          /* ✅ Streamlit default Pages nav ("app" ve liste) gizle */
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
# 4) Import page modules (LAZY + debug-friendly)
#    ✅ Dosya adlarınla uyumlu: pages/Anlik_Risk_Haritasi.py vb.
#    ✅ Hata olursa ekrana traceback basar (gizlemez)
# ---------------------------
def _safe_import(module_path: str, func_name: str):
    try:
        mod = __import__(module_path, fromlist=[func_name])
        fn = getattr(mod, func_name)
        if not callable(fn):
            raise TypeError(f"{module_path}.{func_name} callable değil.")
        return fn, None
    except Exception:
        return None, traceback.format_exc()

render_anlik_risk_haritasi, err_map = _safe_import(
    "pages.Anlik_Risk_Haritasi", "render_anlik_risk_haritasi"
)

# İstersen diğer sayfaları da modüler bağlarız (şimdilik placeholder)
# render_suc_zarar_tahmini, err_fc = _safe_import("pages.Suc_Zarar_Tahmini", "render_suc_zarar_tahmini")
# render_devriye_planlama, err_pt = _safe_import("pages.Devriye_Planlama", "render_devriye_planlama")
# render_raporlar_oneriler, err_rp = _safe_import("pages.Raporlar_Oneriler", "render_raporlar_oneriler")

# ---------------------------
# 5) Simple internal navigation (no page_link)
#    - URL query param: ?p=home/map/forecast/patrol/reports
# ---------------------------
PAGES = {
    "home": "🏠 Ana Sayfa",
    "map": "🗺️ Anlık Risk Haritası",
    "forecast": "📊 Suç & Suç Zararı Tahmini",
    "patrol": "👮 Devriye Planlama",
    "reports": "📄 Raporlar & Kolluğa Öneriler",
}

def get_current_page() -> str:
    q = st.query_params
    p = q.get("p", "home")
    return p if p in PAGES else "home"

def set_page(p: str):
    st.query_params["p"] = p
    st.rerun()

# ---------------------------
# 6) Sidebar (ONLY 5 items + live clock)
# ---------------------------
def render_corporate_sidebar(active_key: str):
    st.sidebar.markdown("## Kurumsal Menü")

    # SF time (kolluk dili)
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

current_page = get_current_page()
render_corporate_sidebar(current_page)

# ---------------------------
# 7) Page renderers (Home is FAST)
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
              <div class="sutam-card-title">📊 Suç & Suç Zararı Tahmini</div>
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
              <p class="sutam-card-text">Risk/zarar odaklı devriye önceliklendirmesi sunar.</p>
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

    st.write("")
    st.divider()

    st.subheader("⚖️ Etik ve Sorumlu Kullanım Notları")
    st.markdown(
        """
        <div class="sutam-ethics">
          <ul style="margin: 0 0 0 1.15rem;">
            <li>Çıktılar <b>bağlayıcı değildir</b>; nihai karar her zaman <b>insan değerlendirmesine</b> aittir.</li>
            <li>Sistem <b>bireyleri hedeflemez</b>; yalnızca mekânsal-zamansal örüntüler üzerinden risk farkındalığı sağlar.</li>
            <li>Risk seviyeleri <b>olasılıksal</b> göstergelerdir; yerel koşullar ve saha bilgisiyle birlikte yorumlanmalıdır.</li>
          </ul>
          <div class="sutam-muted" style="margin-top: 8px;">
            Not: Teknik performans metrikleri ve model ayrıntıları analist odaklı raporlamada sunulur.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.divider()

def render_placeholder(title: str):
    st.markdown(f"# {title}")
    st.info("Bu sayfa modüler şekilde eklenecek. Şimdilik navigasyon ve kurumsal tasarım tamam.")

# ---------------------------
# 8) Router
# ---------------------------
if current_page == "home":
    render_home()

elif current_page == "map":
    if render_anlik_risk_haritasi is None:
        render_placeholder(PAGES["map"])
        st.error("Harita modülü yüklenemedi. `pages/Anlik_Risk_Haritasi.py` dosyasını kontrol edin.")
        if err_map:
            st.caption("Import hatası (debug):")
            st.code(err_map)
    else:
        render_anlik_risk_haritasi()

elif current_page == "forecast":
    render_placeholder(PAGES["forecast"])

elif current_page == "patrol":
    render_placeholder(PAGES["patrol"])

elif current_page == "reports":
    render_placeholder(PAGES["reports"])

else:
    render_home()
