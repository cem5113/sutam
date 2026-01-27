# app.py — SUTAM (FULL REVIZE • kurumsal sidebar • 60sn saat • hızlı ana sayfa)
from __future__ import annotations

import os
import json
from datetime import datetime

import streamlit as st
import pandas as pd

# ============================
# 0) Page config (FIRST)
# ============================
st.set_page_config(
    page_title="SUTAM — Operasyon Paneli",
    page_icon="🛰️",
    layout="wide",
)

# ============================
# 1) Optional autorefresh (60s)
#    - sadece ana sayfada saat/rozet güncellensin
# ============================
# Not: st_autorefresh eklentisi yoksa otomatik devre dışı kalır.
def enable_autorefresh_60s():
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=60_000, key="sutam_clock_refresh")
    except Exception:
        # paket yoksa sorun değil; saat sadece rerun ile güncellenir
        pass

enable_autorefresh_60s()

# ============================
# 2) Corporate CSS
# ============================
def apply_corporate_style():
    st.markdown(
        """
        <style>
          /* --- Typography --- */
          html, body, [class*="css"]  {
            font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
            color: #0f172a; /* slate-900 */
          }

          /* Main container */
          .block-container {
            padding-top: 1.15rem;
            padding-bottom: 2.5rem;
            max-width: 1200px;
          }

          /* Headings */
          h1, h2, h3 { letter-spacing: -0.02em; }
          h1 { font-size: 1.65rem; margin-bottom: .25rem; }
          h2 { font-size: 1.15rem; margin-top: 1.1rem; }
          p, li { font-size: 0.95rem; line-height: 1.5; }

          .sutam-caption {
            color: #475569; /* slate-600 */
            font-size: 0.90rem;
            margin-top: 0.15rem;
          }
          .sutam-muted {
            color: #64748b;
            font-size: 0.88rem;
          }

          /* Cards */
          .sutam-card {
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 14px 14px;
            background: #ffffff;
            box-shadow: 0 1px 0 rgba(15, 23, 42, 0.03);
          }
          .sutam-card-title {
            font-weight: 700;
            font-size: 0.98rem;
            margin-bottom: 0.25rem;
            color: #0f172a;
          }
          .sutam-card-text {
            color: #334155;
            font-size: 0.92rem;
            margin: 0;
          }

          /* Callouts */
          .sutam-callout {
            border-left: 4px solid #2563eb; /* blue-600 */
            background: #eff6ff; /* blue-50 */
            padding: 12px 14px;
            border-radius: 10px;
            color: #0f172a;
          }
          .sutam-ethics {
            border-left: 4px solid #64748b; /* slate-500 */
            background: #f8fafc; /* slate-50 */
            padding: 12px 14px;
            border-radius: 10px;
            color: #0f172a;
          }

          /* Sidebar border */
          section[data-testid="stSidebar"] { border-right: 1px solid #e2e8f0; }

          /* Buttons */
          .stButton button {
            border-radius: 10px !important;
            padding: 0.55rem 0.85rem !important;
            font-weight: 600 !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_corporate_style()

# ============================
# 3) Hide Streamlit default Pages nav ("app" yazısı dahil)
# ============================
def hide_streamlit_default_nav():
    st.markdown(
        """
        <style>
          /* Streamlit default sidebar navigation (üstte app + sayfa listesi) */
          [data-testid="stSidebarNav"] { display: none !important; }
          section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

hide_streamlit_default_nav()

# ============================
# 4) Fast "last update" badge (VERY LIGHT)
#    - ana sayfa hızlı açılsın diye sadece küçük json okunur (varsa)
# ============================
DATA_DIR = os.getenv("DATA_DIR", "data").rstrip("/")
AUDIT_CAND = [
    f"{DATA_DIR}/deploy_audit.json",
    "deploy/deploy_audit.json",
    "data/deploy_audit.json",
]

def load_deploy_time_utc() -> str:
    for p in AUDIT_CAND:
        try:
            if p.startswith("http://") or p.startswith("https://"):
                # remote json (hafif ama url ise yavaş olabilir)
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

# Cache: aynı rerun içinde tekrar okumasın (hız)
@st.cache_data(show_spinner=False)
def _cached_deploy_time() -> str:
    return load_deploy_time_utc()

DEPLOY_TIME = _cached_deploy_time()

# ============================
# 5) Corporate Sidebar (ONLY requested items + clock)
# ============================
def render_corporate_sidebar():
    st.sidebar.markdown("## Menü")
    st.sidebar.caption(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.sidebar.caption(f"Son güncelleme: {DEPLOY_TIME}")
    st.sidebar.divider()

    # ✅ Sadece istenen 5 link
    st.sidebar.page_link("app.py", label="🏠 Ana Sayfa", icon="🏠")

    # Aşağıdaki dosya adlarını SENİN pages/ dosyalarınla birebir eşleştir
    st.sidebar.page_link("pages/1_Anlik_Risk_Haritasi.py", label="🗺️ Anlık Risk Haritası", icon="🗺️")
    st.sidebar.page_link("pages/2_Suc_ve_Zarar_Tahmini.py", label="📊 Suç & Suç Zararı Tahmini", icon="📊")
    st.sidebar.page_link("pages/3_Devriye_Planlama.py", label="👮 Devriye Planlama", icon="👮")
    st.sidebar.page_link("pages/4_Raporlar_ve_Oneriler.py", label="📄 Raporlar & Kolluğa Öneriler", icon="📄")

render_corporate_sidebar()

# ============================
# 6) HOME (FAST • no heavy data load)
# ============================
st.markdown("# SUTAM — Suç Risk Karar Destek Paneli")
st.markdown(
    f'<div class="sutam-caption">Kolluk operasyonları için mekânsal-zamansal risk farkındalığı • Son güncelleme: <b>{DEPLOY_TIME}</b></div>',
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
          <div class="sutam-card-title">📊 Suç & Zarar Tahmini</div>
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
          <p class="sutam-card-text">Top-K yaklaşımıyla kapasiteye göre öncelik önerir.</p>
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
        Not: Teknik performans metrikleri ve model ayrıntıları “Raporlar & Kolluğa Öneriler” bölümünde (analist odaklı) sunulur.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
st.divider()

st.subheader("🗺️ Anlık Risk Haritası — Ön İzleme")

# Hız için: local görsel göster (assets/)
preview_path_candidates = [
    "assets/risk_map_preview.png",
    "assets/risk_map_preview.jpg",
    "assets/risk_map_preview.jpeg",
]
preview_path = next((p for p in preview_path_candidates if os.path.exists(p)), None)

colL, colR = st.columns([1.45, 1], gap="large")

with colL:
    if preview_path:
        st.image(preview_path, use_container_width=True)
        st.markdown(
            '<div class="sutam-muted">Harita, risk düzeylerini 5’li ölçekle (Düşük → Çok Yüksek) gösterir. Etkileşimli analiz için “Anlık Risk Haritası” sayfasına geçiniz.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Ön izleme için `assets/risk_map_preview.png` ekleyin.")

with colR:
    st.markdown(
        """
        <div class="sutam-card">
          <div class="sutam-card-title">Kullanım</div>
          <p class="sutam-card-text">Detaylı inceleme ve filtreleme için sol menüden ilgili sayfayı seçiniz.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(
        """
        <div class="sutam-muted">
          • Harita: 5’li risk bandı (Düşük/Orta/Yüksek)<br/>
          • Devriye: kapasiteye göre Top-K öncelik<br/>
          • Raporlar: özet + saha önerileri
        </div>
        """,
        unsafe_allow_html=True,
    )
