# app.py — SUTAM (Kurumsal Ana Sayfa)
import os
import json
import streamlit as st
import pandas as pd

# ---------------------------
# Page config (FIRST)
# ---------------------------
st.set_page_config(
    page_title="SUTAM — Operasyon Paneli",
    page_icon="🛰️",
    layout="wide",
)

# ---------------------------
# Corporate CSS (Inter-like)
# ---------------------------
def apply_corporate_style():
    st.markdown(
        """
        <style>
          /* --- Typography --- */
          html, body, [class*="css"]  {
            font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
            color: #0f172a; /* slate-900 */
          }

          /* Main container spacing */
          .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
            max-width: 1200px;
          }

          /* Headings */
          h1, h2, h3 {
            letter-spacing: -0.02em;
          }
          h1 { font-size: 1.65rem; margin-bottom: .25rem; }
          h2 { font-size: 1.15rem; margin-top: 1.2rem; }
          p, li { font-size: 0.95rem; line-height: 1.5; }

          /* Subtle caption */
          .sutam-caption {
            color: #475569; /* slate-600 */
            font-size: 0.9rem;
            margin-top: 0.15rem;
          }

          /* Cards */
          .sutam-card {
            border: 1px solid #e2e8f0; /* slate-200 */
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
            color: #334155; /* slate-700 */
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
          .sutam-muted {
            color: #64748b;
            font-size: 0.88rem;
          }

          /* Sidebar */
          section[data-testid="stSidebar"] {
            border-right: 1px solid #e2e8f0;
          }

          /* Buttons (subtle) */
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

# ---------------------------
# Data update badge (optional)
# If you have deploy_audit.json use it; otherwise show "-" safely
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
            if p.startswith("http://") or p.startswith("https://"):
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

DEPLOY_TIME = load_deploy_time_utc()

# ---------------------------
# Sidebar (minimal, corporate)
# ---------------------------
st.sidebar.title("⚙️ Menü")
st.sidebar.caption(f"DATA_DIR: {DATA_DIR}")
st.sidebar.caption(f"Son güncelleme: {DEPLOY_TIME}")
st.sidebar.divider()

st.sidebar.markdown("**Sayfalar**")
st.sidebar.page_link("app.py", label="🏠 Ana Sayfa", icon="🏠")
# Bu sayfaları gerçekten kullanacaksan pages/ altında oluştur:
# st.sidebar.page_link("pages/1_🗺️_Anlık_Risk_Haritası.py", "🗺️ Anlık Risk Haritası")
# st.sidebar.page_link("pages/2_📊_Suç_ve_Zarar_Tahmini.py", "📊 Suç & Suç Zararı Tahmini")
# st.sidebar.page_link("pages/3_👮_Devriye_Planlama.py", "👮 Devriye Planlama")
# st.sidebar.page_link("pages/4_📄_Raporlar.py", "📄 Raporlar & Öneriler")

# ---------------------------
# HOME PAGE
# ---------------------------
# Hero
st.markdown("# SUTAM — Suç Risk Karar Destek Paneli")
st.markdown(
    f'<div class="sutam-caption">Kolluk operasyonları için mekânsal-zamansal risk farkındalığı • Son güncelleme: <b>{DEPLOY_TIME}</b></div>',
    unsafe_allow_html=True,
)

st.write("")  # whitespace

# Intro callout (short, institutional)
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

# Cards: What you can do (max 4, short)
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

# Ethics + Responsible use (short, visible)
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
        Not: Teknik performans metrikleri ve model ayrıntıları “Raporlar” bölümünde (analist odaklı) sunulur.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
st.divider()

# Map preview (static image) + quick navigation
st.subheader("🗺️ Anlık Risk Haritası — Ön İzleme")

left, right = st.columns([1.35, 1], gap="large")

with left:
    # Put your preview image here (recommended):
    # assets/risk_map_preview.png  (or jpg)
    preview_path_candidates = [
        "assets/risk_map_preview.png",
        "assets/risk_map_preview.jpg",
        "assets/risk_map_preview.jpeg",
    ]
    preview_path = next((p for p in preview_path_candidates if os.path.exists(p)), None)

    if preview_path:
        st.image(preview_path, use_container_width=True)
    else:
        st.info(
            "Ön izleme görseli eklemek için `assets/risk_map_preview.png` dosyasını repoya koy.\n\n"
            "Geçici olarak bu alan boş bırakıldı."
        )

    st.markdown(
        '<div class="sutam-muted">Harita, risk düzeylerini 5’li ölçekle (Düşük → Çok Yüksek) gösterir. Etkileşimli analiz için ilgili sayfaya geçiniz.</div>',
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        """
        <div class="sutam-card">
          <div class="sutam-card-title">Hızlı Erişim</div>
          <p class="sutam-card-text">Detaylı inceleme için aşağıdaki sayfalara geçin.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    # Eğer pages/ dosyalarını oluşturduysan aşağıdaki page_link'ler aktif olur.
    # Şimdilik butonlar placeholder; sayfalar gelince page_link ile değiştir.
    go_map = st.button("🗺️ Anlık Risk Haritasına Git", use_container_width=True)
    go_plan = st.button("👮 Devriye Planlamaya Git", use_container_width=True)
    go_reports = st.button("📄 Raporlara Git", use_container_width=True)

    if go_map:
        st.info("Sayfa oluşturunca: pages/1_🗺️_Anlık_Risk_Haritası.py → st.page_link ile bağlayacağız.")
    if go_plan:
        st.info("Sayfa oluşturunca: pages/3_👮_Devriye_Planlama.py → st.page_link ile bağlayacağız.")
    if go_reports:
        st.info("Sayfa oluşturunca: pages/4_📄_Raporlar.py → st.page_link ile bağlayacağız.")

# Optional: advanced diagnostics hidden
with st.expander("🧪 Gelişmiş Tanılama (Analist)", expanded=False):
    st.write(
        {
            "DATA_DIR": DATA_DIR,
            "deploy_time_utc": DEPLOY_TIME,
            "audit_candidates": AUDIT_CAND,
        }
    )
