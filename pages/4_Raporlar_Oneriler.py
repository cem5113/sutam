# pages/4_Raporlar_Oneriler.py
import os
import streamlit as st
import pandas as pd

DATA_DIR = os.getenv("DATA_DIR", "data")
OPS_DIR  = os.path.join(DATA_DIR, "ops")

st.title("📄 Raporlar & Operasyonel Öneriler (BLOK-9 uyumlu)")

if not os.path.exists(OPS_DIR):
    st.warning(f"ops klasörü yok: {OPS_DIR}\nBLOK-9 çıktılarını data/ops/ içine koy.")
    st.stop()

files = sorted([f for f in os.listdir(OPS_DIR) if not f.startswith(".")])
if not files:
    st.warning("ops klasörü boş.")
    st.stop()

st.subheader("Mevcut BLOK-9 çıktıları")
sel = st.selectbox("Dosya seç", files, index=0)
path = os.path.join(OPS_DIR, sel)

if sel.lower().endswith(".csv"):
    df = pd.read_csv(path)
    st.dataframe(df, use_container_width=True, height=560)
    st.download_button("⬇️ CSV indir", data=open(path,"rb").read(), file_name=sel, mime="text/csv")

elif sel.lower().endswith(".md") or sel.lower().endswith(".txt"):
    txt = open(path, "r", encoding="utf-8", errors="ignore").read()
    if sel.lower().endswith(".md"):
        st.markdown(txt)
    else:
        st.text(txt)
    st.download_button("⬇️ Dosyayı indir", data=open(path,"rb").read(), file_name=sel)

elif sel.lower().endswith(".pdf"):
    st.info("PDF görüntüleme: indirip açabilirsiniz.")
    st.download_button("⬇️ PDF indir", data=open(path,"rb").read(), file_name=sel, mime="application/pdf")
else:
    st.info("Bu dosya tipi doğrudan gösterilemiyor. İndirebilirsiniz.")
    st.download_button("⬇️ İndir", data=open(path,"rb").read(), file_name=sel)
