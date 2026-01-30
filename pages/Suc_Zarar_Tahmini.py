# pages/3_🧭_Suç_Tahmini.py
# SUTAM — Suç Tahmini (Haritalı görünüm: GEOID + centroid/geojson)
# - 3 Saatlik Bloklar (≤7 gün) ve Günlük (≤365 gün)
# - Kaynak: GitHub Actions artifact: "sf-crime-outputs-parquet"
#   Üyeler: risk_3h_next7d_top3 / risk_daily_next365d_top5
# - Fallback: data/crime_forecast_7days_all_geoids_FRstyle.csv (yerel) veya GitHub raw
# - NOT: Bu sayfa SADECE risk (suç tahmini) içindir. HARM/Weather ayrı sayfaya taşınacak.

from __future__ import annotations

import os
import io
import json
import posixpath
import zipfile
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st

import folium
from streamlit_folium import st_folium

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


# =========================================================
# ⚙️ GitHub repo + artifact bilgisi
# =========================================================
REPOSITORY_OWNER = "cem5113"
REPOSITORY_NAME = "crime_prediction_data"
ARTIFACT_NAME_SHOULD_CONTAIN = "sf-crime-outputs-parquet"

# Artifact içindeki beklenen üyeler (stem / dosya adı gövdesi)
ARTIFACT_MEMBER_HOURLY = "risk_3h_next7d_top3"
ARTIFACT_MEMBER_DAILY  = "risk_daily_next365d_top5"

# 3-saatlik FR style CSV fallback (yerel) ve GitHub raw
CSV_HOURLY_FRSTYLE_LOCAL_1 = "data/crime_forecast_7days_all_geoids_FRstyle.csv"
CSV_HOURLY_FRSTYLE_LOCAL_2 = "crime_forecast_7days_all_geoids_FRstyle.csv"
CSV_HOURLY_FRSTYLE_RAW_URL = (
    "https://raw.githubusercontent.com/"
    "cem5113/crimepredict/main/crime_forecast_7days_all_geoids_FRstyle.csv"
)

# Harita geometri (yerel)
GEOJSON_LOCAL = "data/sf_cells.geojson"

# Neighbor + 911 kaynak (yerel)
SF_CRIME_09_CANDIDATES = [
    "sf_crime_09.csv",
    "data/sf_crime_09.csv",
    "/content/drive/MyDrive/sf_crime_09.csv",
]


# =========================================================
# 🔑 Token / Header
# =========================================================
def resolve_github_token() -> str | None:
    if os.getenv("GITHUB_TOKEN"):
        return os.getenv("GITHUB_TOKEN")
    for key in ("github_token", "GH_TOKEN", "GITHUB_TOKEN"):
        try:
            if key in st.secrets and st.secrets[key]:
                os.environ["GITHUB_TOKEN"] = str(st.secrets[key])
                return os.environ["GITHUB_TOKEN"]
        except Exception:
            pass
    return None


def github_api_headers() -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


# =========================================================
# 📦 Artifact ZIP alma (en güncel ve süresi dolmamış)
# =========================================================
def resolve_latest_artifact_zip_url(owner: str, repo: str, name_contains: str):
    token = resolve_github_token()
    if not token:
        return None, {}

    base = f"https://api.github.com/repos/{owner}/{repo}"
    r = requests.get(
        f"{base}/actions/artifacts?per_page=100",
        headers=github_api_headers(),
        timeout=60,
    )
    r.raise_for_status()
    artifacts = (r.json() or {}).get("artifacts", []) or []
    artifacts = [a for a in artifacts if (name_contains in a.get("name", "")) and not a.get("expired")]
    if not artifacts:
        return None, {}

    artifacts.sort(key=lambda a: a.get("updated_at", ""), reverse=True)
    url = f"{base}/actions/artifacts/{artifacts[0]['id']}/zip"
    return url, github_api_headers()


def read_member_from_zip_bytes(zip_bytes: bytes, member_stem: str) -> pd.DataFrame:
    """
    Artifact ZIP içinde:
      - önce doğrudan arar
      - yoksa iç ZIP'leri (ör. *.zip) açıp arar
    member_stem: "risk_3h_next7d_top3" gibi.
    """
    def read_any_table(raw_bytes: bytes, name_hint: str) -> pd.DataFrame:
        buf = BytesIO(raw_bytes)
        hint = name_hint.lower()
        if hint.endswith(".csv"):
            return pd.read_csv(buf)
        try:
            buf.seek(0)
            return pd.read_parquet(buf)
        except Exception:
            buf.seek(0)
            return pd.read_csv(buf)

    def scan_zip(zf: zipfile.ZipFile, stem: str) -> pd.DataFrame | None:
        names = zf.namelist()
        stemL = stem.lower()
        for n in names:
            bn = posixpath.basename(n).lower()
            # stem geçen ilk csv/parquet dosyasını al
            if stemL in bn and (bn.endswith(".parquet") or bn.endswith(".pq") or bn.endswith(".csv")):
                with zf.open(n) as f:
                    return read_any_table(f.read(), bn)
        # stem geçiyor ama uzantı farklıysa (nadir) yine dene
        for n in names:
            bn = posixpath.basename(n).lower()
            if stemL in bn:
                with zf.open(n) as f:
                    return read_any_table(f.read(), bn)
        return None

    with zipfile.ZipFile(BytesIO(zip_bytes)) as outer:
        df = scan_zip(outer, member_stem)
        if df is not None:
            return df

        # iç zip’leri tara
        for name in outer.namelist():
            if name.lower().endswith(".zip"):
                with outer.open(name) as fz:
                    inner_bytes = fz.read()
                try:
                    with zipfile.ZipFile(BytesIO(inner_bytes)) as inner:
                        df2 = scan_zip(inner, member_stem)
                        if df2 is not None:
                            return df2
                except zipfile.BadZipFile:
                    continue

    raise FileNotFoundError(f"ZIP içinde '{member_stem}' gövdesini içeren CSV/PARQUET bulunamadı.")


@st.cache_data(show_spinner=False)
def load_artifact_member(member_stem: str) -> pd.DataFrame:
    url, headers = resolve_latest_artifact_zip_url(
        REPOSITORY_OWNER, REPOSITORY_NAME, ARTIFACT_NAME_SHOULD_CONTAIN
    )
    if not url:
        raise RuntimeError("Artifact bulunamadı veya GITHUB_TOKEN yok.")
    r = requests.get(url, headers=headers, timeout=120, allow_redirects=True)
    r.raise_for_status()
    return read_member_from_zip_bytes(r.content, member_stem)


# =========================================================
# 🧭 Şema doğrulayıcılar (hourly/daily)
# =========================================================
def normalize_hourly_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    3-saatlik risk şeması:
      - date
      - geoid
      - risk_score / risk_prob / p_stack / prob / probability / score / risk
      - hour veya hour_range_3h / hour_range / hour_block

    timestamp = date + hour
    """
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in df.columns:
                return n
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    c_date = pick("date")
    c_geoid = pick("geoid", "GEOID", "cell_id", "id")
    c_risk_score = pick("risk_score", "score", "risk", "p_stack")
    c_risk_prob  = pick("risk_prob", "prob", "probability")  # opsiyonel
    c_hour = pick("hour", "hour_idx", "hour_of_day", "hour_index")
    c_hrange = pick("hour_range_3h", "hour_range", "hour_block")

    if not (c_date and c_geoid and (c_risk_score or c_risk_prob) and (c_hour or c_hrange)):
        raise ValueError("Saatlik veri için date, geoid, risk_* ve hour veya hour_range_3h zorunlu.")

    df["date"] = pd.to_datetime(df[c_date], errors="coerce")
    df["geoid"] = df[c_geoid].astype(str)

    if c_risk_score:
        df["risk_score"] = pd.to_numeric(df[c_risk_score], errors="coerce")
    else:
        df["risk_score"] = np.nan

    if c_risk_prob:
        df["risk_prob"] = pd.to_numeric(df[c_risk_prob], errors="coerce")
    else:
        df["risk_prob"] = np.nan

    if c_hour:
        df["hour"] = pd.to_numeric(df[c_hour], errors="coerce").astype("Int64").clip(0, 23)
    else:
        def parse_start_hour(val):
            if pd.isna(val):
                return np.nan
            s = str(val).strip().replace("–", "-").replace("—", "-")
            if "-" not in s:
                return np.nan
            a, _ = s.split("-", 1)
            try:
                h0 = int(a.strip())
                return max(0, min(23, h0))
            except Exception:
                return np.nan
        df["hour"] = df[c_hrange].map(parse_start_hour).astype("Int64")
        df["hour_range_3h"] = df[c_hrange].astype(str)

    df = df.dropna(subset=["date", "geoid", "hour"]).copy()
    df["timestamp"] = df["date"].dt.floor("D") + pd.to_timedelta(df["hour"].astype(int), unit="h")
    return df


def normalize_daily_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in df.columns:
                return n
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    c_date = pick("date")
    c_geoid = pick("geoid", "GEOID", "cell_id", "id")
    c_risk_score = pick("risk_score", "score", "risk", "p_stack")
    c_risk_prob  = pick("risk_prob", "prob", "probability")

    if not (c_date and c_geoid and (c_risk_score or c_risk_prob)):
        raise ValueError("Günlük veri için date, geoid ve risk_* zorunlu.")

    df["date"] = pd.to_datetime(df[c_date], errors="coerce").dt.floor("D")
    df["geoid"] = df[c_geoid].astype(str)

    if c_risk_score:
        df["risk_score"] = pd.to_numeric(df[c_risk_score], errors="coerce")
    else:
        df["risk_score"] = np.nan

    if c_risk_prob:
        df["risk_prob"] = pd.to_numeric(df[c_risk_prob], errors="coerce")
    else:
        df["risk_prob"] = np.nan

    df = df.dropna(subset=["date", "geoid"]).copy()
    return df


def normalize_geoid_for_map(df: pd.DataFrame) -> pd.DataFrame:
    """
    - geoid == 0 -> "0"
    - diğerleri -> 11 hane zfill
    """
    df = df.copy()
    if "geoid" not in df.columns:
        return df
    df["geoid"] = df["geoid"].astype(str)
    mask_city = df["geoid"].isin(["0", "0.0"])
    mask_cells = ~mask_city
    if mask_cells.any():
        df.loc[mask_cells, "geoid"] = (
            pd.to_numeric(df.loc[mask_cells, "geoid"], errors="coerce")
            .astype("Int64").astype(str).str.zfill(11)
        )
    if mask_city.any():
        df.loc[mask_city, "geoid"] = "0"
    return df


# =========================================================
# 🧮 Risk bucket (sabit eşikler)
# =========================================================
RISK_BUCKETS = [
    (0.00, 0.20, "Çok Düşük", [220, 220, 220, 160]),
    (0.20, 0.40, "Düşük",     [56, 168, 0, 200]),
    (0.40, 0.60, "Orta",      [255, 221, 0, 210]),
    (0.60, 0.80, "Yüksek",    [255, 140, 0, 220]),
    (0.80, 1.01, "Çok Yüksek",[160, 0, 0, 240]),
]
COLOR_MAP = {name: rgba for _, _, name, rgba in RISK_BUCKETS}

def bucket_of(v: float) -> str:
    x = 0.0 if pd.isna(v) else float(v)
    for lo, hi, name, _ in RISK_BUCKETS:
        if lo <= x < hi:
            return name
    return "Çok Düşük"

def rgba_to_hex(rgba):
    try:
        r, g, b, _ = rgba
        return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))
    except Exception:
        return "#dddddd"


# =========================================================
# 📌 GeoJSON load + enrich
# =========================================================
@st.cache_data(show_spinner=False)
def load_geojson() -> dict:
    if os.path.exists(GEOJSON_LOCAL):
        with open(GEOJSON_LOCAL, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _digits11(x) -> str:
    s = "".join(ch for ch in str(x) if ch.isdigit())
    return s.zfill(11) if s else ""

def enrich_geojson_with_risk(gj: dict, agg_df: pd.DataFrame) -> dict:
    if not gj or agg_df is None or agg_df.empty:
        return gj

    agg_df = agg_df.copy()
    agg_df["geoid"] = agg_df["geoid"].astype(str)
    risk_map = agg_df.set_index("geoid")

    feats_out = []
    for feat in gj.get("features", []):
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

        key = _digits11(raw)
        props["display_id"] = str(raw) if raw not in (None, "") else key
        props["geoid_norm"] = key

        props.setdefault("risk_bucket", "")
        props.setdefault("risk_mean_txt", "")
        props.setdefault("fill_color", [220, 220, 220, 160])

        if key and key in risk_map.index:
            row = risk_map.loc[key]
            b = row.get("risk_bucket", "") or bucket_of(row.get("risk_mean", 0))
            props["risk_bucket"] = str(b)
            props["fill_color"] = COLOR_MAP.get(b, [220, 220, 220, 160])

            try:
                r = float(row.get("risk_mean", np.nan))
                if r == r:
                    props["risk_mean_txt"] = f"{min(max(r, 0.0), 0.999):.3f}"
            except Exception:
                pass

        feats_out.append({**feat, "properties": props})

    return {**gj, "features": feats_out}


# =========================================================
# 📥 Veri yükleme
# =========================================================
@st.cache_data(show_spinner=False)
def load_hourly_dataframe() -> pd.DataFrame:
    # 1) yerel
    for path in (CSV_HOURLY_FRSTYLE_LOCAL_1, CSV_HOURLY_FRSTYLE_LOCAL_2):
        if os.path.exists(path):
            raw = pd.read_csv(path)
            return normalize_geoid_for_map(normalize_hourly_schema(raw))

    # 2) GitHub raw
    resp = requests.get(CSV_HOURLY_FRSTYLE_RAW_URL, timeout=60)
    resp.raise_for_status()
    raw = pd.read_csv(io.StringIO(resp.text))
    return normalize_geoid_for_map(normalize_hourly_schema(raw))


@st.cache_data(show_spinner=False)
def load_daily_dataframe() -> pd.DataFrame:
    raw = load_artifact_member(ARTIFACT_MEMBER_DAILY)
    return normalize_geoid_for_map(normalize_daily_schema(raw))


# =========================================================
# 🧾 sf_crime_09 → neighbor + 911 (seçili GEOID+hour için)
# =========================================================
def get_neighbors_and_911_from_sf_crime_09(selected_geoid: str, selected_hour: int) -> dict:
    path = next((p for p in SF_CRIME_09_CANDIDATES if os.path.exists(p)), None)
    if path is None:
        return {"error": "sf_crime_09.csv bulunamadı."}

    df = pd.read_csv(path, low_memory=False)

    # tarih
    if "date" not in df.columns:
        return {"error": "sf_crime_09.csv içinde 'date' kolonu yok."}
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    latest_real_date = df["date"].max()
    if pd.isna(latest_real_date):
        return {"error": "sf_crime_09.csv içinde geçerli tarih bulunamadı."}

    # GEOID kolonu
    geoid_col = "GEOID" if "GEOID" in df.columns else ("geoid" if "geoid" in df.columns else None)
    if not geoid_col:
        return {"error": "sf_crime_09.csv içinde GEOID/geoid kolonu yok."}

    # hour kolonu
    hour_col = "event_hour" if "event_hour" in df.columns else ("hour" if "hour" in df.columns else None)
    if not hour_col:
        return {"error": "sf_crime_09.csv içinde event_hour/hour kolonu yok."}

    g11 = str(selected_geoid).replace(".0", "").zfill(11)

    df[geoid_col] = (
        df[geoid_col].astype(str).str.replace(".0", "", regex=False).str.zfill(11)
    )
    df[hour_col] = pd.to_numeric(df[hour_col], errors="coerce").astype("Int64")

    sub = df[
        (df[geoid_col] == g11)
        & (df[hour_col] == int(selected_hour))
        & (df["date"] == latest_real_date)
    ].copy()

    if sub.empty:
        return {
            "geoid": g11,
            "date_used": str(latest_real_date.date()),
            "neighbor_1d": 0.0,
            "neighbor_3d": 0.0,
            "neighbor_7d": 0.0,
            "c911_24h": 0.0,
            "c911_3d": 0.0,
            "c911_7d": 0.0,
            "note": "Eşleşme yok (son gerçek tarih, GEOID+hour).",
        }

    # event_id/incident_id ile en küçük kaydı seç (yoksa ilk satır)
    sort_cols = []
    if "event_id" in sub.columns:
        sub["__eid"] = pd.to_numeric(sub["event_id"], errors="coerce")
        sort_cols = ["__eid"]
    elif "incident_id" in sub.columns:
        sub["__eid"] = pd.to_numeric(sub["incident_id"], errors="coerce")
        sort_cols = ["__eid"]

    if sort_cols:
        sub = sub.sort_values(sort_cols, ascending=True)
    row = sub.iloc[0]

    return {
        "geoid": g11,
        "date_used": str(latest_real_date.date()),
        "neighbor_1d": float(row.get("neighbor_crime_1d", 0) or 0),
        "neighbor_3d": float(row.get("neighbor_crime_3d", 0) or 0),
        "neighbor_7d": float(row.get("neighbor_crime_7d", 0) or 0),
        "c911_24h": float(row.get("911_request_count_daily(before_24_hours)", 0) or 0),
        "c911_3d": float(row.get("911_geo_hr_last3d", 0) or 0),
        "c911_7d": float(row.get("911_geo_hr_last7d", 0) or 0),
        "note": "sf_crime_09: son gerçek tarihte GEOID+hour ilk kayıt kullanıldı.",
    }


# =========================================================
# 🎛️ UI
# =========================================================
st.set_page_config(page_title="🧭 Suç Tahmini", layout="wide")
st.title("🧭 Suç Tahmini — Haritalı GEOID görünüm")
st.caption("3-saatlik (≤7 gün) veya günlük (≤365 gün) pencerede GEOID bazlı ortalama risk.")

st.sidebar.header("⚙️ Ayarlar")

mode = st.sidebar.radio(
    "Zaman çözünürlüğü",
    ["3 Saatlik Bloklar (≤7 gün)", "Günlük (≤365 gün)"],
    index=0,
)

# Saat blokları (3-saatlik mod)
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

def default_hour_block_label() -> str:
    fallback = "18–21"
    try:
        if ZoneInfo is None:
            return fallback
        now_sf = datetime.now(ZoneInfo("America/Los_Angeles"))
        h = now_sf.hour
        for label, (h0, h1) in hour_blocks.items():
            if h0 <= h <= h1:
                return label
        return fallback
    except Exception:
        return fallback

selected_hours = []
selected_label = None
if mode.startswith("3 Saatlik"):
    st.sidebar.subheader("Saat Aralığı")
    selected_label = st.sidebar.select_slider(
        "Saat aralığı",
        options=list(hour_blocks.keys()),
        value=default_hour_block_label(),
    )
    h0, h1 = hour_blocks[selected_label]
    selected_hours = list(range(h0, h1 + 1))

# Tarih aralığı (SF yerel)
if ZoneInfo is not None:
    now_sf = datetime.now(ZoneInfo("America/Los_Angeles"))
else:
    now_sf = datetime.utcnow()

max_days = 7 if mode.startswith("3 Saatlik") else 365
st.sidebar.caption(
    f"{'3 Saatlik' if max_days == 7 else 'Günlük'} görünümde en fazla {max_days} gün seçebilirsiniz "
    "(San Francisco yerel zamanı)."
)

d_start_default = now_sf.date()
d_end_default = now_sf.date()

d_start = st.sidebar.date_input("Başlangıç tarihi", value=d_start_default)
d_end = st.sidebar.date_input("Bitiş tarihi", value=d_end_default)

if (pd.to_datetime(d_end) - pd.to_datetime(d_start)).days > max_days:
    d_end = (pd.to_datetime(d_start) + pd.Timedelta(days=max_days)).date()
    st.sidebar.warning(f"Seçim {max_days} günü aşamaz; bitiş {d_end} olarak güncellendi.")

# GEOID filtre
geof_txt = st.sidebar.text_input("GEOID filtre (virgülle ayır)", value="")
geoids_sel = [g.strip() for g in geof_txt.split(",") if g.strip()]

# Top-K
top_k = st.sidebar.slider("Top-K (tablo)", 10, 200, 50, step=10)


# =========================================================
# 📥 Veri yükle + filtrele
# =========================================================
view_df = pd.DataFrame()
view_df_city = pd.DataFrame()
view_df_cells = pd.DataFrame()
agg = pd.DataFrame()
time_col = "timestamp"

with st.spinner("Veriler yükleniyor…"):
    if mode.startswith("3 Saatlik"):
        src = load_hourly_dataframe()
        t0 = pd.to_datetime(d_start)
        t1 = pd.to_datetime(d_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        df = src[(src["timestamp"] >= t0) & (src["timestamp"] <= t1)].copy()
        if geoids_sel:
            df = df[df["geoid"].isin(geoids_sel)].copy()
        if selected_hours:
            df = df[df["hour"].isin(selected_hours)].copy()

        if df.empty:
            st.info("Seçilen tarih/saat aralığı için kayıt bulunamadı; en güncel saatlik risk çıktısı gösteriliyor.")
            df = src.copy()

        view_df = df
        time_col = "timestamp"
    else:
        src = load_daily_dataframe()
        t0 = pd.to_datetime(d_start).floor("D")
        t1 = pd.to_datetime(d_end).floor("D")

        df = src[(src["date"] >= t0) & (src["date"] <= t1)].copy()
        if geoids_sel:
            df = df[df["geoid"].isin(geoids_sel)].copy()

        if df.empty:
            st.info("Seçilen tarih aralığı için kayıt bulunamadı; en güncel günlük risk çıktısı gösteriliyor.")
            df = src.copy()

        view_df = df
        time_col = "date"

    if len(view_df):
        mask_city = view_df["geoid"].astype(str) == "0"
        view_df_city = view_df[mask_city].copy()
        view_df_cells = view_df[~mask_city].copy()

        if len(view_df_cells):
            # risk_mean: öncelik risk_prob, yoksa risk_score
            metric_col = "risk_prob" if "risk_prob" in view_df_cells.columns and view_df_cells["risk_prob"].notna().any() else "risk_score"
            tmp = view_df_cells.copy()
            tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors="coerce")
            grp = tmp.groupby("geoid", as_index=False)[metric_col].mean()

            # 0–1 normalize
            if metric_col == "risk_prob":
                grp["risk_mean"] = grp[metric_col].clip(0.0, 1.0)
            else:
                max_val = grp[metric_col].max()
                if pd.notna(max_val) and max_val > 1.0:
                    grp["risk_mean"] = grp[metric_col].clip(0, 100) / 100.0
                else:
                    grp["risk_mean"] = grp[metric_col].clip(0.0, 1.0)

            agg = grp[["geoid", "risk_mean"]].copy()
            agg["geoid"] = agg["geoid"].astype(str)
        else:
            agg = pd.DataFrame()


# =========================================================
# 🗺️ Harita
# =========================================================
st.subheader("🗺️ Harita — 5 seviye risk renklendirme")

geojson = load_geojson()
clicked_geoid = None

if len(agg):
    agg["risk_bucket"] = agg["risk_mean"].map(bucket_of)
    agg_sorted = agg.sort_values("risk_mean", ascending=False).reset_index(drop=True)
else:
    agg_sorted = agg

if not len(agg_sorted):
    if len(view_df_city):
        st.info("Bu aralıkta sadece şehir geneli (GEOID=0) var; hücre bazlı risk olmadığı için harita devre dışı.")
    else:
        st.info("Seçilen aralıkta GEOID bazlı risk verisi bulunamadı.")
elif not geojson:
    st.info("GeoJSON (data/sf_cells.geojson) bulunamadı; harita devre dışı.")
else:
    gj_enriched = enrich_geojson_with_risk(geojson, agg_sorted)

    st.markdown(
        "**Lejand:** "
        "<span style='background:#dcdcdc;padding:2px 6px;border-radius:4px;'>Çok Düşük</span> "
        "<span style='background:#38a800;padding:2px 6px;border-radius:4px;'>Düşük</span> "
        "<span style='background:#ffdd00;padding:2px 6px;border-radius:4px;'>Orta</span> "
        "<span style='background:#ff8c00;padding:2px 6px;border-radius:4px;'>Yüksek</span> "
        "<span style='background:#a00000;padding:2px 6px;border-radius:4px;'>Çok Yüksek</span>",
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

    def highlight_fn(_feature):
        return {"weight": 2, "color": "#000000"}

    # Tooltip: boş satır yok, sade alanlar
    tooltip = folium.GeoJsonTooltip(
        fields=["display_id", "risk_bucket", "risk_mean_txt"],
        aliases=["GEOID:", "Risk seviyesi:", "Ortalama risk (0–1):"],
        sticky=True,
    )

    folium.GeoJson(
        gj_enriched,
        name="Risk",
        style_function=style_fn,
        highlight_function=highlight_fn,
        tooltip=tooltip,
    ).add_to(m)

    folium_ret = st_folium(
        m,
        width=None,
        height=520,
        returned_objects=["last_active_drawing"],
        key="sutam_risk_map",
    )

    if folium_ret and folium_ret.get("last_active_drawing"):
        props = (folium_ret["last_active_drawing"].get("properties", {}) or {})
        clicked_geoid = str(
            props.get("geoid_norm") or props.get("display_id") or props.get("geoid") or props.get("GEOID") or ""
        ).strip()
        if clicked_geoid:
            st.session_state["clicked_geoid_risk"] = clicked_geoid


# =========================================================
# 🧠 Özet / Sekmeler
# =========================================================
def geoid_label(g: str) -> str:
    return "Şehir geneli (GEOID=0)" if str(g) == "0" else str(g)

# GEOID seçenekleri
options = []
if len(view_df_city):
    options.append("0")
if len(view_df_cells):
    options.extend(sorted(view_df_cells["geoid"].astype(str).unique().tolist()))

selected_geoid = None
if options:
    clicked_geoid = st.session_state.get("clicked_geoid_risk", clicked_geoid)
    default_index = options.index(clicked_geoid) if clicked_geoid in options else 0
    selected_geoid = st.selectbox(
        "Detay göstermek için GEOID seç:",
        options=options,
        index=default_index,
        format_func=geoid_label,
    )

topk = agg_sorted.head(top_k).copy() if len(agg_sorted) else pd.DataFrame()

tab1, tab2, tab3 = st.tabs(["Özet", "Zaman Serisi", "Top-K"])


with tab1:
    st.subheader("📌 Seçili GEOID özeti")

    if selected_geoid is None or len(view_df) == 0:
        st.info("Görüntülenecek veri bulunamadı.")
    else:
        df_sel = view_df[view_df["geoid"] == selected_geoid].sort_values(time_col).copy()
        if df_sel.empty:
            st.info("Seçili GEOID için veri yok.")
        else:
            latest = df_sel.iloc[-1]

            def gv(col, default=np.nan):
                return latest[col] if col in df_sel.columns and pd.notna(latest[col]) else default

            c1, c2, c3 = st.columns(3)
            c1.metric("GEOID", geoid_label(selected_geoid))

            rs = gv("risk_score", np.nan)
            c2.metric("Risk skoru", f"{rs:.4f}" if rs == rs else "—")

            rp = gv("risk_prob", np.nan)
            if rp == rp:
                c3.metric("Risk olasılığı", f"{rp:.4f}")
            else:
                # agg'den ortalama risk
                if len(agg_sorted) and selected_geoid != "0" and (agg_sorted["geoid"] == selected_geoid).any():
                    rm = float(agg_sorted.loc[agg_sorted["geoid"] == selected_geoid, "risk_mean"].iloc[0])
                    c3.metric("Ortalama risk", f"{rm:.4f}")
                else:
                    c3.metric("Ortalama risk", "—")

            # 3-saatlik modda saat bilgisi
            if mode.startswith("3 Saatlik") and "hour" in df_sel.columns:
                st.caption(f"Seçili saat bloğu: {selected_label or '—'} | Son kayıt saati: {int(gv('hour', 0))}")

            # sf_crime_09 → neighbor + 911 (yalnızca hücreler için anlamlı)
            if selected_geoid != "0":
                hour_for_lookup = int(gv("hour", 0)) if "hour" in df_sel.columns else (selected_hours[0] if selected_hours else 0)
                extra = get_neighbors_and_911_from_sf_crime_09(selected_geoid, hour_for_lookup)
                if "error" not in extra:
                    c4, c5, c6 = st.columns(3)
                    c4.metric("Komşu suç (1/3/7g)", f"{extra['neighbor_1d']:.1f} / {extra['neighbor_3d']:.1f} / {extra['neighbor_7d']:.1f}")
                    c5.metric("911 (24s/3g/7g)", f"{extra['c911_24h']:.1f} / {extra['c911_3d']:.1f} / {extra['c911_7d']:.1f}")
                    c6.metric("Gerçek tarih", extra.get("date_used", "—"))
                else:
                    st.caption(f"sf_crime_09 ek metrikleri okunamadı: {extra['error']}")


with tab2:
    st.subheader("📈 Zaman serisi (risk_score)")

    if len(view_df) == 0:
        st.info("Seçilen aralık için veri yok.")
    else:
        default_geoids = []
        if len(view_df_city):
            default_geoids.append("0")
        if len(topk):
            default_geoids.extend(topk["geoid"].head(3).tolist())

        options_geoids = []
        if len(view_df_city):
            options_geoids.append("0")
        options_geoids.extend(sorted([g for g in view_df["geoid"].astype(str).unique().tolist() if g != "0"]))

        chosen = st.multiselect(
            "Grafikte gösterilecek GEOID'ler",
            options=options_geoids,
            default=default_geoids,
            format_func=geoid_label,
        )

        if chosen:
            piv = (
                view_df[view_df["geoid"].isin(chosen)]
                .pivot_table(index=time_col, columns="geoid", values="risk_score", aggfunc="mean")
                .sort_index()
            )
            if len(piv):
                st.line_chart(piv, height=360)
            else:
                st.caption("Seçilen GEOID'ler için veri yok.")


with tab3:
    st.subheader("🔝 Top-K GEOID tablo")

    if len(topk):
        st.dataframe(topk, use_container_width=True, height=420)

        def csv_bytes(df: pd.DataFrame) -> bytes:
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            return buf.getvalue().encode("utf-8")

        st.download_button(
            "⬇️ CSV indir (Top-K)",
            data=csv_bytes(topk),
            file_name="risk_topk.csv",
            mime="text/csv",
        )
    else:
        st.caption("Top-K tablosu için yeterli veri yok.")


st.caption(
    "Kaynak: GitHub Actions artifact 'sf-crime-outputs-parquet' → "
    "risk_3h_next7d_top3 / risk_daily_next365d_top5 (parquet/csv). "
    "Fallback: yerel/GitHub raw 'crime_forecast_7days_all_geoids_FRstyle.csv'. "
    "Harita geometri: data/sf_cells.geojson."
)
