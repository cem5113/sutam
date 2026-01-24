# src/ops_rules.py
import numpy as np
import pandas as pd

def reasons_actions_fallback(row: pd.Series, profile_row: pd.Series | None = None):
    reasons = []

    # 911 sinyali
    for c in ["911_geo_last7d","911_request_count_hour_range","911_geo_hr_last7d"]:
        if c in row.index and pd.notna(row[c]):
            v = float(row[c])
            if v >= 20:
                reasons.append("911 çağrı yoğunluğu yüksek (son günlerde artış).")
                break
            if v >= 10:
                reasons.append("911 çağrıları ortalamanın üzerinde.")
                break

    # komşu suç sinyali
    for c in ["neighbor_crime_7d","neighbor_crime_3d"]:
        if c in row.index and pd.notna(row[c]):
            v = float(row[c])
            if v >= 200:
                reasons.append("Komşu bölgelerde son günlerde suç yoğunluğu çok yüksek.")
                break
            if v >= 100:
                reasons.append("Komşu bölgelerde son günlerde suç yoğunluğu yüksek.")
                break

    # zaman bağlamı
    if "is_night" in row.index and int(row.get("is_night", 0) or 0) == 1:
        reasons.append("Gece saatleri (risk artışı).")
    if "is_weekend" in row.index and int(row.get("is_weekend", 0) or 0) == 1:
        reasons.append("Hafta sonu etkisi.")
    if "is_business_hour" in row.index and int(row.get("is_business_hour", 1) or 1) == 0:
        reasons.append("Mesai dışı zaman dilimi.")

    # profile bağlamı
    if profile_row is not None and len(profile_row) > 0:
        vs = str(profile_row.get("vs_sf_avg_daily", "NA"))
        if vs in ("ABOVE_AVG","VERY_ABOVE_AVG"):
            reasons.append("Bölge 5 yıllık ortalamada şehir geneline göre daha yoğun.")

        if pd.notna(profile_row.get("bus_stop_count", np.nan)) and float(profile_row["bus_stop_count"]) >= 20:
            reasons.append("Toplu taşıma düğümü (otobüs durağı yoğunluğu yüksek).")
        if pd.notna(profile_row.get("distance_to_train", np.nan)) and float(profile_row["distance_to_train"]) <= 500:
            reasons.append("Tren/istasyon yakınlığı (yoğunluk çekimi).")

    reasons = reasons[:4] if len(reasons) > 4 else reasons
    if not reasons:
        reasons = ["Genel risk skoruna göre öncelikli hücre (çoklu faktör etkisi)."]

    level = str(row.get("risk_level","NA"))
    hr = str(row.get("hour_range",""))

    if level == "VERY_HIGH":
        action = f"{hr} bandında görünür devriye + hotspot tarama (15–30 dk)."
    elif level == "HIGH":
        action = f"{hr} bandında döngüsel devriye (en az 2 tur) + transit/kalabalık noktaları kontrol."
    elif level == "MED":
        action = f"{hr} bandında rutin rota içine al + kısa süreli kontrol."
    else:
        action = f"{hr} bandında rutin izleme."

    top_cat = row.get("top1_category", None)
    if (top_cat is None or (isinstance(top_cat, float) and pd.isna(top_cat))) and profile_row is not None:
        tc = str(profile_row.get("top5_categories",""))
        if tc:
            top_cat = tc.split(",")[0].split(":")[0]
    if top_cat and isinstance(top_cat, str) and len(top_cat) > 0:
        action += f" (Öncelikli tür: {top_cat})"

    return reasons, action
