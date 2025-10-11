crime_prediction
AI-powered crime prediction app using San Francisco crime and environmental data.

Açılışta Top-10: app_ops.py otomatik olarak seçilen saat aralığı için rank, GEOID, priority_score, p_crime, [LCB–UCB], top3_crime_types verir.
Saat filtresi: Tek tıkla yeni saat penceresi için Top-K üretilir.
GEOID sayfası: Belirsizlik bandı, kısa risk anlatısı, (varsa) crime-mix.
Preskriptif çıktı: plan_routes ile ekip bazlı öneri.
Model entegrasyonu: models/ altında gerçek stacking_model.joblib ve kalibrasyon dosyaları konunca DummyPredictor devre dışı kalır.
Veri yoksa çalışırlık: Hiçbir giriş dosyası olmasa bile UI örnek evren üretir, hata fırlatmaz.
