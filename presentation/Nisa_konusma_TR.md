# Konuşma Metni — Nisanur Kara (≈ 6–7 dk)

## Slayt 1 — Araştırma sorusu
"Merhaba, biz Akıllı Kampüs Duyuru Asistanı'nı geliştiriyoruz. Sistem üniversite duyurularını sınıflandırıyor;
tarih, saat, yer ve ders kodunu çıkarıyor ve öğrencinin sorularını kaynak göstererek cevaplıyor.

Sorumuz: **Fine-tune edilmiş BERTurk, duyuruları TF-IDF + lojistik regresyon temel modelinden daha iyi
sınıflandırıyor mu? Bunu şablonlarımızın dışında yazılmış metinlerde ölçüyoruz.**
Kazanç 0.03 Macro-F1'in altındaysa ya da güven aralığı sıfırı içeriyorsa cevap 'hayır'.
Öneri raporundaki 'nasıl' sorusu yanlışlanamadığı için soruyu daralttık."

## Slayt 2 — Veri
"6 kategoride 420 etiketli duyurumuz var: 252 eğitim, 84 geliştirme, 84 test. Test setini dondurduk.
Bu veri sentetik, bu yüzden gerçek üslupta 24 duyuru da yazdık; buna Test B diyoruz.
Ayrıca GTÜ'nün 24 bölümünden **818 gerçek duyuru** topladık; henüz etiketlenmedi."

## Slayt 3 — Temel model sonucu
"TF-IDF + lojistik regresyon test A'da **1.000** alıyor, yani sentetik test doymuş durumda.
Elle yazılmış Test B'de ise **0.874**'e düşüyor. %95 bootstrap güven aralıklarını raporluyoruz; Test B'de sadece 24
örnek olduğu için aralığı geniş."

## Slayt 4 — Hata analizi
"Temel modellerin **65 hatasını** 7 kategoriye ayırdık. En büyüğü: bir sınavın salonu değişince model 'derslik
değişikliği' yerine 'sınav' diyor, 21 hata. İkincisi: hiçbir anahtar kelime tetiklenmiyor, 16 hata."

## Slayt 5 — Hatalar ne söylüyor
"Etiketi belirleyen konu değil, öğrencinin yapması gereken eylem. Kelime sayan modeller 'A salonu yerine B salonu'
yapısını anlayamaz; BERTurk gibi bağlamsal bir model anlayabilir. İkinci sorun eğitimde görülmemiş kelimeler. Bu
yüzden BERTurk'ü şablon dışı veride test ediyoruz."

## Slayt 6 — Yöntemin durumu
"BERTurk'ü uyguladık ve çalıştırdık. Değişen tek şey model; veri ve metrik aynı.
Geliştirme setindeki ilk sonuç **0.807 ± 0.134**: TF-IDF'in altında ve seed'ler arasında dengesiz, çünkü sadece 64
adım eğitildi. Sıradaki adım GPU'da daha uzun eğitmek."

## Slayt 7 — Plan
"Her adımın bir sorumlusu ve tarihi var. 27 Kasım'a kadar 150 GTÜ duyurusunu etiketleyip Test C yapacağız. Ablation
olarak BERTurk'ü XLM-R ile karşılaştıracağız. 9 Aralık'ta tüm sistemleri test setlerinde bir kez çalıştıracağız.
**En büyük risk:** gerçek GTÜ metinleri etiketlerimizden uzak; modelin ortanca güveni 0.89'dan 0.37'ye düşüyor.
**B planı:** 150 duyuru etiketlemek ve 'diğer' kategorisi eklemek.
Teşekkürler, sorularınızı alabiliriz."
