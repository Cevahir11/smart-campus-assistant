# Ara Sunum Konuşma Metni (10 dakika, 7 slayt)

Toplam süre: yaklaşık 9 dakika konuşma + 1 dakika pay. Her slaytın yanında süre ve önerilen konuşmacı var;
isimleri kendi aranızda değiştirebilirsiniz. Hoca herkese soru sorabilir, bu yüzden herkes tüm metni okumalı.
Project Card sayfaları (A1–A4) sunulmaz, sadece notlandırılır.

---

## Slayt 1 — Araştırma sorusu (≈ 1 dk 15 sn) · Cevahir

"Merhaba, biz Akıllı Kampüs Duyuru Asistanı projesini yapıyoruz. Sistem dağınık üniversite duyurularını
sınıflandırıyor, içinden tarih, saat, yer ve ders kodunu çıkarıyor, tek cümlelik özet üretiyor ve öğrencinin
sorusuna kaynak göstererek cevap veriyor.

Araştırma sorumuz şu: **Fine-tune edilmiş BERTurk, duyuruları TF-IDF + lojistik regresyon temel modelinden daha
doğru sınıflandırıyor mu? Bunu özellikle bizim şablonlarımızın dışında yazılmış duyurularda, Macro-F1 ile ölçüyoruz.**

'Hayır' cevabını önceden tanımladık: BERTurk'ün kazancı 0.03 Macro-F1'den küçükse ya da paired bootstrap güven
aralığı sıfırı içeriyorsa cevap hayırdır. O durumda basit lineer model yeterli demektir ve emeğimizi bilgi çıkarımına
ve soru-cevaba kaydırırız.

Öneri raporuna göre soruyu değiştirdik. Eski soru 'NLP teknikleri duyuruları nasıl yapılandırılmış bilgiye
dönüştürebilir?' idi. Bu bir 'nasıl' sorusu olduğu için 'hayır' sonucu yok, yani yanlışlanamıyor. Ayrıca şablon
tabanlı test setimizde TF-IDF zaten 1.000 alıyor. Sistemi değiştirmedik, sadece soruyu hâlâ açık olan tek
karşılaştırmaya daralttık."

---

## Slayt 2 — Veri (≈ 1 dk 15 sn) · Helin

"Verimizi yüklendiği haliyle gösteriyoruz. 420 etiketli duyurumuz var: 6 kategori (sınav, etkinlik, burs, ders
iptali, derslik değişikliği, kayıt), %80 Türkçe, %20 İngilizce. Kategoriye göre tabakalı olarak 60/20/20 böldük:
**252 train, 84 dev, 84 test.** Bölme seed 2026 ile yapıldı ve id listesi repoda sabit; dosyanın hash'ini de slayta
yazdık, yani test setini dondurduk.

İkinci bir test setimiz daha var: **Test B, elimizle gerçek duyuru üslubunda yazdığımız 24 duyuru.** Hiçbir model
bunlarla eğitilmedi.

Sağda gerçek bir örnek var: 'KIM107 dersi iptal'. Etiketi cancellation, içinden tarih ve ders kodu çıkarılıyor,
referans özeti de var.

Burada dürüst olmamız gereken nokta şu: train, dev ve test A **sentetik**. Kendi yazdığımız bir üreticiyle
oluşturuldu, bu yüzden kurallarımızla aynı kelimeleri paylaşıyor. Test B tam da bu yüzden var, ve gerçek
duyurulardan oluşacak Test C planımızın en önemli maddesi. Gerçek veri artık sadece bir söz değil: dün kendi toplayıcımızla **GTÜ'nün 24
bölümünden 818 gerçek duyuru** topladık, henüz etiketlenmedi. Lisans: kendi metinlerimiz CC BY 4.0; GTÜ sayfaları için
sadece URL'leri ve kendi etiketlerimizi paylaşacağız."

---

## Slayt 3 — Temel model sonuçları (≈ 1 dk 30 sn) · Cevahir

"Tabloda tüm temel modeller train üzerinde eğitildi. Çoğunluk sınıfı 0.059, anahtar kelime kuralları test A'da
0.840 Macro-F1 alıyor. **Ana temel modelimiz TF-IDF + lojistik regresyon, test A'da 1.000.** Yani test A doymuş
durumda, modelleri ayırt edemiyor.

Asıl bulgu sağ sütunda: elle yazılmış Test B'de aynı model **0.874**'e düşüyor, 0.126 puanlık bir düşüş. Şablon
dışı metin, yani gerçek dünyaya en yakın veri, farkın olduğu yer.

Dağılımı nasıl verdik: bu modeller deterministik olduğu için seed değiştirmek bir şey değiştirmiyor. Onun yerine
test setinden 1000 bootstrap örneklemiyle %95 güven aralığı hesapladık. Test B'nin aralığı 0.68 ile 1.00 arası,
yani yaklaşık 0.3 genişliğinde. 24 örnekle 0.03'lük bir farkı göremeyiz; bu da bize daha büyük gerçek bir test
setine ihtiyacımız olduğunu söylüyor. Tablonun tamamı tek komutla üretiliyor: `python evaluation/midproject.py`."

---

## Slayt 4 — Hata analizi: kategoriler (≈ 1 dk 15 sn) · Nisanur

"Temel modellerin **65 hatasını** tek tek okuyup 7 kategoriye ayırdık. Kural modelleri eğitilmediği için train
üzerindeki hatalarını da analiz etmek adil; test A'ya ise dokunmadık, dondurulmuş durumda.

En büyük kategori **E1, 21 hata: 'konu aksiyonu yeniyor'.** Bir sınavın salonu değiştiğinde duyurunun etiketi
derslik değişikliği, ama metinde 'sınav' kelimesi geçtiği için model sınav diyor.
İkincisi **E4, 16 hata:** hiçbir anahtar kelime tetiklenmiyor, örneğin 'Don't miss', 'no lecture', 'ekle-sil'.
**E2, 13 hata:** 'başvuru' kelimesi hem burs hem kayıt duyurularında geçiyor.
E3'te sınavla ilgili kayıt duyuruları, E5'te sınavlardan bahseden etkinlikler, E6'da 'Seminer Salonu' gibi bir
salon adının etkinlik sanılması var. **E7, 3 hata:** TF-IDF'in Test B'de yaptığı, eğitimde hiç görmediği
kelimelerden kaynaklanan hatalar."

---

## Slayt 5 — Hatalar bize ne söylüyor (≈ 1 dk 15 sn) · Nisanur

"Bu hatalardan dört sonuç çıkarıyoruz.

Bir: **etiket konu değil, aksiyondur.** 'MAT 102 sınavı Konferans Salonu yerine E-205'te' cümlesinde her ipucu
sınav diyor, ama öğrencinin yapması gereken şey yeni salona gitmek. Kelime torbası modeller 'X yerine Y' yapısını
tartamaz, bağlamsal bir encoder tartabilir. BERTurk için en güçlü argüman bu.

İki: **kelime boşlukları.** 'guest lecture', 'tuition waiver', 'ilişik kesme' gibi ifadeler eğitimde hiç yok.
Önceden eğitilmiş bir model bu bilgiyi taşıyor olabilir.

Üç: 'başvuru' gibi belirsiz ipuçlarını TF-IDF veriden zaten öğreniyor, test A'da sıfır hata yapıyor. Yani bu bir
öğrenme problemi değil, kural problemi.

Dört: bilgi çıkarımında Test B'de 7 hata var: yılı yazılmamış tarih aralıkları, serbest yazılmış yer adları ve
'17.30'da' gibi noktalı saatler. Buna rağmen micro-F1 0.947.

Sonuç olarak BERTurk'ün sadece ilk iki kategoride fayda sağlamasını bekliyoruz. Bunlar da çoğunlukla şablon dışında
ortaya çıkıyor, o yüzden Test B ve gerçek Test C üzerinde ölçeceğiz, test A üzerinde değil."

---

## Slayt 6 — Yöntemin durumu (≈ 1 dk 15 sn) · Emre

"Yöntemimiz implement edildi ve çalıştırıldı. **Tek kontrollü değişiklik:** TF-IDF özellikleri + lojistik
regresyon yerine fine-tune edilmiş BERTurk encoder'ı koyuyoruz. Bölme, girdi (başlık + metin), etiketler ve metrik
tamamen aynı.

Ayarlar: 4 epoch, öğrenme oranı 3e-5, batch 16, 3 seed. Dizüstü bilgisayarda CPU üzerinde seed başına yaklaşık
6 dakika sürdü.

**Dev setinde ilk sonucumuz: BERTurk üç seed ortalamasında 0.807 ± 0.134 Macro-F1 alıyor, yani TF-IDF'in
1.000'ının altında.** Seed'ler 0.62 ile 0.92 arasında değişiyor, model çok dengesiz. Eğitim kaybı sadece 64 adımdan
sonra hâlâ 1.1 ile 1.6 arasında. Yani bu modelin zayıflığı değil, yetersiz eğitim. Üç seed raporlamamızın sebebi de
tam olarak bu. Sıradaki adım: GPU'da 10–20 epoch eğitip epoch sayısını dev setinde seçmek. Test setlerine
9 Aralık'a kadar dokunmuyoruz.

Sistemin geri kalanı da çalışıyor: bilgi çıkarımı Test B'de 0.947 micro-F1, şablon özetleyici ROUGE-L 0.953
(lead-1 baseline 0.432), soru-cevap ise tahmin edilen etiketlerle uçtan uca 50 sorunun 50'sini doğru cevaplıyor.
Flask web arayüzü ve komut satırı hazır."

---

## Slayt 7 — Final planı (≈ 1 dk 15 sn) · Helin (tablo) + Cevahir (protokol/risk)

"Kalan her adımın bir sorumlusu ve tarihi var. Veri toplama tamamlandı: GTÜ'den 818 gerçek duyuru. 27 Kasım'a kadar bunların 150'sini
'diğer' kategorisini de ekleyerek iki kişi bağımsız etiketleyip Cohen's kappa hesaplıyoruz ve Test C
olarak donduruyoruz. BERTurk'ü GPU'da 3 seed ile çalıştırıyoruz. Ablation olarak 4 Aralık'a kadar BERTurk'ü çok
dilli XLM-R ile ve başlık alanı çıkarılmış haliyle karşılaştırıyoruz. 9 Aralık'ta tüm sistemleri test setlerinde
**sadece bir kez** çalıştırıp paired bootstrap yapıyoruz. 18 Aralık'ta final raporu makale formatında hazır.

Değerlendirme protokolünü bugün, yöntem sonuçlarını görmeden sabitledik: bir kazancı ancak 0.03 Macro-F1'den
büyükse ve güven aralığı sıfırı içermiyorsa gerçek sayacağız.

**En büyük risk:** gerçek GTÜ metinleri etiketlerimizden çok uzak. Modelin ortanca güveni test A'da 0.89, GTÜ
duyurularında ise sadece **0.37**; birçok duyuru 6 kategorimizin hiçbirine uymuyor. Bu yüzden etiketleme yavaş
olabilir ya da uyum düşük çıkabilir. **B planı:** 300 yerine 150 duyuru etiketlemek, 'diğer' kategorisi eklemek ve
tespit edilebilen en küçük farkı bir kısıt olarak raporlamak.

Teşekkürler, sorularınızı alabiliriz."

---

## Olası sorular ve kısa cevaplar

- **"Veriniz sentetikse sonuçlar ne kadar anlamlı?"** → Test A'nın doymasının sebebi bu, o yüzden asıl kararı Test B
  ve gerçek Test C üzerinde veriyoruz. Test C 27 Kasım'da, hiçbir model görmeden donduruluyor.
- **"Neden seed yerine bootstrap?"** → TF-IDF modelleri deterministik, seed değişince sonuç değişmiyor. Seed std'sini
  BERTurk ve XLM-R için 3 seed ile raporluyoruz.
- **"0.03 eşiği nereden geliyor?"** → Pratikte anlamlı en küçük fark olarak önceden seçtik; ayrıca paired bootstrap
  CI'nın sıfırı dışlaması gerekiyor (Dror vd., 2018).
- **"Hata analizini neden test A'da yapmadınız?"** → Test seti dondurulmuş; üzerinde hata incelemek onu fiilen
  geliştirme setine çevirir.
- **"Ablation ne?"** → BERTurk vs XLM-R (aynı tarif), başlıksız BERTurk ve %25/50/100 öğrenme eğrisi.
- **"Kim ne yaptı?"** → Helin veri ve kılavuz, Nisanur research lead (ilgili çalışmalar, ablation tasarımı), bilgi çıkarımı ve hata analizi, Cevahir sınıflandırıcılar ve
  değerlendirme, Emre özetleme, soru-cevap ve web demo.
- **"BERTurk neden TF-IDF'ten kötü?"** → Sadece 252 eğitim örneği ve 64 adım; eğitim kaybı yakınsamadı (1.1–1.6) ve
  bir seed 0.62'de kaldı. Bu yetersiz eğitim. Test çalıştırmasından önce epoch ve öğrenme oranını dev'de ayarlayacağız.
  Yine de Test B/C'de TF-IDF'i 0.03 farkla geçemezse araştırma sorumuzun cevabı "hayır" olacak.
