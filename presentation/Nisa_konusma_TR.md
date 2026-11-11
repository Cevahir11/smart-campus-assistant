# Konuşma Metni — Nisanur Kara · Syntax Error Takımı (≈ 7 dk)

İpucu: kalın yazılan kelimeleri vurgula, "—" işaretlerinde kısa bir duraksama yap.

## Slayt 1 — Araştırma sorusu
"İyi günler, biz **Syntax Error** takımıyız; projemiz **Akıllı Kampüs Duyuru Asistanı**.

Öğrenciler her hafta sınav tarihlerini, salon değişikliklerini ve son başvuru tarihlerini kaçırıyor, çünkü
duyurular bölüm sayfalarına, e-postalara ve kulüp kanallarına dağılmış durumda. Bizim sistemimiz bu duyuruları
okuyor, kategorilere ayırıyor; tarih, saat, yer ve ders kodunu çıkarıyor ve *'Bu hafta sınav var mı?'* gibi
sorulara — her zaman kaynağını göstererek — cevap veriyor.

Cevaplamak istediğimiz soru şu: **Türkçe için eğitilmiş BERTurk modeli, duyuruları klasik TF-IDF temel modelinden
daha iyi sınıflandırıyor mu — özellikle şablonlarımızın dışında yazılmış metinlerde?**
'Hayır' cevabının ne olduğunu da baştan belirledik: kazanç **0.03 Macro-F1**'den küçükse ya da güven aralığı
sıfırı içeriyorsa cevap hayır.

Öneri raporunda daha geniş bir 'nasıl' sorusu sormuştuk. Onu daralttık, çünkü bir 'nasıl' sorusu hiçbir zaman
yanlışlanamaz."

## Slayt 2 — Veri
"Peki bunu hangi veriyle test ediyoruz?

Altı kategoride **420 etiketli duyurumuz** var: **252'si eğitim, 84'ü geliştirme, 84'ü test** için. Test setini
**dondurduk**. Sağ tarafta gerçek bir örnek görüyorsunuz: bir kimya dersinin iptali; etiketi ve çıkarılan tarih ile
ders kodu yanında.

Açık konuşmak gerekirse bu 420 duyuru **sentetik**, yani kurallarımızla aynı kelimeleri paylaşıyor. Bu yüzden gerçek
üslupta **24 duyuruyu elle yazdık**; buna **Test B** diyoruz.

Ve dünden beri gerçek verimiz de var: kendi yazdığımız toplayıcıyla **GTÜ'nün 24 bölümünden 818 duyuru** topladık.
Henüz etiketlenmedi; sıradaki adımımız bu."

## Slayt 3 — Temel model sonucu
"Şimdi temel model sonuçlarına bakalım.

Sentetik test setinde TF-IDF ile lojistik regresyon kusursuz bir **1.000** alıyor. Bu da bize bu test setinin
**fazla kolay** olduğunu söylüyor.

Asıl ilginç sayı sağda: elle yazdığımız Test B'de aynı model **0.874**'e düşüyor. Yani fark, metin gerçek dünyaya
benzediği anda ortaya çıkıyor.

Bu modeller deterministik olduğu için seed yerine **%95 bootstrap güven aralıkları** veriyoruz. Test B'de sadece
24 örnek olduğu için aralık geniş; bu da daha büyük, gerçek bir test setine ihtiyacımız olduğunu gösteriyor."

## Slayt 4 — Hata analizi
"Bu farkı anlamak için temel modelin **65 hatasını** tek tek inceledik ve yedi kategoriye ayırdık.

En büyük grup, **21 hatayla**, *'konunun eylemi yenmesi'*: bir sınavın salonu değiştiğinde model *'sınav'*
kelimesini görüp **sınav** diyor, oysa doğru etiket **derslik değişikliği**.
İkinci grup, **16 hatayla**, hiçbir anahtar kelimenin tetiklenmediği duyurular."

## Slayt 5 — Hatalar ne söylüyor
"Buradan ana çıkarımımıza geliyoruz: **etiketi konu değil, eylem belirliyor.**

Kelime sayan bir model *'sınav A salonu yerine B salonunda yapılacak'* gibi bir cümleyi anlayamaz. BERTurk gibi
bağlamsal bir model **anlayabilir** — onu test etmemizin sebebi de tam olarak bu.
İkinci sorun **eğitimde görülmemiş kelimeler**; önceden eğitilmiş modeller burada da yardımcı olabilir.

Yani BERTurk'ün farkı **sadece gerçek, şablon dışı metinlerde** göstermesini bekliyoruz; kolay sentetik sette değil."

## Slayt 6 — Yöntemin durumu
"Peki yöntemde ne durumdayız?

BERTurk'ü **uyguladık ve çalıştırdık**. Değiştirdiğimiz tek şey model; veri, bölme ve metrik tamamen aynı.

Geliştirme setindeki ilk sonucumuz **0.807**, üç seed arasında da büyük bir oynama var; yani şimdilik TF-IDF'in
**altında**. Sebebi açık: dizüstü bilgisayarın işlemcisinde sadece **64 adım** eğitildi. Sıradaki adım onu GPU'da
düzgünce eğitmek.

Sistemin geri kalanı zaten çalışıyor: bilgi çıkarımı **0.947 F1**'e ulaşıyor, soru-cevap ise **50 test sorusunun
50'sini** doğru cevaplıyor."

## Slayt 7 — Final planı
"Son olarak finale kadar planımız.

Her adımın bir **sorumlusu ve tarihi** var. **27 Kasım**'a kadar **150 gerçek GTÜ duyurusunu** etiketleyeceğiz;
bu bizim gerçek test setimiz, **Test C** olacak. Ablation olarak BERTurk'ü çok dilli **XLM-R** ile
karşılaştıracağız. **9 Aralık**'ta da tüm sistemleri test setlerinde **sadece bir kez** çalıştıracağız.

**En büyük riskimiz**, gerçek GTÜ metinlerinin etiketlerimizden çok farklı olması: modelin güveni **0.89'dan
0.37'ye** düşüyor. **B planımız** 300 yerine 150 duyuru etiketlemek ve bir **'diğer'** kategorisi eklemek.

Böylece finalde sorumuza **gerçek veriyle** cevap verebileceğiz: BERTurk gerçekten işe yarıyor mu?

Teşekkürler, sorularınızı almaktan memnuniyet duyarız."
