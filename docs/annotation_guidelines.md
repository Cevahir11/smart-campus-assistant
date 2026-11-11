# Etiketleme Kılavuzu / Annotation Guidelines

Sunumdaki plan: **300–500 gerçek duyurunun elle etiketlenmesi**. Bu projedeki `data/announcements.jsonl`
dosyası aynı şemaya sahip sentetik bir başlangıç verisidir; gerçek veriyle değiştirildiğinde tüm eğitim
ve değerlendirme betikleri değişiklik yapmadan çalışır.

## Kayıt şeması (JSONL, her satır bir duyuru)

```json
{
  "id": "ann-0001",
  "source": "university_website | department_website | email | student_club",
  "lang": "tr | en",
  "published": "2026-11-09",
  "title": "CENG 301 Sınav Duyurusu",
  "text": "CENG 301 ara sınavı 14 Kasım 2026 saat 10:00'da B-204 salonunda yapılacaktır.",
  "category": "exam",
  "entities": [
    {"start": 0, "end": 8, "label": "COURSE", "text": "CENG 301", "norm": "CENG 301"},
    {"start": 21, "end": 34, "label": "DATE", "text": "14 Kasım 2026", "norm": "2026-11-14"},
    {"start": 40, "end": 45, "label": "TIME", "text": "10:00", "norm": "10:00"},
    {"start": 50, "end": 55, "label": "LOCATION", "text": "B-204", "norm": "B-204"}
  ],
  "summary": "CENG 301 ara sınavı 14 Kasım'da saat 10:00'da B-204'te yapılacak."
}
```

`start/end` karakter indeksleri `text` alanına göredir (NER eğitimi için gereklidir; değerlendirme yalnızca
`label + norm` çiftlerini karşılaştırır). `fields` alanı isteğe bağlıdır.

## Kategoriler

| Etiket | Açıklama | Örnek ipuçları |
|---|---|---|
| `exam` | Sınav tarihi/saati/yeri duyurusu (vize, final, quiz, bütünleme, mazeret) | sınav, vize, final |
| `event` | Kulüp/üniversite etkinliği, seminer, konser, atölye | etkinlik, seminer, davet |
| `scholarship` | Burs, değişim programı, destek başvurusu | burs, Erasmus, TÜBİTAK |
| `cancellation` | Ders iptali / telafi | iptal, yapılmayacak |
| `room_change` | Ders veya sınav yerinin değişmesi | yerine, taşınmıştır |
| `registration` | Kayıt, ekle-bırak, harç, akademik takvim | kayıt, harç, ekle-bırak |

Bir duyuru birden çok konuya değiniyorsa **öğrencinin yapması gereken asıl eylemi** belirleyen kategori seçilir
(ör. "sınav salonu değişti" → `room_change`; "mazeret sınavı başvuruları" → `registration`).

## Varlık türleri ve normalizasyon

| Etiket | Kapsam | Normal form |
|---|---|---|
| `DATE` | Takvim tarihi (hafta günü dahil edilmez). Aralıklarda her tarih ayrı etiketlenir. | `YYYY-MM-DD` (yıl yoksa yayın tarihine en yakın yıl) |
| `TIME` | Saat ifadesi | `HH:MM` (24 saat) |
| `LOCATION` | Derslik, amfi, salon, laboratuvar | Yazıldığı gibi; "Amfi 3", "Lab 2" biçimine getirilir, Türkçe ekler ('da, 'nda) hariç |
| `COURSE` | Ders kodu | `CENG 301` (harf + boşluk + numara) |

## Özet
Tek cümle; tüm tarih, saat, yer ve ders kodları korunur; 25 kelimeyi geçmez.

## Kalite kontrol
- Her duyuru en az iki kişi tarafından etiketlenip anlaşmazlıklar tartışılarak çözülür; Cohen's kappa raporlanır.
- Önerilen bölme: %80 eğitim / %20 test (`evaluation/evaluate.py` içindeki `stratified_split`).
