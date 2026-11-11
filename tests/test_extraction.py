from datetime import date

from campus_assistant.extraction import RuleExtractor

REF = date(2026, 11, 9)
ex = RuleExtractor()


def norms(text, label):
    return [e.norm for e in ex.extract(text, REF) if e.label == label]


def test_turkish_dates():
    assert norms("Sınav 14 Kasım 2026 Cuma günü", "DATE") == ["2026-11-14"]
    assert norms("Tarih: 14.11.2026", "DATE") == ["2026-11-14"]
    assert norms("Son gün 3 Ocak", "DATE") == ["2027-01-03"]  # year inferred near reference date
    assert norms("17-21 Kasım tarihleri arasında", "DATE") == ["2026-11-17", "2026-11-21"]


def test_english_dates():
    assert norms("on November 14, 2026 at 10:00", "DATE") == ["2026-11-14"]
    assert norms("on Dec 3 at 4 pm", "DATE") == ["2026-12-03"]


def test_times():
    assert norms("saat 14.30 itibarıyla", "TIME") == ["14:30"]
    assert norms("10:00 - 12:00 arası", "TIME") == ["10:00", "12:00"]
    assert norms("at 4 pm", "TIME") == ["16:00"]
    assert norms("saat 10'da başlayacak", "TIME") == ["10:00"]
    assert norms("14.11.2026", "TIME") == []  # a date is not a time


def test_courses_and_locations():
    assert norms("CENG301 ve MAT 101 dersleri", "COURSE") == ["CENG 301", "MAT 101"]
    assert norms("1500 TL burs", "COURSE") == []
    assert norms("B-204 yerine Amfi 3 salonunda", "LOCATION") == ["B-204", "Amfi 3"]
    assert norms("Konferans Salonu'nda", "LOCATION") == ["Konferans Salonu"]
