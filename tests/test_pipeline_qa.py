from datetime import date

import pytest

from campus_assistant.classification import KeywordClassifier, TfidfClassifier, doc_text
from campus_assistant.metrics import rouge
from campus_assistant.pipeline import AnnouncementProcessor, read_jsonl
from campus_assistant.qa import QAEngine
from campus_assistant.service import ROOT

RECORDS = read_jsonl(ROOT / "data" / "announcements.jsonl")
TODAY = date(2026, 11, 9)


@pytest.fixture(scope="module")
def processor():
    return AnnouncementProcessor(train_records=RECORDS)


@pytest.fixture(scope="module")
def engine(processor):
    return QAEngine([processor.process(r) for r in RECORDS], today=TODAY)


def test_classifiers_learn():
    clf = TfidfClassifier("logreg").fit([doc_text(r) for r in RECORDS], [r["category"] for r in RECORDS])
    assert clf.predict(["CENG 301 final sınavı 20 Kasım'da B-204'te yapılacaktır."]) == ["exam"]
    assert KeywordClassifier().fit([], ["exam"]).predict(["Yarınki ders iptal edilmiştir."]) == ["cancellation"]


def test_pipeline_room_change(processor):
    r = processor.process({"id": "t", "title": "Derslik değişikliği",
                           "text": "CENG 351 dersi 12 Kasım 2026 tarihinden itibaren B-204 yerine Amfi 3 dersliğinde yapılacaktır.",
                           "published": "2026-11-09"})
    assert r["category"] == "room_change"
    assert r["fields"]["location"] == "Amfi 3" and r["fields"]["old_location"] == "B-204"
    assert r["fields"]["course"] == "CENG 351" and r["fields"]["date"] == "2026-11-12"


def test_qa_lists_and_no_answer(engine):
    res = engine.answer("Bu hafta sınav var mı?")
    assert res["found"] and res["parsed"]["ask"] == "list"
    assert not engine.answer("XYZ 999 sınavı ne zaman?")["found"]


def test_qa_new_announcement_is_searchable(processor, engine):
    rec = processor.process({"id": "new-1", "title": "HUK 205 final sınavı",
                             "text": "HUK 205 final sınavı 12 Kasım 2026 saat 13:00'te Amfi 1'de yapılacaktır.",
                             "published": "2026-11-09"})
    engine.add(rec)
    res = engine.answer("HUK 205 final sınavı ne zaman?")
    assert "12 Kasım" in res["answer"] and "13:00" in res["answer"]


def test_rouge():
    assert rouge("a b c", "a b c")["rougeL"] == 1.0
    assert rouge("a b", "c d")["rouge1"] == 0.0
