"""Generate the seed dataset of annotated campus announcements + QA evaluation set.

The proposal plans 300-500 *manually* annotated real announcements. Until that
annotation is done, this script produces a realistic synthetic stand-in with the
same schema (category label, entity spans, normalized fields, reference summary),
so every model and evaluation script in the project runs end to end. Replace
`data/announcements.jsonl` with real annotated data (same schema, see
docs/annotation_guidelines.md) and re-run training/evaluation.

Usage:
    python scripts/generate_dataset.py                    # 420 announcements, base date 2026-11-09
    python scripts/generate_dataset.py --n 500 --base-date 2026-12-01
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from campus_assistant.textutils import (EN_MONTHS, EN_WEEKDAYS, TR_MONTHS,  # noqa: E402
                                        TR_WEEKDAYS, format_date_short)

# --------------------------------------------------------------------------- lexicons
COURSES = [
    ("CENG", 301, "Veritabanı Sistemleri", "Database Systems"),
    ("CENG", 351, "Bilgisayar Ağları", "Computer Networks"),
    ("CENG", 213, "Veri Yapıları", "Data Structures"),
    ("CENG", 442, "Doğal Dil İşleme", "Natural Language Processing"),
    ("CENG", 371, "Bilimsel Hesaplama", "Scientific Computing"),
    ("CENG", 315, "Algoritmalar", "Algorithms"),
    ("CENG", 334, "İşletim Sistemleri", "Operating Systems"),
    ("CENG", 483, "Bilgisayarlı Görü", "Computer Vision"),
    ("CENG", 499, "Makine Öğrenmesi", "Machine Learning"),
    ("CENG", 140, "Programlamaya Giriş", "Introduction to Programming"),
    ("MAT", 101, "Matematik I", "Calculus I"),
    ("MAT", 102, "Matematik II", "Calculus II"),
    ("MAT", 241, "Lineer Cebir", "Linear Algebra"),
    ("MAT", 271, "Olasılık ve İstatistik", "Probability and Statistics"),
    ("FIZ", 101, "Fizik I", "Physics I"),
    ("FIZ", 102, "Fizik II", "Physics II"),
    ("EEE", 211, "Devre Teorisi", "Circuit Theory"),
    ("EEE", 303, "Sinyaller ve Sistemler", "Signals and Systems"),
    ("EEE", 342, "Elektronik II", "Electronics II"),
    ("END", 305, "Yöneylem Araştırması", "Operations Research"),
    ("END", 211, "Mühendislik Ekonomisi", "Engineering Economics"),
    ("ISL", 201, "İşletme Yönetimi", "Business Management"),
    ("ISL", 312, "Pazarlama İlkeleri", "Principles of Marketing"),
    ("IKT", 101, "İktisada Giriş", "Introduction to Economics"),
    ("ING", 102, "Akademik İngilizce", "Academic English"),
    ("TDL", 101, "Türk Dili I", "Turkish Language I"),
    ("ATA", 102, "Atatürk İlkeleri ve İnkılap Tarihi II", "Principles of Atatürk II"),
    ("KIM", 107, "Genel Kimya", "General Chemistry"),
    ("BIO", 110, "Genel Biyoloji", "General Biology"),
    ("MAK", 221, "Statik", "Statics"),
    ("MAK", 312, "Akışkanlar Mekaniği", "Fluid Mechanics"),
    ("INS", 201, "Mukavemet", "Strength of Materials"),
    ("PSI", 101, "Psikolojiye Giriş", "Introduction to Psychology"),
    ("HUK", 205, "Bilişim Hukuku", "IT Law"),
    ("CENG", 444, "Derin Öğrenme", "Deep Learning"),
    ("MAT", 345, "Sayısal Analiz", "Numerical Analysis"),
]
LOCATIONS = ["B-204", "A-101", "A-215", "C-012", "D-110", "Z-12", "B-310", "E-205",
             "Amfi 1", "Amfi 3", "Amfi 5", "D Blok 105", "C Blok 204", "Lab 2", "Lab 4",
             "Konferans Salonu", "Kütüphane Seminer Salonu", "Kültür Merkezi", "Kongre Merkezi"]
EVENT_LOCATIONS = ["Konferans Salonu", "Kültür Merkezi", "Kongre Merkezi", "Kütüphane Seminer Salonu",
                   "Amfi 1", "Amfi 5", "A-101", "Spor Salonu"]
EXAM_TYPES_TR = ["ara sınavı", "vize sınavı", "final sınavı", "quiz", "bütünleme sınavı", "mazeret sınavı"]
EXAM_TYPES_EN = ["midterm exam", "final exam", "quiz", "make-up exam"]
INSTRUCTORS = ["Dr. Öğr. Üyesi Ayşe Yılmaz", "Prof. Dr. Mehmet Demir", "Doç. Dr. Elif Kaya",
               "Dr. Öğr. Üyesi Can Öztürk", "Prof. Dr. Zeynep Arslan", "Doç. Dr. Burak Şahin",
               "Arş. Gör. Deniz Koç"]
CLUBS = ["IEEE Öğrenci Kolu", "Yapay Zeka Kulübü", "Girişimcilik Kulübü", "Müzik Kulübü",
         "Tiyatro Topluluğu", "Dağcılık Kulübü", "Bilgisayar Mühendisliği Topluluğu", "Kariyer Merkezi",
         "Satranç Kulübü", "Fotoğrafçılık Kulübü"]
EVENTS = [  # (tr name, en name)
    ("Yapay Zeka Zirvesi", "AI Summit"), ("Kariyer Günleri", "Career Days"),
    ("Bahar Konseri", "Spring Concert"), ("Hackathon 2026", "Hackathon 2026"),
    ("Girişimcilik Paneli", "Entrepreneurship Panel"), ("Python Atölyesi", "Python Workshop"),
    ("Siber Güvenlik Semineri", "Cyber Security Seminar"), ("Satranç Turnuvası", "Chess Tournament"),
    ("Fotoğraf Sergisi", "Photography Exhibition"), ("Tiyatro Gösterisi", "Theatre Night"),
    ("Mezunlarla Buluşma", "Alumni Meetup"), ("Stres Yönetimi Semineri", "Stress Management Seminar"),
    ("Veri Bilimi Söyleşisi", "Data Science Talk"), ("Robotik Yarışması", "Robotics Contest"),
    ("Kodlama Kampı", "Coding Bootcamp"), ("Bilim Şenliği", "Science Fair"),
    ("Doğa Yürüyüşü", "Nature Hike"), ("Film Gösterimi", "Film Screening"),
    ("Startup Pitch Günü", "Startup Pitch Day"), ("Kan Bağışı Kampanyası", "Blood Donation Drive"),
    ("Açık Kaynak Günü", "Open Source Day"), ("Oyun Geliştirme Atölyesi", "Game Dev Workshop"),
]
SCHOLARSHIPS = [
    "TÜBİTAK 2209-A", "TÜBİTAK 2210 Yüksek Lisans Bursu", "Vakıf Başarı Bursu", "Erasmus+ Öğrenim Hareketliliği",
    "Erasmus+ Staj Hareketliliği", "Yemek Bursu", "KYK Burs/Kredi", "Mezunlar Derneği Bursu",
    "Türk Eğitim Vakfı Bursu", "Kısmi Zamanlı Öğrenci Çalışma Programı", "Mevlana Değişim Programı",
    "Farabi Değişim Programı", "Rektörlük Üstün Başarı Bursu", "Teknopark Proje Desteği",
    "Uluslararası Konferans Katılım Desteği", "Sanayi Destekli Bitirme Projesi Bursu",
]
REG_TOPICS = [  # (tr, en)
    ("ders ekleme-bırakma", "add-drop"), ("yaz okulu kayıtları", "summer school registration"),
    ("çift anadal başvuruları", "double major applications"), ("yandal başvuruları", "minor applications"),
    ("harç ödemeleri", "tuition payments"), ("mezuniyet başvuruları", "graduation applications"),
    ("mazeret sınavı başvuruları", "make-up exam applications"), ("kayıt yenileme", "course registration"),
    ("danışman onayları", "advisor approvals"), ("yatay geçiş başvuruları", "transfer applications"),
    ("muafiyet başvuruları", "exemption applications"), ("İngilizce hazırlık muafiyet sınavı başvuruları",
                                                         "English proficiency exam applications"),
]
SOURCES = ["university_website", "department_website", "email", "student_club"]
FILLERS_TR = [
    "Tüm öğrencilerimize başarılar dileriz.",
    "Detaylı bilgi için bölüm sekreterliği ile iletişime geçebilirsiniz.",
    "Öğrenci kimlik kartınızı yanınızda bulundurmayı unutmayınız.",
    "Duyuru ilgili tüm öğrencilere önemle duyurulur.",
    "Sorularınız için öğrenci işleri daire başkanlığına başvurabilirsiniz.",
    "Lütfen e-posta kutunuzu düzenli olarak kontrol ediniz.",
    "Güncellemeler bölüm web sayfasından takip edilebilir.",
]
FILLERS_EN = [
    "Please contact the department office for further information.",
    "Do not forget to bring your student ID card.",
    "Updates will be posted on the department website.",
    "We wish all students success.",
]


# --------------------------------------------------------------------------- surface helpers
def date_surface(d: date, lang: str, rng: random.Random) -> str:
    if lang == "en":
        v = rng.choice(["mdy", "dmy", "wmd"])
        if v == "mdy":
            return f"{EN_MONTHS[d.month - 1]} {d.day}, {d.year}"
        if v == "dmy":
            return f"{d.day} {EN_MONTHS[d.month - 1]} {d.year}"
        return f"{EN_WEEKDAYS[d.weekday()]}, {EN_MONTHS[d.month - 1]} {d.day}"
    v = rng.choices(["long", "long_wd", "dots", "short", "slash"], [4, 3, 3, 2, 1])[0]
    if v == "long":
        return f"{d.day} {TR_MONTHS[d.month - 1]} {d.year}"
    if v == "long_wd":
        return f"{d.day} {TR_MONTHS[d.month - 1]} {d.year} {TR_WEEKDAYS[d.weekday()]}"
    if v == "dots":
        return f"{d.day:02d}.{d.month:02d}.{d.year}"
    if v == "short":
        return f"{d.day} {TR_MONTHS[d.month - 1]}"
    return f"{d.day:02d}/{d.month:02d}/{d.year}"


def course_surface(c, rng):
    return f"{c[0]} {c[1]}" if rng.random() < 0.8 else f"{c[0]}{c[1]}"


def rand_time(rng, start=8, end=17):
    return f"{rng.randint(start, end):02d}:{rng.choice(['00', '00', '30', '15', '45'])}"


class Rendered:
    """Fill a template like 'Sınav {date} saat {time}' and keep entity spans."""

    PH = re.compile(r"\{(\w+)\}")

    def __init__(self, template: str, values: dict, entity_slots: dict):
        out, ents, pos = [], [], 0
        for m in self.PH.finditer(template):
            out.append(template[pos:m.start()])
            key = m.group(1)
            surface = str(values[key])
            start = sum(len(x) for x in out)
            out.append(surface)
            if key in entity_slots:
                label, norm = entity_slots[key]
                ents.append({"start": start, "end": start + len(surface), "label": label,
                             "text": surface, "norm": norm})
            pos = m.end()
        out.append(template[pos:])
        self.text = "".join(out)
        self.entities = ents


# --------------------------------------------------------------------------- templates
T = {
    "exam": {
        "tr": [
            ("{course} {cname} {etype} Hakkında",
             "{course} {cname} dersinin {etype} {date} tarihinde saat {time} itibarıyla {loc} salonunda yapılacaktır. Sınav süresi 90 dakikadır."),
            ("{course} Sınav Duyurusu",
             "Sevgili öğrenciler, {course} {etype} {date} günü yapılacaktır. Sınav saati: {time}. Sınav yeri: {loc}. Sınava gelirken kimlik kartınızı getirmeyi unutmayınız."),
            ("{cname} {etype} tarihi",
             "{instructor} tarafından verilen {course} dersinin {etype} için tarih belirlenmiştir: {date}, saat {time}. Sınav {loc} dersliğinde gerçekleştirilecektir."),
            ("{course} {etype} yeri ve saati",
             "{course} kodlu {cname} dersi {etype} {loc} salonunda {date} tarihinde saat {time} - {time2} arasında yapılacaktır. Hesap makinesi kullanılabilir."),
            ("Sınav programı: {course}",
             "Bölümümüz {course} {cname} dersinin {etype} {date} tarihinde yapılacaktır. Öğrencilerin saat {time} itibarıyla {loc} önünde hazır bulunmaları gerekmektedir."),
            ("{course} {etype}",
             "Duyuru: {course} ({cname}) {etype}; Tarih: {date} | Saat: {time} | Salon: {loc}. Sınav kapsamı ilk altı haftanın konularıdır."),
        ],
        "en": [
            ("{course} {etype} announcement",
             "The {etype} of {course} {cname} will be held on {date} at {time} in {loc}. The exam lasts 90 minutes."),
            ("{cname} exam schedule",
             "Dear students, the {course} {etype} is scheduled for {date}, {time}, room {loc}. Please bring your student ID."),
        ],
    },
    "event": {
        "tr": [
            ("{event} etkinliğine davetlisiniz",
             "{club} olarak düzenlediğimiz {event} {date} tarihinde saat {time} itibarıyla {loc} salonunda gerçekleşecektir. Katılım ücretsizdir."),
            ("{event}",
             "{club} sunar: {event}! {date} günü saat {time} itibarıyla {loc} salonunda buluşuyoruz. Kayıt için kulüp sayfamızı ziyaret edin."),
            ("Söyleşi: {event}",
             "{speaker} konuğumuz olarak {event} kapsamında öğrencilerle buluşacaktır. Etkinlik {date} tarihinde, saat {time} itibarıyla {loc} salonunda yapılacaktır."),
            ("{club} - {event}",
             "Sınav haftası öncesi moral ve motivasyon için {event} düzenliyoruz. Tarih: {date}, Saat: {time}, Yer: {loc}. Tüm öğrenciler davetlidir."),
            ("{event} başvuruları başladı",
             "{club} tarafından düzenlenen {event} {date} tarihinde yapılacaktır. Etkinlik saat {time} itibarıyla {loc} salonunda başlayacaktır. Kontenjan sınırlıdır."),
            ("Etkinlik duyurusu: {event}",
             "Kampüsümüzde {date} günü {event} düzenlenecektir. Saat {time} - {time2} arasında {loc} salonunda gerçekleşecek etkinliğe tüm öğrenciler davetlidir."),
        ],
        "en": [
            ("Join us: {event_en}",
             "{club} invites you to {event_en} on {date} at {time} in {loc}. Participation is free."),
            ("{event_en}",
             "Don't miss {event_en}! It will take place on {date}, {time}, at {loc}. Register on the club page."),
        ],
    },
    "scholarship": {
        "tr": [
            ("{sch} başvuruları başladı",
             "{sch} için başvurular başlamıştır. Son başvuru tarihi {date} olup başvurular çevrimiçi sistem üzerinden yapılacaktır. Aylık burs miktarı {amount} TL'dir."),
            ("{sch} duyurusu",
             "{sch} kapsamında başvuru yapmak isteyen öğrencilerimizin gerekli belgeleri en geç {date} tarihine kadar öğrenci işlerine teslim etmeleri gerekmektedir."),
            ("{sch} - son gün yaklaşıyor",
             "Hatırlatma: {sch} başvuruları {date} tarihinde sona erecektir. Başvuru sonuçları {date2} tarihinde ilan edilecektir."),
            ("{sch} bilgilendirme toplantısı",
             "{sch} hakkında bilgilendirme toplantısı {loc} salonunda yapılacaktır. Toplantı saat {time} itibarıyla başlayacaktır. Son başvuru tarihi {date} olarak belirlenmiştir."),
            ("{sch}",
             "Not ortalaması 3.00 ve üzeri olan öğrenciler {sch} için başvurabilir. Başvurular {date} tarihine kadar kabul edilecektir. Sınav döneminde başvuru alınmayacaktır."),
        ],
        "en": [
            ("{sch} applications are open",
             "Applications for {sch} are now open. The application deadline is {date}. Monthly stipend: {amount} TL."),
            ("{sch} deadline reminder",
             "Reminder: the deadline for {sch} is {date}. Results will be announced on {date2}."),
        ],
    },
    "cancellation": {
        "tr": [
            ("{course} dersi iptal",
             "{date} tarihinde yapılması planlanan {course} {cname} dersi, öğretim üyesinin rahatsızlığı nedeniyle yapılmayacaktır."),
            ("{course} ders iptali ve telafi",
             "{instructor} tarafından verilen {course} dersi {date} günü yapılmayacaktır. Telafi dersi {date2} tarihinde saat {time} itibarıyla {loc} dersliğinde yapılacaktır."),
            ("Ders iptali hakkında",
             "{course} kodlu {cname} dersi {date} tarihinde iptal edilmiştir. Ara sınav tarihinde herhangi bir değişiklik yoktur."),
            ("{cname} dersi yapılmayacak",
             "Bilimsel bir konferans nedeniyle {date} tarihli {course} dersi yapılamayacaktır. Ders notları sisteme yüklenecektir."),
            ("{course} - bugünkü ders",
             "Öğrencilerin dikkatine: {course} {cname} dersi {date} günü saat {time} oturumu iptal edilmiştir. Telafi tarihi daha sonra duyurulacaktır."),
        ],
        "en": [
            ("{course} class cancelled",
             "The {course} {cname} class on {date} is cancelled. A make-up class will be held on {date2} at {time} in {loc}."),
            ("No {course} lecture",
             "Please note that there will be no {course} lecture on {date} due to a conference."),
        ],
    },
    "room_change": {
        "tr": [
            ("{course} derslik değişikliği",
             "{course} {cname} dersi {date} tarihinden itibaren {loc} yerine {loc2} dersliğinde yapılacaktır."),
            ("Derslik değişikliği duyurusu",
             "{course} dersinin işleneceği derslik {loc} olarak değil, {loc2} olarak güncellenmiştir. Değişiklik {date} tarihinden itibaren geçerlidir."),
            ("{course} sınav salonu değişti",
             "{date} tarihinde yapılacak {course} {etype} {loc} yerine {loc2} salonunda yapılacaktır. Sınav saati ({time}) değişmemiştir."),
            ("{cname} ders yeri",
             "Kapasite nedeniyle {course} dersi {date} itibarıyla {loc} dersliğinden {loc2} dersliğine taşınmıştır."),
            ("Salon değişikliği: {course}",
             "Önemli: {course} dersinin {date} tarihli oturumu bakım çalışması nedeniyle {loc2} salonunda yapılacaktır."),
        ],
        "en": [
            ("{course} room change",
             "Starting {date}, {course} {cname} lectures will be held in {loc2} instead of {loc}."),
            ("{course} exam room changed",
             "The {course} {etype} on {date} has been moved from {loc} to {loc2}. The time ({time}) is unchanged."),
        ],
    },
    "registration": {
        "tr": [
            ("{topic} hakkında",
             "{topic_cap} {date} tarihinde başlayıp {date2} tarihinde sona erecektir. İşlemler öğrenci bilgi sistemi üzerinden yapılacaktır."),
            ("{topic_cap} takvimi",
             "Akademik takvim gereği {topic} {date} - {date2} tarihleri arasında yapılacaktır. Süresi içinde işlem yapmayan öğrencilerin başvuruları kabul edilmeyecektir."),
            ("Hatırlatma: {topic}",
             "{topic_cap} için son gün {date} olarak belirlenmiştir. Danışman onayı alınması zorunludur."),
            ("{topic_cap}",
             "{topic_cap} {date} tarihinde saat {time} itibarıyla başlayacaktır. Sistem yoğunluğu nedeniyle işlemlerinizi erken yapmanız önerilir."),
            ("Öğrenci işleri: {topic}",
             "Öğrenci İşleri Daire Başkanlığı'nın duyurusudur: {topic} {date} ile {date2} tarihleri arasında gerçekleştirilecektir. Bu tarihlerde sınav yapılmayacaktır."),
        ],
        "en": [
            ("{topic_en} schedule",
             "The {topic_en} period starts on {date} and ends on {date2}. All steps are completed through the student information system."),
            ("Reminder: {topic_en}",
             "The last day for {topic_en} is {date}. Advisor approval is required."),
        ],
    },
}
COUNTS = {"exam": 0.22, "event": 0.18, "scholarship": 0.14, "cancellation": 0.15,
          "room_change": 0.14, "registration": 0.17}
SUMMARY = {
    "tr": {
        "exam": "{course} {cname} {etype} {d} tarihinde saat {time}'da {loc} salonunda yapılacak.",
        "event": "{event} {d} tarihinde saat {time}'da {loc} salonunda gerçekleşecek.",
        "scholarship": "{sch} için son başvuru tarihi {d}.",
        "cancellation": "{course} dersi {d} tarihinde yapılmayacak.",
        "room_change": "{course} dersi/sınavı {d} tarihinden itibaren {loc2} salonunda yapılacak.",
        "registration": "{topic_cap} {d} tarihinde başlıyor.",
    },
    "en": {
        "exam": "{course} {cname} {etype} will be held on {d} at {time} in {loc}.",
        "event": "{event_en} takes place on {d} at {time} in {loc}.",
        "scholarship": "The deadline for {sch} is {d}.",
        "cancellation": "{course} class on {d} is cancelled.",
        "room_change": "{course} moves to {loc2} from {d}.",
        "registration": "{topic_en} starts on {d}.",
    },
}


EXAM_KIND = {"ara sınavı": "midterm", "vize sınavı": "midterm", "midterm exam": "midterm",
             "final sınavı": "final", "final exam": "final", "quiz": "quiz",
             "bütünleme sınavı": "resit", "mazeret sınavı": "makeup", "make-up exam": "makeup"}
KIND_TR = {"midterm": "ara sınavı", "final": "final sınavı", "quiz": "quiz", "resit": "bütünleme sınavı",
           "makeup": "mazeret sınavı"}
KIND_EN = {"midterm": "midterm exam", "final": "final exam", "quiz": "quiz", "resit": "resit exam",
           "makeup": "make-up exam"}


def generate(n: int, base: date, seed: int = 42):
    """Generate n announcements describing ONE consistent campus 'world': every exam
    (course+kind), event, scholarship and registration period has fixed facts, and
    repeated announcements about it are reminders with the same facts."""
    rng = random.Random(seed)
    cats = list(COUNTS)
    plan = []
    for c in cats:
        plan += [c] * round(n * COUNTS[c])
    while len(plan) < n:
        plan.append(rng.choice(cats))
    plan = plan[:n]
    rng.shuffle(plan)

    world: dict = {}

    def fact(key, maker):
        if key not in world:
            world[key] = maker()
        return world[key]

    def future(lo=1, hi=28):
        return base + timedelta(days=rng.randint(-10, hi - 7))  # some past, mostly upcoming

    exam_records = defaultdict(list)
    changed_exams = set()
    records = []
    for idx, cat in enumerate(plan, start=1):
        lang = "en" if rng.random() < 0.18 else "tr"
        course = rng.choice(COURSES)
        loc, loc2 = rng.sample(LOCATIONS, 2)
        t1 = rand_time(rng)
        d1 = future()
        d2 = d1 + timedelta(days=rng.randint(3, 14))
        kind = rng.choice(list(KIND_TR))
        ev_i = rng.randrange(len(EVENTS))
        sch = rng.choice(SCHOLARSHIPS)
        topic = rng.choice(REG_TOPICS)
        club = rng.choice(CLUBS)
        exam_ref = None

        if cat == "exam":
            w = fact(("exam", course[0], course[1], kind),
                     lambda: {"date": future(), "time": rand_time(rng, 9, 16), "loc": rng.choice(LOCATIONS)})
            d1, t1, loc = w["date"], w["time"], w["loc"]
        elif cat == "event":
            w = fact(("event", ev_i), lambda: {"date": future(), "time": rand_time(rng, 10, 19),
                                               "loc": rng.choice(EVENT_LOCATIONS), "club": rng.choice(CLUBS)})
            d1, t1, loc, club = w["date"], w["time"], w["loc"], w["club"]
        elif cat == "scholarship":
            w = fact(("sch", sch), lambda: {"date": future(5, 35), "loc": rng.choice(LOCATIONS),
                                            "time": rand_time(rng, 10, 16)})
            d1, loc, t1 = w["date"], w["loc"], w["time"]
            d2 = d1 + timedelta(days=10)
        elif cat == "registration":
            w = fact(("reg", topic[0]), lambda: {"date": future(), "time": "09:00"})
            d1, t1 = w["date"], w["time"]
            d2 = d1 + timedelta(days=4)
        elif cat == "cancellation":
            w = fact(("canc", course[0], course[1]), lambda: {"date": future()})
            d1 = w["date"]
        elif cat == "room_change":
            candidates = [k for k in exam_records if k not in changed_exams]
            if candidates and rng.random() < 0.45:
                key = rng.choice(candidates)
                changed_exams.add(key)
                exam_ref = exam_records[key][0]
                course = next(c for c in COURSES if c[0] == key[1] and c[1] == key[2])
                kind = key[3]
                w = world[key]
                d1, t1, loc = w["date"], w["time"], w["loc"]
                loc2 = rng.choice([x for x in LOCATIONS if x != loc])
                w["new_loc"] = loc2

        published = base - timedelta(days=rng.randint(0, 14))
        if published >= d1:
            published = d1 - timedelta(days=rng.randint(1, 5))

        if cat == "room_change":
            pool = [t for t in T[cat][lang] if ("{etype}" in t[1]) == bool(exam_ref)]
        else:
            pool = T[cat][lang]
        title_t, body_t = rng.choice(pool)
        etype = (KIND_EN if lang == "en" else KIND_TR)[kind]

        course_s = course_surface(course, rng)
        course_norm = f"{course[0]} {course[1]}"
        h, m = map(int, t1.split(":"))
        t2 = f"{min(h + 2, 21):02d}:{m:02d}"
        time_s = t1
        if lang == "tr" and "saat {time}" in body_t and rng.random() < 0.3:
            time_s = t1.replace(":", ".")
        ev = EVENTS[ev_i]
        values = {
            "course": course_s, "cname": course[3] if lang == "en" else course[2],
            "date": date_surface(d1, lang, rng), "date2": date_surface(d2, lang, rng),
            "time": time_s, "time2": t2, "loc": loc, "loc2": loc2, "etype": etype,
            "instructor": rng.choice(INSTRUCTORS), "club": club,
            "event": ev[0], "event_en": ev[1], "speaker": rng.choice(INSTRUCTORS),
            "sch": sch, "amount": rng.choice([1500, 2000, 2500, 3000, 4000]),
            "topic": topic[0], "topic_cap": topic[0][0].upper() + topic[0][1:], "topic_en": topic[1],
        }
        slots = {"course": ("COURSE", course_norm), "date": ("DATE", d1.isoformat()),
                 "date2": ("DATE", d2.isoformat()), "time": ("TIME", t1), "time2": ("TIME", t2),
                 "loc": ("LOCATION", loc), "loc2": ("LOCATION", loc2)}
        title = Rendered(title_t, values, {}).text
        body = Rendered(body_t, values, slots)
        text = body.text
        if rng.random() < 0.6:
            text += " " + rng.choice(FILLERS_EN if lang == "en" else FILLERS_TR)

        used = set(re.findall(r"\{(\w+)\}", body_t))
        fields = {}
        if "course" in used:
            fields["course"] = course_norm
        if "date" in used:
            fields["date"] = d1.isoformat()
        if "date2" in used:
            fields["date2"] = d2.isoformat()
        if "time" in used:
            fields["time"] = t1
        if "loc" in used:
            fields["location"] = loc
        if "loc2" in used:
            fields["location"] = loc2
            if "loc" in used:
                fields["old_location"] = loc
        if cat == "event":
            fields["event_name"] = ev[1] if lang == "en" else ev[0]
            fields["event_key"] = ev[0]
        if cat == "scholarship":
            fields["scholarship"] = sch
        if cat == "registration":
            fields["topic"] = topic[1] if lang == "en" else topic[0]
            fields["topic_key"] = topic[0]
        if cat == "exam" or exam_ref:
            fields["exam_type"] = etype
            fields["exam_kind"] = kind
        if exam_ref:
            fields["refers_to"] = exam_ref["id"]

        sv = dict(values, d=format_date_short(d1, lang), time=t1)
        summary = SUMMARY[lang][cat].format(**sv)
        rec = {
            "id": f"ann-{idx:04d}",
            "source": rng.choice(SOURCES) if cat != "event" else
            rng.choice(["student_club", "student_club", "university_website", "email"]),
            "lang": lang, "published": published.isoformat(), "title": title, "text": text,
            "category": cat, "entities": body.entities, "fields": fields, "summary": summary,
        }
        records.append(rec)
        if cat == "exam":
            exam_records[("exam", course[0], course[1], kind)].append(rec)
    return records, world


# --------------------------------------------------------------------------- QA set
def week_range(d: date):
    start = d - timedelta(days=d.weekday())
    return start, start + timedelta(days=6)


def build_qa(records, world, base: date, seed: int = 7, n: int = 50):
    rng = random.Random(seed)
    qs = []

    def add(q, lang, ids, facts, qtype):
        qs.append({"id": f"q-{len(qs) + 1:03d}", "question": q, "lang": lang, "type": qtype,
                   "expected_ids": sorted(ids), "expected_facts": facts})

    def ids_where(pred):
        return [r["id"] for r in records if pred(r)]

    exam_keys = [k for k in world if k[0] == "exam"]
    rng.shuffle(exam_keys)
    # exam_when: 14
    for k in exam_keys[:14]:
        _, pre, num, kind = k
        code, w = f"{pre} {num}", world[k]
        ids = ids_where(lambda r: r["category"] == "exam" and r["fields"].get("course") == code
                        and r["fields"].get("exam_kind") == kind)
        if rng.random() < 0.2:
            q, lang = f"When is the {code} {KIND_EN[kind]}?", "en"
        else:
            q = rng.choice([f"{code} {KIND_TR[kind]} ne zaman?", f"{code} {KIND_TR[kind]} hangi gün ve saatte?",
                            f"{pre}{num} {KIND_TR[kind]} ne zaman yapılacak?"])
            lang = "tr"
        add(q, lang, ids, [{"type": "DATE", "value": w["date"].isoformat()}, {"type": "TIME", "value": w["time"]}],
            "exam_when")
    # exam_where: 8 (prefer exams whose room was changed)
    changed = [k for k in exam_keys if "new_loc" in world[k]]
    unchanged = [k for k in exam_keys[14:] if "new_loc" not in world[k]]
    for k in changed[:6] + unchanged[:2]:
        _, pre, num, kind = k
        code, w = f"{pre} {num}", world[k]
        ids = ids_where(lambda r: r["category"] in ("exam", "room_change") and r["fields"].get("course") == code
                        and r["fields"].get("exam_kind") == kind)
        add(rng.choice([f"{code} {KIND_TR[kind]} nerede?", f"{code} {KIND_TR[kind]} hangi salonda yapılacak?"]),
            "tr", ids, [{"type": "LOCATION", "value": w.get("new_loc", w["loc"])}], "exam_where")
    # scholarships: 6
    sch_keys = [k for k in world if k[0] == "sch"]
    rng.shuffle(sch_keys)
    for k in sch_keys[:6]:
        ids = ids_where(lambda r: r["fields"].get("scholarship") == k[1])
        if rng.random() < 0.25:
            q, lang = f"What is the application deadline for {k[1]}?", "en"
        else:
            q, lang = f"{k[1]} son başvuru tarihi ne zaman?", "tr"
        add(q, lang, ids, [{"type": "DATE", "value": world[k]["date"].isoformat()}], "scholarship_deadline")
    # cancellations: 6
    canc_keys = [k for k in world if k[0] == "canc"]
    rng.shuffle(canc_keys)
    for k in canc_keys[:6]:
        code = f"{k[1]} {k[2]}"
        ids = ids_where(lambda r: r["category"] == "cancellation" and r["fields"].get("course") == code)
        add(rng.choice([f"{code} dersi iptal mi?", f"{code} dersi hangi gün yapılmayacak?"]), "tr", ids,
            [{"type": "DATE", "value": world[k]["date"].isoformat()}], "cancellation")
    # events: 5
    ev_keys = [k for k in world if k[0] == "event"]
    rng.shuffle(ev_keys)
    for k in ev_keys[:5]:
        tr_name, en_name = EVENTS[k[1]]
        ids = ids_where(lambda r: r["fields"].get("event_key") == tr_name)
        w = world[k]
        facts = [{"type": "DATE", "value": w["date"].isoformat()}, {"type": "LOCATION", "value": w["loc"]}]
        if rng.random() < 0.3:
            add(f"When and where is the {en_name}?", "en", ids, facts, "event")
        else:
            add(f"{tr_name} ne zaman ve nerede?", "tr", ids, facts, "event")
    # registration: 4
    reg_keys = [k for k in world if k[0] == "reg"]
    rng.shuffle(reg_keys)
    for k in reg_keys[:4]:
        tr_t = k[1]
        en_t = dict(REG_TOPICS)[tr_t]
        ids = ids_where(lambda r: r["fields"].get("topic_key") == tr_t)
        facts = [{"type": "DATE", "value": world[k]["date"].isoformat()}]
        if rng.random() < 0.3:
            add(f"When does {en_t} start?", "en", ids, facts, "registration")
        else:
            add(f"{tr_t[0].upper() + tr_t[1:]} ne zaman başlıyor?", "tr", ids, facts, "registration")
    # time-range listing questions
    ranges = [("bu hafta", "this week", week_range(base)),
              ("yarın", "tomorrow", (base + timedelta(days=1),) * 2),
              ("gelecek hafta", "next week", week_range(base + timedelta(days=7))),
              ("bugün", "today", (base, base))]
    for cat, tr_word, en_word, wkey, label in [("exam", "sınav", "exams", "exam", "course"),
                                              ("event", "etkinlik", "events", "event", "event")]:
        for tr_r, en_r, (s, e) in ranges:
            keys = [k for k in world if k[0] == wkey and s <= world[k]["date"] <= e]
            if not 1 <= len(keys) <= 8:
                continue
            if cat == "exam":
                facts = [{"type": "TEXT", "value": f"{k[1]} {k[2]}"} for k in keys]
                kk = {(f"{k[1]} {k[2]}", k[3]) for k in keys}
                ids = ids_where(lambda r: r["category"] == "exam"
                                and (r["fields"].get("course"), r["fields"].get("exam_kind")) in kk)
            else:
                facts = [{"type": "TEXT_ANY", "value": list(EVENTS[k[1]])} for k in keys]
                names = {EVENTS[k[1]][0] for k in keys}
                ids = ids_where(lambda r: r["fields"].get("event_key") in names)
            facts = [dict(t) for t in {tuple((a, tuple(b) if isinstance(b, list) else b) for a, b in f.items())
                                       for f in facts}]
            facts = [{k: (list(v) if isinstance(v, tuple) else v) for k, v in f.items()} for f in facts]
            if rng.random() < 0.3:
                add(f"Are there any {en_word} {en_r}?", "en", ids, facts, f"list_{cat}")
            else:
                add(f"{tr_r[0].upper() + tr_r[1:]} {tr_word} var mı?", "tr", ids, facts, f"list_{cat}")
    for q, lang in [("MAT 999 sınavı ne zaman?", "tr"), ("When is the CENG 998 final exam?", "en"),
                    ("HUK 777 dersi iptal mi?", "tr")]:
        add(q, lang, [], [{"type": "NO_ANSWER", "value": ""}], "no_answer")
    # top up with "what time" exam questions
    for k in exam_keys[14:]:
        if len(qs) >= n:
            break
        _, pre, num, kind = k
        code = f"{pre} {num}"
        ids = ids_where(lambda r: r["category"] == "exam" and r["fields"].get("course") == code
                        and r["fields"].get("exam_kind") == kind)
        add(f"{code} {KIND_TR[kind]} saat kaçta?", "tr", ids, [{"type": "TIME", "value": world[k]["time"]}],
            "exam_when")
    return qs[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=420)
    ap.add_argument("--base-date", default="2026-11-09",
                    help="reference 'today' of the dataset (YYYY-MM-DD or 'today')")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "data"))
    args = ap.parse_args()
    base = date.today() if args.base_date == "today" else date.fromisoformat(args.base_date)
    records, world = generate(args.n, base, args.seed)
    qa = build_qa(records, world, base)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "announcements.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "qa_eval.jsonl", "w", encoding="utf-8") as fh:
        for q in qa:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    meta = {"base_date": base.isoformat(), "n_announcements": len(records), "n_questions": len(qa),
            "seed": args.seed, "synthetic": True}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    cats = defaultdict(int)
    for r in records:
        cats[r["category"]] += 1
    print(f"Wrote {len(records)} announcements, {len(qa)} QA questions to {out}")
    print("Category counts:", dict(cats))


if __name__ == "__main__":
    main()
