"""Collect real announcements from Gebze Technical University (www.gtu.edu.tr).

    python scripts/collect_gtu.py                      # departments in scripts/gtu_sources.json (CENG first)
    python scripts/collect_gtu.py --dept ceng --limit 50
    python scripts/collect_gtu.py --discover           # find every faculty/department + its announcement list
                                                       #   -> scripts/gtu_sources.json (review, then collect)
    python scripts/collect_gtu.py --all                # collect every department in gtu_sources.json

Site structure (checked Nov 2026):
  * department page  : /kategori/<id>/3/<slug>.aspx     (listed in the site menu JSON)
  * announcement list: /tr/kategori/<annId>/0/display.aspx   (the "Duyurular >> Tümü" link on the dept page)
  * one announcement : /tr/icerik/<annId>/<itemId>/display.aspx -> <h1> title + <div class="rich-text-content">
  * many announcements are PDFs (exam / course schedules) -> stored with title + url (kind = "pdf")
The site shows no publication date, so `published` is the collection date (`published_estimated: true`).
There is no robots.txt; we still send one request per second and identify ourselves.

Output: data/gtu/<dept>.jsonl, raw schema (id, source, department, lang, published, title, text, url, kind).
These rows have no labels yet: the app classifies them; for evaluation they must be annotated
(docs/annotation_guidelines.md).
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
import unicodedata
import urllib.request
from datetime import date
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from campus_assistant.textutils import clean_text, detect_lang  # noqa: E402

BASE = "https://www.gtu.edu.tr"
UA = "SmartCampusAssistant/0.1 (student NLP research project, Gebze Technical University course)"
SOURCES = ROOT / "scripts" / "gtu_sources.json"
OUT_DIR = ROOT / "data" / "gtu"
DELAY = 1.0

DEFAULT_SOURCES = [
    {"dept": "ceng", "name": "Bilgisayar Mühendisliği", "faculty": "Mühendislik Fakültesi",
     "page": f"{BASE}/kategori/91/3/bilgisayar-muhendisligi.aspx",
     "announcements": f"{BASE}/tr/kategori/120/0/display.aspx"},
]


def safe_url(url: str) -> str:
    p = urlsplit(url)
    return urlunsplit((p.scheme, p.netloc, quote(p.path, safe="/%"), p.query, p.fragment))


def fetch(url: str) -> str:
    req = urllib.request.Request(safe_url(url), headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=25) as r:
        return r.read().decode(r.headers.get_content_charset() or "utf-8", errors="replace")


def text_of(fragment: str) -> str:
    fragment = re.sub(r"(?is)<(script|style).*?</\1>", " ", fragment)
    fragment = re.sub(r"(?i)<br\s*/?>|</p>|</li>|</tr>|</h\d>", "\n", fragment)
    fragment = re.sub(r"(?i)</td>", " ", fragment)
    txt = html.unescape(re.sub(r"<[^>]+>", " ", fragment)).replace("\xa0", " ")
    lines = [" ".join(l.split()) for l in txt.splitlines()]
    return clean_text("\n".join(l for l in lines if l))


def list_items(list_url: str):
    """(url, title) pairs of one announcement list page, newest first."""
    page = fetch(list_url)
    start = page.find("list-disc")
    seg = page[start:] if start >= 0 else page
    out, seen = [], set()
    for href, title in re.findall(r'<li[^>]*>\s*<a href="([^"]+)"[^>]*>(.*?)</a>\s*</li>', seg, re.S):
        url = urljoin(BASE + "/", html.unescape(href))
        if url in seen:
            continue
        seen.add(url)
        out.append((url, text_of(title)))
    return out


def detail(url: str):
    page = fetch(url)
    h1 = re.search(r"<h1[^>]*>(.*?)</h1>", page, re.S)
    body = re.search(r'<div class="rich-text-content">(.*?)</div>\s*(?:</div>|<div|<footer)', page, re.S)
    return (text_of(h1.group(1)) if h1 else ""), (text_of(body.group(1)) if body else "")


def collect(src: dict, limit: int, include_pdf: bool = True):
    today = date.today().isoformat()
    items = list_items(src["announcements"])
    print(f"[{src['dept']}] {src['name']}: {len(items)} items in list, taking the newest {limit}")
    rows = []
    for url, title in items[:limit]:
        is_pdf = urlsplit(url).path.lower().endswith(".pdf")
        if is_pdf:
            if not include_pdf:
                continue
            text, kind = title, "pdf"
        elif "/icerik/" in url and "gtu.edu.tr" in url:
            time.sleep(DELAY)
            try:
                t2, text = detail(url)
            except Exception as exc:  # network error, removed page ...
                print(f"  skip {url}: {exc}")
                continue
            title = t2 or title
            text, kind = (text or title), "page"
        else:
            text, kind = title, "link"
        hid = hashlib.sha1(url.encode()).hexdigest()[:8]
        rows.append({"id": f"gtu-{src['dept']}-{hid}", "source": "department_website", "university": "GTÜ",
                     "department": src["name"], "faculty": src.get("faculty"), "lang": detect_lang(title + " " + text),
                     "published": today, "published_estimated": True, "title": title, "text": text,
                     "url": url, "kind": kind})
        print(f"  {kind:4s} {title[:80]}")
    return rows


def discover():
    """Read the site menu (Next.js payload on the home page) and find each department's announcement list."""
    home = fetch(BASE + "/").replace('\\"', '"')
    menu = re.findall(r'"title":"([^"]+)","adress":"([^"]+)","isAnnouncement":(?:true|false),"isNewPage":'
                      r'(?:true|false),"description":(?:null|"[^"]*"),"children":\[\]', home)
    fac = re.findall(r'"title":"([^"]*(?:Fakültesi|Enstitüsü|Yüksekokulu))","adress"', home)
    print(f"menu entries: {len(menu)}; faculties/institutes seen: {sorted(set(fac))}")
    out = []
    for title, adress in menu:
        if "/kategori/" not in adress or "/3/" not in adress or title in ("Fakülte Sayfası",):
            continue
        if not re.search(r"(?i)mühendisli|bilim|fizik|kimya|matematik|biyoloji|mimarl|planlama|tasarım|işletme|"
                         r"ekonomi|yönetim|bölüm", title):
            continue
        time.sleep(DELAY)
        try:
            page = fetch(adress)
        except Exception as exc:
            print(f"  skip {title}: {exc}")
            continue
        m = re.findall(r'href="(/tr/kategori/(\d+)/0/display\.aspx)"', page)
        if not m:
            print(f"  {title}: no announcement list found")
            continue
        ann = BASE + m[0][0]
        ascii_title = "".join(c for c in unicodedata.normalize("NFKD", title.replace("ı", "i").replace("İ", "I"))
                              if not unicodedata.combining(c))
        dept = re.sub(r"[^a-z0-9]+", "-", ascii_title.lower()).strip("-")
        out.append({"dept": dept, "name": title, "faculty": None, "page": adress, "announcements": ann})
        print(f"  {title:45s} -> {ann}")
    SOURCES.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(out)} departments to {SOURCES}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dept", default="ceng", help="department key in gtu_sources.json")
    ap.add_argument("--all", action="store_true", help="collect every department in gtu_sources.json")
    ap.add_argument("--limit", type=int, default=40, help="newest N items per department")
    ap.add_argument("--no-pdf", action="store_true", help="skip PDF-only announcements")
    ap.add_argument("--discover", action="store_true")
    args = ap.parse_args()
    if args.discover:
        return discover()
    sources = json.loads(SOURCES.read_text(encoding="utf-8")) if SOURCES.exists() else DEFAULT_SOURCES
    todo = sources if args.all else [s for s in sources if s["dept"] == args.dept]
    if not todo:
        sys.exit(f"unknown --dept {args.dept}; known: {[s['dept'] for s in sources]}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for src in todo:
        rows = collect(src, args.limit, include_pdf=not args.no_pdf)
        path = OUT_DIR / f"{src['dept']}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"-> {len(rows)} announcements written to {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
