"""Collect announcements from university web pages / RSS feeds (proposal: "Collect & clean").

Configure sources in scripts/sources.json, e.g.
    [{"name": "ceng", "type": "rss",  "url": "https://example.edu.tr/duyurular/rss"},
     {"name": "main", "type": "html", "url": "https://example.edu.tr/duyurular",
      "item_selector_regex": "<a[^>]+href=\\"(/duyuru/[^\\"]+)\\"[^>]*>(.*?)</a>"}]

    python scripts/collect_announcements.py --out data/raw_announcements.jsonl

Output rows have the same raw schema as the dataset (id, source, title, text, published, url)
and still need manual labels (see docs/annotation_guidelines.md). Respect each site's
robots.txt and terms of use; keep request rates low.
"""
import argparse
import hashlib
import html
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from campus_assistant.textutils import clean_text  # noqa: E402

UA = "SmartCampusAssistant/0.1 (student research project)"


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read().decode(r.headers.get_content_charset() or "utf-8", errors="replace")


def strip_html(s):
    s = re.sub(r"(?is)<(script|style).*?</\1>", " ", s)
    s = re.sub(r"(?i)<br\s*/?>|</p>", "\n", s)
    return clean_text(html.unescape(re.sub(r"<[^>]+>", " ", s)))


def from_rss(src):
    root = ET.fromstring(fetch(src["url"]))
    for item in root.iter("item"):
        title = item.findtext("title") or ""
        body = item.findtext("description") or ""
        link = item.findtext("link")
        pub = item.findtext("pubDate")
        try:
            pub = parsedate_to_datetime(pub).date().isoformat() if pub else date.today().isoformat()
        except Exception:
            pub = date.today().isoformat()
        yield {"title": strip_html(title), "text": strip_html(body), "url": link, "published": pub}


def from_html(src):
    page = fetch(src["url"])
    for m in re.finditer(src["item_selector_regex"], page, re.S):
        url = urljoin(src["url"], m.group(1))
        time.sleep(1.0)
        try:
            detail = fetch(url)
        except Exception as exc:
            print(f"  skip {url}: {exc}")
            continue
        body = re.search(r"(?is)<(article|main)[^>]*>(.*?)</\1>", detail)
        yield {"title": strip_html(m.group(2)), "text": strip_html(body.group(2) if body else detail), "url": url,
               "published": date.today().isoformat()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=str(ROOT / "scripts" / "sources.json"))
    ap.add_argument("--out", default=str(ROOT / "data" / "raw_announcements.jsonl"))
    args = ap.parse_args()
    sources = json.loads(Path(args.sources).read_text(encoding="utf-8"))
    seen, rows = set(), []
    for src in sources:
        print(f"[{src['name']}] {src['url']}")
        gen = from_rss(src) if src["type"] == "rss" else from_html(src)
        try:
            for item in gen:
                h = hashlib.sha1((item["title"] + item["text"]).encode()).hexdigest()[:10]
                if h in seen or not item["text"]:
                    continue
                seen.add(h)
                rows.append({"id": f"{src['name']}-{h}", "source": src.get("source_type", "university_website"),
                             **item})
        except Exception as exc:
            print(f"  failed: {exc}")
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} announcements to {args.out}")


if __name__ == "__main__":
    main()
