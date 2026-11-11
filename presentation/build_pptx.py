"""Build the mid-project deck as PowerPoint (.pptx), same content as build_slides.py (PDF).

    pip install python-pptx
    python presentation/build_pptx.py        # -> presentation/TeamXX_MidProject.pptx
Numbers come from reports/midproject_results.json and reports/midproject_errors.csv.
Speaker notes (Nisanur's script) are on every presented slide.
"""
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
TEAM = "Team##"  # <- your team number
MEMBERS = "Helin Saygılı · Nisanur Kara · Cevahir Atıcı · Emre Kılıç"
REPO = "https://github.com/<team>/smart-campus-assistant"

R = json.loads((ROOT / "reports" / "midproject_results.json").read_text(encoding="utf-8"))
ERR = list(csv.DictReader(open(ROOT / "reports" / "midproject_errors.csv", encoding="utf-8")))
B, SP = R["baselines"], R["split"]
CNT = Counter(e["category"] for e in ERR)
N_CLS = sum(v for k, v in CNT.items() if k.startswith("E"))
N_EX = sum(v for k, v in CNT.items() if k.startswith("X"))
bt = R.get("berturk_dev", [])
f1s, accs = [s["macro_f1"] for s in bt], [s["accuracy"] for s in bt]
BT_F1 = f"{statistics.mean(f1s):.3f} ± {statistics.pstdev(f1s):.3f}" if bt else "not run yet"
BT_ACC = f"{statistics.mean(accs):.3f} ± {statistics.pstdev(accs):.3f}" if bt else "not run yet"
BT_SEEDS = ", ".join(f"{x:.3f}" for x in f1s)
tf, kw = B["tfidf_logreg"], B["keyword_rules"]

NAVY, TEAL, INK, MUTED = "16264A", "0F766E", "1C2333", "6B7489"
TINT, GREY, WARN_T, WARN, GOLD_T = "E6F2F0", "F1F4F8", "FBEDEA", "B4401A", "F6F1E4"
WHITE, NAVY2 = "FFFFFF", "24365F"
FONT = "Calibri"


def rgb(h):
    return RGBColor.from_string(h)


prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def text(slide, x, y, w, h, content, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         italic=False, spacing=4, name=None):
    """content: str | list of paragraphs; a paragraph is str or list of (text, {bold,color,size,italic})."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf_ = tb.text_frame
    tf_.word_wrap = True
    tf_.vertical_anchor = anchor
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf_, side, 0)
    paras = content if isinstance(content, list) else [content]
    for i, para in enumerate(paras):
        p = tf_.paragraphs[0] if i == 0 else tf_.add_paragraph()
        p.alignment = align
        p.space_after = Pt(spacing)
        runs = para if isinstance(para, list) else [(para, {})]
        for t, o in runs:
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = FONT
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", italic)
            f.color.rgb = rgb(o.get("color", color))
    return tb


def box(slide, x, y, w, h, fill=GREY, name=None):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = 0.06
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    s.line.fill.background()
    s.shadow.inherit = False
    if name:
        s.name = name
    return s


def card(slide, x, y, w, h, label, body, fill=GREY, label_color=TEAL, size=15, body_color=INK):
    box(slide, x, y, w, h, fill, name=f"card {label}")
    text(slide, x + 0.25, y + 0.18, w - 0.5, 0.3, label.upper(), size=11, bold=True, color=label_color)
    text(slide, x + 0.25, y + 0.5, w - 0.5, h - 0.65, body, size=size, color=body_color)


def header(slide, kicker, title, dark=False):
    text(slide, 0.6, 0.4, 12, 0.3, kicker.upper(), size=12, bold=True, color="7FD1C6" if dark else TEAL)
    text(slide, 0.6, 0.7, 12.1, 0.8, title, size=30, bold=True, color=WHITE if dark else NAVY, name="Title")


def footer(slide, n, label="Mid-project · CSE 484/654", dark=False):
    c = "9AA6C2" if dark else "9AA2B3"
    text(slide, 0.6, 7.05, 6, 0.25, f"{TEAM} · Smart Campus Announcement Assistant", size=10, color=c)
    text(slide, 6.2, 7.05, 4, 0.25, label, size=10, color=c, align=PP_ALIGN.CENTER)
    text(slide, 11.7, 7.05, 1.0, 0.25, str(n), size=10, color=c, align=PP_ALIGN.RIGHT)


def bg(slide, color):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = rgb(color)


def table(slide, x, y, w, rows, col_w, size=13, header_fill=NAVY, hl_rows=(), row_h=0.36):
    t = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w),
                               Inches(row_h * len(rows))).table
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.08)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            c.fill.fore_color.rgb = rgb(header_fill if i == 0 else (TINT if i in hl_rows else WHITE))
            tf_ = c.text_frame
            tf_.word_wrap = True
            p = tf_.paragraphs[0]
            runs = val if isinstance(val, list) else [(str(val), {})]
            for tx, o in runs:
                r = p.add_run()
                r.text = tx
                r.font.name = FONT
                r.font.size = Pt(o.get("size", size))
                r.font.bold = o.get("bold", i == 0 or i in hl_rows)
                r.font.italic = o.get("italic", False)
                r.font.color.rgb = rgb(WHITE if i == 0 else o.get("color", INK))
    return t


def ci(r):
    return f"[{r['f1_ci'][0]:.2f}, {r['f1_ci'][1]:.2f}]"


def f3(x):
    return f"{x:.3f}"


NOTES = {}
for i, block in enumerate((OUT / "Nisa_konusma_EN.md").read_text(encoding="utf-8").split("## Slide ")[1:], 1):
    NOTES[i] = block.split("\n", 1)[1].strip().strip('"').replace('**', '').replace('"\n', '\n')

# ── 1 research question (dark)
s = prs.slides.add_slide(BLANK)
bg(s, NAVY)
text(s, 0.6, 0.45, 12, 0.3, f"{TEAM} · {MEMBERS}".upper(), size=12, bold=True, color="7FD1C6")
text(s, 0.6, 0.8, 12, 0.8, "Smart Campus Announcement Assistant", size=36, bold=True, color=WHITE, name="Title")
card(s, 0.6, 1.85, 12.1, 1.55, "Research question",
     [[("Does fine-tuning BERTurk classify campus announcements more accurately (Macro-F1) than a TF-IDF + "
        "logistic-regression baseline, on announcements written outside our generation templates?",
        {"bold": True, "size": 20})]], fill=NAVY2, label_color="7FD1C6", body_color=WHITE)
card(s, 0.6, 3.6, 5.85, 2.4, "What a “no” looks like",
     ["BERTurk gains < 0.03 Macro-F1, or the paired-bootstrap 95% CI of the gain includes 0.",
      "→ A linear model is enough for stage 01; effort moves to extraction and QA."],
     fill=NAVY2, label_color="F4A582", body_color=WHITE, size=16)
card(s, 6.85, 3.6, 5.85, 2.4, "Changed since the proposal — why",
     [[("Old: ", {"bold": True}), ("“How can NLP techniques turn scattered announcements into structured, searchable "
                                   "information and answer students' questions?”", {})],
      [("Why: ", {"bold": True}), (f"a “how” question has no “no” result, and TF-IDF already scores "
                                   f"{f3(tf['test']['macro_f1'])} on our template test set.", {})]],
     fill=NAVY2, label_color="E9C46A", body_color=WHITE, size=15)
footer(s, 1, dark=True)

# ── 2 data
s = prs.slides.add_slide(BLANK)
header(s, "Data as loaded", "420 labelled + 24 hand-written + 818 real GTÜ announcements")
table(s, 0.6, 1.65, 7.4, [
    ["Set", "Source", "N", "TR / EN"],
    ["Train", "Synthetic “campus world” (seed 42)", str(SP["train"]), "198 / 54"],
    ["Dev", "Synthetic", str(SP["dev"]), "66 / 18"],
    ["Test A (frozen)", "Synthetic, held out", str(SP["test"]), "71 / 13"],
    ["Test B (frozen)", "Hand-written, never trained on", str(SP["challenge"]), "19 / 5"],
    ["GTÜ raw (unlabelled)", "gtu.edu.tr, 24 departments, Nov 11", "818", "≈ all TR"],
], [1.9, 3.6, 0.8, 1.1], size=13, hl_rows=(3, 4), row_h=0.42)
text(s, 0.6, 4.4, 7.4, 2.4, [
    "6 classes: exam 92 · event 76 · registration 71 · cancellation 63 · room change 59 · scholarship 59.",
    "Split: stratified 60/20/20, seed 2026 → data/splits/midproject_split.json (sha256 2cafd21d7400).",
    "Preprocessing: 420 → clean_text 420 → dedup 420. 1,292 entity spans.",
    [("Licence: ", {"bold": True}), ("our text CC BY 4.0; GTÜ pages © GTÜ → we share only URLs + our labels.", {})],
], size=14, color=INK, spacing=6)
card(s, 8.4, 1.65, 4.3, 3.0, "One real example · ann-0001",
     [[("“1 Kasım 2026 tarihinde yapılması planlanan KIM107 Genel Kimya dersi, öğretim üyesinin rahatsızlığı "
        "nedeniyle yapılmayacaktır.”", {"italic": True})],
      [("category ", {"bold": True}), ("cancellation", {})],
      [("entities ", {"bold": True}), ("DATE 2026-11-01 · COURSE KIM 107", {})]], size=14)
card(s, 8.4, 4.85, 4.3, 1.95, "Honest caveat",
     "Train / dev / test A are synthetic and share vocabulary with our rules — hence Test B and the real Test C.",
     fill=WARN_T, label_color=WARN, size=14)
footer(s, 2)

# ── 3 baseline
s = prs.slides.add_slide(BLANK)
header(s, "Baseline result", "Test A is saturated; out-of-template text is where the gap is")
names = [("majority", "Majority class"), ("keyword_rules", "Keyword rules"), ("tfidf_nb", "TF-IDF + NB"),
         ("tfidf_svm", "TF-IDF + linear SVM"), ("tfidf_logreg", "TF-IDF + LogReg (main)")]
rows = [["System (trained on 252)", "Dev Macro-F1", "Test A Macro-F1 [95% CI]", "Test A Acc",
         "Test B Macro-F1 [95% CI]", "Test B Acc"]]
for k, n in names:
    r = B[k]
    rows.append([n, f3(r["dev"]["macro_f1"]), f"{f3(r['test']['macro_f1'])}  {ci(r['test'])}",
                 f3(r["test"]["accuracy"]), f"{f3(r['challenge']['macro_f1'])}  {ci(r['challenge'])}",
                 f3(r["challenge"]["accuracy"])])
table(s, 0.6, 1.6, 12.1, rows, [3.0, 1.5, 2.4, 1.4, 2.4, 1.4], size=13, hl_rows=(5,), row_h=0.4)
for i, (big, lab) in enumerate([(f3(tf["challenge"]["macro_f1"]), "TF-IDF + LogReg on hand-written Test B (n = 24)"),
                                (f"−{tf['test']['macro_f1'] - tf['challenge']['macro_f1']:.3f}",
                                 "drop from Test A to Test B")]):
    x = 0.6 + i * 3.2
    box(s, x, 4.3, 3.0, 2.4, TINT if i == 0 else WARN_T)
    text(s, x + 0.25, 4.5, 2.6, 1.0, big, size=48, bold=True, color=TEAL if i == 0 else WARN)
    text(s, x + 0.25, 5.6, 2.6, 0.9, lab, size=13, color=INK)
text(s, 7.2, 4.4, 5.5, 2.4, [
    [("Spread: ", {"bold": True}), ("baselines are deterministic, so we report a 95% bootstrap CI "
                                    "(1,000 resamples of the test set, seed 0).", {})],
    "Test B's CI is ~0.3 wide: with 24 items we cannot resolve a 0.03 difference → we need a larger real test set.",
    [("Reproduce: ", {"bold": True}), ("python evaluation/midproject.py", {"color": TEAL})],
], size=15, spacing=8)
footer(s, 3)

# ── 4 error analysis (native bar chart + examples)
s = prs.slides.add_slide(BLANK)
header(s, "Error analysis", f"{N_CLS} baseline errors in 7 categories")
cats = [("E1", "Topic beats action"), ("E4", "No keyword fires"), ("E2", "Shared cue “başvuru”"),
        ("E3", "Registration about an exam"), ("E5", "Event mentions exams"), ("E7", "TF-IDF: unseen wording"),
        ("E6", "Room name = event cue")]
cd = CategoryChartData()
cd.categories = [f"{k} · {n}" for k, n in reversed(cats)]
cd.add_series("Errors", [CNT[k] for k, _ in reversed(cats)])
ch = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.6), Inches(1.6), Inches(6.6), Inches(5.2), cd).chart
ch.has_legend = False
ch.has_title = False
ch.plots[0].gap_width = 45
ch.plots[0].has_data_labels = True
dl = ch.plots[0].data_labels
dl.position = XL_LABEL_POSITION.OUTSIDE_END
dl.font.size, dl.font.bold, dl.font.name = Pt(13), True, FONT
dl.font.color.rgb = rgb(NAVY)
ser = ch.series[0]
ser.format.fill.solid()
ser.format.fill.fore_color.rgb = rgb(TEAL)
ch.value_axis.visible = False
ch.value_axis.has_major_gridlines = False
ch.category_axis.tick_labels.font.size = Pt(13)
ch.category_axis.tick_labels.font.name = FONT
ch.category_axis.tick_labels.font.color.rgb = rgb(INK)
ch.category_axis.format.line.fill.background()
ex = [("E1 · room_change → exam", "“… CENG 213 mazeret sınavı Kütüphane Seminer Salonu yerine Kültür Merkezi "
                                  "salonunda yapılacaktır.”"),
      ("E4 · registration → exam", "“The double major applications period starts on November 18, 2026 …”"),
      ("E2 · registration → scholarship", "“Akademik takvim gereği çift anadal başvuruları 18.11.2026 - 22.11.2026 …”")]
for i, (lab, quote) in enumerate(ex):
    card(s, 7.6, 1.6 + i * 1.62, 5.1, 1.45, lab, [[(quote, {"italic": True})]], size=13)
text(s, 7.6, 6.5, 5.1, 0.45, "Keyword rules: 45 train + 12 dev + 5 Test B; TF-IDF: 3 Test B. Test A not inspected.",
     size=11, color=MUTED)
footer(s, 4)

# ── 5 what errors suggest (2x2)
s = prs.slides.add_slide(BLANK)
header(s, "Error analysis", "What the errors tell us")
grid = [
    (f"1 · Label = action, not topic  ({CNT['E1'] + CNT['E3'] + CNT['E5']})",
     "“MAT 102 sınavı Konferans Salonu yerine E-205'te” is a room change, but every cue says exam. "
     "Bag-of-words cannot weigh “X yerine Y”; a contextual encoder can.", TINT, TEAL),
    (f"2 · Vocabulary gaps  ({CNT['E4'] + CNT['E7']})",
     "“guest lecture”, “tuition waiver”, “ilişik kesme” never appear in training — pretrained subwords may help.",
     TINT, TEAL),
    (f"3 · Ambiguous shared cues  ({CNT['E2'] + CNT['E6']})",
     "TF-IDF already learns “başvuru” and “Seminer Salonu” from data (0 errors on Test A): a rules problem.",
     GREY, NAVY),
    (f"4 · Extraction on Test B  ({N_EX})",
     f"Year-less date ranges ({CNT['X1']}), free-form places ({CNT['X2']}), dotted time “17.30'da” ({CNT['X3']}). "
     "Micro-F1 still 0.947.", GREY, NAVY),
]
for i, (lab, body, fill, lc) in enumerate(grid):
    x, y = 0.6 + (i % 2) * 6.15, 1.6 + (i // 2) * 2.25
    card(s, x, y, 5.95, 2.05, lab, body, fill=fill, label_color=lc, size=15)
box(s, 0.6, 6.15, 12.1, 0.75, NAVY)
text(s, 0.85, 6.15, 11.6, 0.75, [[("So: ", {"bold": True, "color": "7FD1C6"}),
                                  ("expect BERTurk to help on 1 and 2 only — measure on Test B and the real Test C, "
                                   "not on Test A.", {"color": WHITE})]], size=16, anchor=MSO_ANCHOR.MIDDLE)
footer(s, 5)

# ── 6 method status
s = prs.slides.add_slide(BLANK)
header(s, "Method status", "BERTurk fine-tuning implemented and run")
card(s, 0.6, 1.6, 5.9, 1.75, "The one controlled change",
     "Replace TF-IDF + LogReg with a fine-tuned BERTurk encoder (dbmdz/bert-base-turkish-cased). "
     "Same split, input (title + text), labels and metric.", size=15)
card(s, 0.6, 3.55, 5.9, 1.75, "As implemented",
     ["4 epochs · lr 3e-5 · batch 16 · max length 128 · seeds 13, 42, 2026",
      "~6 min / seed on a laptop CPU; final runs ≈ 2 GPU-hours (Colab T4)."], size=15)
card(s, 0.6, 5.5, 5.9, 1.35, "Rest of the pipeline",
     "Extraction F1 0.947 · summaries ROUGE-L 0.953 · QA 50/50", size=14)
table(s, 6.9, 1.6, 5.8, [["Dev set (n = 84)", "Macro-F1", "Accuracy"],
                         ["Keyword rules", f3(kw["dev"]["macro_f1"]), f3(kw["dev"]["accuracy"])],
                         ["TF-IDF + LogReg", f3(tf["dev"]["macro_f1"]), f3(tf["dev"]["accuracy"])],
                         ["BERTurk (3 seeds)", BT_F1, BT_ACC]], [2.6, 1.6, 1.6], size=14, hl_rows=(3,), row_h=0.45)
box(s, 6.9, 3.75, 5.8, 3.1, WARN_T)
text(s, 7.15, 3.95, 5.3, 0.3, "FIRST FINDING", size=11, bold=True, color=WARN)
text(s, 7.15, 4.3, 5.3, 2.5, [
    [("BERTurk is below TF-IDF on dev and unstable", {"bold": True})],
    f"Per seed: {BT_SEEDS}.",
    "Training loss still 1.1–1.6 after 64 steps → under-trained, not a ceiling effect.",
    "Next: 10–20 epochs, lr 5e-5, epochs chosen on dev, on GPU. Test sets untouched until Dec 9.",
], size=14, spacing=6)
footer(s, 6)

# ── 7 plan
s = prs.slides.add_slide(BLANK)
header(s, "Plan to the final", "Remaining experiments, owners, dates")
table(s, 0.6, 1.6, 7.3, [
    ["Step", "Owner", "Date"],
    ["Collect real announcements — done: 818 from 24 GTÜ depts", "Helin S.", "Nov 11 ✓"],
    ["Label 150 GTÜ items (+ “other”) → Test C, κ on 50, freeze", "Helin + Nisanur", "Nov 27"],
    ["BERTurk, 3 seeds on GPU, epochs chosen on dev", "Cevahir A.", "Nov 24"],
    ["Ablation: XLM-R vs BERTurk; title removed", "Cevahir A.", "Dec 4"],
    ["Fix extraction errors X1–X3; hybrid NER", "Nisanur K.", "Dec 4"],
    ["QA on 50 real student questions", "Emre K.", "Dec 7"],
    ["One test run of all systems + paired bootstrap", "Cevahir A.", "Dec 9"],
    ["Error analysis of the method", "Nisanur K.", "Dec 11"],
    ["Final report draft (paper) + demo", "Emre K. (all)", "Dec 18"],
], [4.6, 1.6, 1.1], size=12.5, hl_rows=(1,), row_h=0.47)
card(s, 8.3, 1.6, 4.4, 1.75, "Protocol — fixed before method results",
     "Macro-F1 · frozen tests A, B (+C) · 3 seeds · bootstrap CI · real gain = Δ ≥ 0.03 and CI excludes 0",
     size=13.5)
card(s, 8.3, 3.55, 4.4, 2.15, "Biggest risk → fallback",
     ["Median confidence 0.89 on Test A vs 0.37 on GTÜ; many items fit none of the 6 classes.",
      "Fallback: label 150, add an “other” class."], fill=WARN_T, label_color=WARN, size=13.5)
card(s, 8.3, 5.9, 4.4, 0.95, "Proposal feedback", "RQ made testable; baselines first.", size=13.5)
footer(s, 7)

for i, sl in enumerate(prs.slides, 1):
    if i in NOTES:
        sl.notes_slide.notes_text_frame.text = NOTES[i]

# ── appendix: Project Card
CARD = [
    ("Fields 1–3", [
        ("1 · Title and abstract", [
            [("Working title: ", {"bold": True}), ("Do We Need BERTurk to Sort Campus Announcements? Linear Baselines vs. "
                                                   "Fine-tuned Encoders on Out-of-Template Turkish Notices", {})],
            [("Draft abstract: ", {"bold": True}),
             (f"University announcements arrive through websites, e-mail and club channels, and students miss exam, room "
              f"and deadline changes. We build a four-stage assistant (classification, extraction, summarisation, cited QA) "
              f"for Turkish and English announcements and ask whether a fine-tuned Turkish encoder is needed for the first "
              f"stage. On 420 labelled synthetic announcements a TF-IDF + logistic-regression baseline reaches "
              f"{f3(tf['test']['macro_f1'])} Macro-F1, but only {f3(tf['challenge']['macro_f1'])} on 24 hand-written "
              f"announcements; most baseline errors confuse the topic of an announcement with the action it asks for. "
              f"Fine-tuned BERTurk reaches ", {}), ("[GAP: Test B / C Macro-F1 ± CI]", {"bold": True, "color": WARN}),
             (", which ", {}), ("[GAP: does / does not]", {"bold": True, "color": WARN}),
             (" exceed our 0.03 threshold.", {})]], True),
        ("2 · Research question", [
            "Does fine-tuning BERTurk classify announcements more accurately (Macro-F1) than TF-IDF + LogReg on "
            "out-of-template announcements?",
            [("No: ", {"bold": True}), ("Δ < 0.03 or paired-bootstrap CI includes 0.", {})],
            [("Changed: ", {"bold": True}), ("the proposal's “how” question had no falsifiable outcome and the in-template "
                                             "test is saturated; the system is unchanged.", {})]], False),
        ("3 · Contributions (testable)", [
            "C1. On Test B/C BERTurk beats TF-IDF + LogReg by ≥ 0.03 Macro-F1 (refuted if CI includes 0).",
            "C2. The gain comes from topic-vs-action errors: BERTurk removes ≥ 50% of E1/E3/E5.",
            "C3. BERTurk ≥ XLM-R on English items (refuted if XLM-R is ≥ 0.03 better)."], False),
    ]),
    ("Fields 4–6", [
        ("4 · Related work", [
            "Devlin et al. (2019) BERT. NAACL — we fine-tune on a small 6-class domain task.",
            "Schweter (2020) BERTurk. Zenodo 10.5281/zenodo.3770924 — our method model, tested on short TR/EN notices.",
            "Conneau et al. (2020) XLM-R. ACL — our ablation for English announcements.",
            "Wang & Manning (2012) Baselines and Bigrams. ACL — strong linear baseline; we add char n-grams.",
            "Joulin et al. (2017) Bag of Tricks. EACL — linear models competitive; tested on low-resource Turkish.",
            "Lewis et al. (2020) RAG. NeurIPS — our QA retrieves structured fields and cites sources.",
            "Dror et al. (2018) Hitchhiker's Guide to Significance Testing. ACL — our paired bootstrap rule.",
            "Dodge et al. (2020) Fine-Tuning Pretrained LMs. arXiv:2002.06305 — why we report 3 seeds."], True),
        ("5 · Data", [
            "Synthetic generator (seed 42) + 24 hand-written (CC BY 4.0) + 818 raw GTÜ announcements, 24 departments "
            "(scripts/collect_gtu.py; © GTÜ → URLs + labels only).",
            f"420 → clean 420 → dedup 420 → train {SP['train']} / dev {SP['dev']} / test A {SP['test']}; Test B 24.",
            "GTÜ: 517 pages, 259 PDF titles, 42 links; 0 duplicates → 150 to label as Test C.",
            "Guideline: docs/annotation_guidelines.md. Agreement: not measured yet (κ on 50, Nov 27)."], False),
        ("6 · Method and baseline", [
            "Baseline: TF-IDF (word 1–2 + char 3–5 grams) + LogReg (C = 5); also rules, NB, SVM, majority.",
            f"Method: BERTurk fine-tuned, 4 epochs, lr 3e-5; dev Macro-F1 {BT_F1} (3 seeds).",
            "One change: representation + classifier; data, input, split, metric identical.",
            "Compute: baselines < 10 s CPU; BERTurk ~6 min/seed CPU; need ≈ 2 GPU-hours."], False),
    ]),
    ("Fields 7–9", [
        ("7 · Evaluation protocol (fixed Nov 11)", [
            "Metric: Macro-F1 (primary), accuracy; extraction micro P/R/F1.",
            "Frozen: A = 84 (sha256 2cafd21d7400), B = 24, C = 150 GTÜ (Nov 27). One test run per system (Dec 9).",
            "Spread: 3 seeds (13, 42, 2026) for neural models; 95% bootstrap CI (1,000, seed 0).",
            "Real difference: Δ ≥ 0.03 and paired-bootstrap CI excludes 0."], False),
        ("8 · Results so far", [
            *[f"{n}: A {f3(B[k]['test']['macro_f1'])} {ci(B[k]['test'])} · B {f3(B[k]['challenge']['macro_f1'])} "
              f"{ci(B[k]['challenge'])} · C —" for k, n in names],
            "BERTurk (3 seeds) · XLM-R · BERTurk no-title: Dec 9"], False),
        (f"9 · Error analysis — {N_CLS} classification + {N_EX} extraction errors", [
            " · ".join(f"{k} {n}: {CNT[k]}" for k, n in cats[:4]),
            " · ".join(f"{k} {n}: {CNT[k]}" for k, n in cats[4:]),
            f"X1–3 extraction (year-less date range / free-form place / dotted time): {CNT['X1']} / {CNT['X2']} / {CNT['X3']}",
            "Example E1: “… sınavı Kütüphane Seminer Salonu yerine Kültür Merkezi salonunda” room_change → exam. "
            "One example per category: reports/midproject_errors.csv"], True),
    ]),
    ("Fields 10–12", [
        ("10 · Research lead section — Nisanur Kara", [
            "Uludoğan et al. (2024) TURNA: A Turkish Encoder-Decoder Language Model. Findings of ACL — general "
            "benchmarks; we test a tiny, noisy, mixed-language domain where a linear model is near its ceiling.",
            "Edwards & Camacho-Collados (2024) Language Models for Text Classification: Is In-Context Learning "
            "Enough? LREC-COLING — we ask whether fine-tuning beats a linear model at all, in Turkish.",
            "Ablation: XLM-R vs BERTurk; BERTurk without title; learning curve 25/50/100%."], True),
        ("11 · Plan and risk", [
            "Real data done Nov 11 · Test C Nov 27 (Helin + Nisanur) · BERTurk GPU Nov 24, ablation Dec 4, test run "
            "Dec 9 (Cevahir) · extraction Dec 4, method errors Dec 11 (Nisanur) · QA Dec 7, report Dec 18 (Emre).",
            "Risk: GTÜ text far from labels (confidence 0.37 vs 0.89). Fallback: 150 items + “other” class.",
            "Feedback: RQ made falsifiable; baselines first."], False),
        ("12 · Reproducibility", [
            f"Repository: {REPO}",
            "One command: python evaluation/midproject.py (+ --with-berturk; error_analysis.py for field 9).",
            "Seeds 42 / 2026 / 0 / 13,42,2026 · Python 3.12.6, scikit-learn 1.9.1, torch 2.15.0, transformers 5.22.2.",
            "Roles: Helin data · Nisanur research lead, extraction · Cevahir classifiers, evaluation · Emre QA, demo.",
            "AI tools: Claude (Claude Code) for scripts and slide drafts; numbers produced by the scripts."], False),
    ]),
]
for n, (title, fields) in enumerate(CARD, 1):
    s = prs.slides.add_slide(BLANK)
    header(s, f"Appendix · Project Card {n}/4", title)
    rows_ = []  # each wide field is its own row; consecutive small fields share a row
    for f_ in fields:
        if f_[2] or not rows_ or rows_[-1][0][2]:
            rows_.append([f_])
        else:
            rows_[-1].append(f_)
    wide_h = {id(r): (2.45 if len(r[0][1]) > 4 else 2.0) for r in rows_ if r[0][2]}
    n_small = sum(1 for r in rows_ if not r[0][2])
    small_h = (6.85 - 1.5 - sum(wide_h.values()) - 0.2 * (len(rows_) - 1)) / max(1, n_small)
    y = 1.5
    for r in rows_:
        h = wide_h.get(id(r), small_h)
        w = (12.1 - 0.25 * (len(r) - 1)) / len(r)
        for i, (lab, body, _) in enumerate(r):
            card(s, 0.6 + i * (w + 0.25), y, w, h, lab, body, size=11.5)
        y += h + 0.2
    footer(s, f"A{n}", "Project Card (graded, not presented)")

out = OUT / f"{TEAM.replace('#', 'X')}_MidProject.pptx"
prs.save(out)
print("saved", out)
