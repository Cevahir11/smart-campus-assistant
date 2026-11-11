"""Build the mid-project slides + Project Card (HTML -> PDF with headless Edge/Chrome).

    python evaluation/midproject.py --with-berturk && python evaluation/error_analysis.py
    python presentation/build_slides.py            # -> presentation/TeamXX_MidProject.pdf
All numbers are read from reports/midproject_results.json and reports/midproject_errors.csv.
"""
import csv
import json
import statistics
import subprocess
import sys
from collections import Counter
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
TEAM = "Team##"  # <- put your team number here
MEMBERS = ["Helin Saygılı", "Nisanur Kara", "Cevahir Atıcı", "Emre Kılıç"]
LEAD = "Nisanur Kara"  # research lead (field 10)
REPO = "https://github.com/&lt;team&gt;/smart-campus-assistant"

R = json.loads((ROOT / "reports" / "midproject_results.json").read_text(encoding="utf-8"))
ERR = list(csv.DictReader(open(ROOT / "reports" / "midproject_errors.csv", encoding="utf-8")))
B = R["baselines"]
CNT = Counter(e["category"] for e in ERR)
N_CLS_ERR = sum(v for k, v in CNT.items() if k.startswith("E"))
N_EX_ERR = sum(v for k, v in CNT.items() if k.startswith("X"))

bt = R.get("berturk_dev", [])
if bt:
    f1s = [s["macro_f1"] for s in bt]
    accs = [s["accuracy"] for s in bt]
    BT_F1 = f"{statistics.mean(f1s):.3f} ± {statistics.pstdev(f1s):.3f}"
    BT_ACC = f"{statistics.mean(accs):.3f} ± {statistics.pstdev(accs):.3f}"
    BT_MIN = f"{statistics.mean(s['minutes'] for s in bt):.1f}"
    BT_SEEDS = ", ".join(str(s["seed"]) for s in bt)
    BT_ERR = sum(len(s["errors"]) for s in bt)
else:
    BT_F1 = BT_ACC = BT_MIN = "not run yet"
    BT_SEEDS, BT_ERR = "13, 42, 2026", 0


def f(x):
    return f"{x:.3f}"


def ci(r):
    return f"[{r['f1_ci'][0]:.2f}, {r['f1_ci'][1]:.2f}]"


NAMES = {"majority": "Majority class", "keyword_rules": "Keyword rules", "tfidf_nb": "TF-IDF + NB",
         "tfidf_svm": "TF-IDF + linear SVM", "tfidf_logreg": "TF-IDF + LogReg (main baseline)"}
SHORT = {**NAMES, "tfidf_logreg": "TF-IDF + LogReg (main)"}


def results_rows(include_method=False):
    rows = []
    for k, n in NAMES.items():
        r = B[k]
        cls = ' class="hl"' if k == "tfidf_logreg" else ""
        rows.append(f"<tr{cls}><td>{n}</td><td>{f(r['dev']['macro_f1'])}</td>"
                    f"<td>{f(r['test']['macro_f1'])} <span class=ci>{ci(r['test'])}</span></td>"
                    f"<td>{f(r['test']['accuracy'])}</td>"
                    f"<td>{f(r['challenge']['macro_f1'])} <span class=ci>{ci(r['challenge'])}</span></td>"
                    f"<td>{f(r['challenge']['accuracy'])}</td></tr>")
    if include_method:
        rows.append(f"<tr class=method><td>BERTurk fine-tuned (method)</td><td>{BT_F1}</td>"
                    "<td class=gap colspan=2>run once at the end (Dec 9)</td>"
                    "<td class=gap colspan=2>run once at the end (Dec 9)</td></tr>")
        rows.append("<tr class=method><td>XLM-R base (ablation)</td><td class=gap>–</td>"
                    "<td class=gap colspan=2>–</td><td class=gap colspan=2>–</td></tr>")
    return "\n".join(rows)


def example(cat, n=120):
    e = next(x for x in ERR if x["category"] == cat)
    t = e["text"] if len(e["text"]) < n else e["text"][:n - 3] + "…"
    return f"<i>“{escape(t)}”</i> <b>{e['gold']} → {e['pred']}</b>"


CATS = [
    ("E1", "Topic beats action", "exam room change → exam"),
    ("E4", "No keyword fires", "falls back to a default class"),
    ("E2", "Shared cue “başvuru”", "registration → scholarship"),
    ("E3", "Registration about an exam", "“exam applications” → exam"),
    ("E5", "Event mentions exams", "“sınav haftası öncesi …” → exam"),
    ("E6", "Room name = event cue", "“Seminer Salonu” → event"),
    ("E7", "TF-IDF: unseen wording", "hand-written set only"),
]


def err_rows(examples=True):
    rows = []
    for k, name, d in CATS:
        ex = f"<td class=ex>{example(k)}</td>" if examples else ""
        rows.append(f"<tr><td><b>{k}</b></td><td>{name}<br><span class=muted>{d}</span></td>"
                    f"<td class=num>{CNT[k]}</td>{ex}</tr>")
    return "\n".join(rows)


tf = B["tfidf_logreg"]
kw = B["keyword_rules"]
SP = R["split"]

CSS = """
@page { size: 1280px 720px; margin: 0 }
* { box-sizing: border-box }
body { margin: 0; font-family: 'Segoe UI', Calibri, Arial, sans-serif; color: #1c2333; background: #fff }
.slide { width: 1280px; height: 720px; padding: 44px 60px 36px; position: relative; page-break-after: always;
         overflow: hidden; display: flex; flex-direction: column }
.slide:last-child { page-break-after: auto }
h1 { font-size: 34px; margin: 0 0 18px; color: #16264a; letter-spacing: -.3px }
h1 small { font-size: 17px; color: #0f766e; font-weight: 600; display: block; letter-spacing: 1.2px;
           text-transform: uppercase; margin-bottom: 4px }
.foot { position: absolute; bottom: 16px; left: 60px; right: 60px; font-size: 12px; color: #8a93a6;
        display: flex; justify-content: space-between; border-top: 1px solid #e6e9ef; padding-top: 6px }
p, li { font-size: 19px; line-height: 1.38; margin: 0 0 8px }
ul { margin: 0; padding-left: 22px }
.cols { display: flex; gap: 32px; flex: 1; min-height: 0 }
.cols > div { flex: 1; min-width: 0 }
.box { border-radius: 10px; padding: 14px 18px; margin-bottom: 14px; background: #f3f6fa }
.box.q { background: #eaf4f2; border-left: 6px solid #0f766e }
.box.no { background: #fdf0ef; border-left: 6px solid #c2410c }
.box.chg { background: #f6f3ea; border-left: 6px solid #a16207 }
.box h3 { margin: 0 0 6px; font-size: 15px; text-transform: uppercase; letter-spacing: 1px; color: #475069 }
.box p { margin: 0 }
table { border-collapse: collapse; width: 100%; font-size: 16.5px }
th { text-align: left; background: #16264a; color: #fff; padding: 7px 10px; font-weight: 600 }
td { padding: 6px 10px; border-bottom: 1px solid #e3e7ee; vertical-align: top }
tr.hl td { background: #eaf4f2; font-weight: 600 }
tr.method td { background: #fbfbfd; color: #39415a }
td.gap { color: #a0a7b8; font-style: italic }
.ci { color: #6b7489; font-size: 14px; font-weight: 400 }
.num { text-align: right; font-weight: 700; font-size: 19px; color: #16264a }
.ex { font-size: 13.5px; color: #39415a }
.muted { color: #6b7489; font-size: 14px }
.big { font-size: 46px; font-weight: 700; color: #0f766e; line-height: 1 }
.kpi { display: flex; gap: 18px; margin-bottom: 14px }
.kpi div { flex: 1; background: #f3f6fa; border-radius: 10px; padding: 12px 16px }
.kpi span { display: block; font-size: 14px; color: #5b6479; margin-top: 4px }
code { font-family: Consolas, monospace; font-size: .9em; background: #eef1f6; padding: 1px 5px; border-radius: 4px }
.mono { font-family: Consolas, monospace; font-size: 13.5px; background: #f3f6fa; border-radius: 8px;
        padding: 10px 14px; white-space: pre-wrap; line-height: 1.4 }
/* project card */
.card h1 { font-size: 26px; margin-bottom: 10px }
.card .fields { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 22px; flex: 1; min-height: 0; align-content: start }
.card .fld { background: #f6f8fb; border-radius: 8px; padding: 9px 13px; overflow: hidden }
.card .fld.wide { grid-column: 1 / -1 }
.card .fld h4 { margin: 0 0 4px; font-size: 13.5px; color: #0f766e; text-transform: uppercase; letter-spacing: .6px }
.card .fld p, .card .fld li { font-size: 12.6px; line-height: 1.36; margin: 0 0 3px }
.card table { font-size: 12px } .card th { padding: 3px 7px } .card td { padding: 2px 7px } .card .ex, .card td i { font-size: 11.5px }
.card .gapmark { background: #fff3c4; padding: 0 4px; border-radius: 3px; font-weight: 600 }
"""


def foot(n, label="Mid-project · CSE 484/654"):
    return f'<div class=foot><span>{TEAM} · Smart Campus Announcement Assistant</span><span>{label}</span><span>{n}</span></div>'


slides = []

# 1 ── research question
slides.append(f"""
<section class=slide>
<h1><small>{TEAM} · {' · '.join(MEMBERS)}</small>Smart Campus Announcement Assistant</h1>
<div class="box q"><h3>Research question</h3>
<p style="font-size:24px;line-height:1.35"><b>Does fine-tuning BERTurk classify campus announcements more accurately
(Macro-F1) than a TF-IDF + logistic-regression baseline, on announcements written outside our generation templates?</b></p></div>
<div class="box no"><h3>What a “no” looks like</h3>
<p>BERTurk gains <b>&lt; 0.03 Macro-F1</b>, or the paired-bootstrap 95% CI of the gain includes 0 → a linear model is
enough for stage 01, and the project's effort should go to extraction and QA.</p></div>
<div class="box chg"><h3>Changed since the proposal — and why</h3>
<p><b>Old:</b> “How can NLP techniques turn scattered university announcements into structured, searchable information
and answer students' questions?”<br><b>Why:</b> an open “how” question has no “no” result, and on our template-based
test set TF-IDF already scores <b>{f(tf['test']['macro_f1'])}</b>. We kept the system and narrowed the question to the
one comparison that is still open: out-of-template text.</p></div>
{foot(1)}</section>""")

# 2 ── data
slides.append(f"""
<section class=slide>
<h1><small>Data as loaded</small>420 labelled announcements + 24 hand-written, test set frozen</h1>
<div class=cols>
<div>
<table>
<tr><th>Set</th><th>Source</th><th>N</th><th>TR / EN</th></tr>
<tr><td>Train</td><td rowspan=3>Synthetic “campus world”<br><span class=muted>scripts/generate_dataset.py, seed 42</span></td><td class=num>{SP['train']}</td><td>198 / 54</td></tr>
<tr><td>Dev</td><td class=num>{SP['dev']}</td><td>66 / 18</td></tr>
<tr class=hl><td>Test A (frozen)</td><td class=num>{SP['test']}</td><td>71 / 13</td></tr>
<tr class=hl><td>Test B (frozen)</td><td>Hand-written, real-style, never trained on</td><td class=num>{SP['challenge']}</td><td>19 / 5</td></tr>
<tr><td>GTÜ raw (unlabelled)</td><td>gtu.edu.tr, 24 departments, collected Nov 11<br><span class=muted>scripts/collect_gtu.py</span></td><td class=num>818</td><td>≈ all TR</td></tr>
</table>
<p style="margin-top:12px;font-size:16px">6 classes: exam 92 · event 76 · registration 71 · cancellation 63 · room change 59 ·
scholarship 59. Split: stratified 60/20/20, seed 2026, ids in <code>data/splits/midproject_split.json</code>
(sha256 <code>2cafd21d7400</code>). Preprocessing: 420 → clean_text 420 → dedup 420 (0 removed).
1,292 entity spans (DATE/TIME/LOCATION/COURSE).</p>
<p style="font-size:16px"><b>Licence:</b> all text is team-written (generator + hand-written set) → released CC BY 4.0;
GTÜ announcements are public pages © GTÜ: we release only URLs + our labels, not the text.</p>
</div>
<div style="flex:.85">
<div class=box><h3>One real example (ann-0001, train)</h3>
<p style="font-size:16.5px"><b>Title:</b> KIM107 dersi iptal<br><b>Text:</b> “1 Kasım 2026 tarihinde yapılması planlanan
KIM107 Genel Kimya dersi, öğretim üyesinin rahatsızlığı nedeniyle yapılmayacaktır.”</p>
<p style="font-size:15px;margin-top:8px"><b>category</b> = cancellation<br>
<b>entities</b> = DATE “1 Kasım 2026” → 2026-11-01 · COURSE “KIM107” → KIM 107<br>
<b>summary</b> = “KIM107 dersi 1 Kasım tarihinde yapılmayacak.”</p></div>
<div class="box no"><h3>Honest caveat</h3><p style="font-size:16px">Train/dev/test A are synthetic, so they
share vocabulary with our rules and templates. That is why Test B exists and why a real test set (Test C)
is the biggest item on the plan.</p></div>
</div></div>
{foot(2)}</section>""")

# 3 ── baseline results
slides.append(f"""
<section class=slide>
<h1><small>Baseline result</small>Test A is saturated; out-of-template text is where the gap is</h1>
<table>
<tr><th>System (trained on train, 252)</th><th>Dev Macro-F1</th><th>Test A Macro-F1 [95% CI]</th><th>Test A Acc</th>
<th>Test B Macro-F1 [95% CI]</th><th>Test B Acc</th></tr>
{results_rows()}
</table>
<div class=cols style="margin-top:18px">
<div><div class=kpi>
<div><div class=big>{f(tf['challenge']['macro_f1'])}</div><span>TF-IDF + LogReg on hand-written Test B (n = 24)</span></div>
<div><div class=big>{f(tf['test']['macro_f1'] - tf['challenge']['macro_f1'])}</div><span>drop from Test A to Test B</span></div>
</div></div>
<div><p style="font-size:16.5px">Spread: all baselines are deterministic, so the interval is a <b>95% bootstrap CI</b>
(1,000 resamples of the test set, seed 0). Test B's CI is ~0.3 wide: with 24 items we cannot resolve a 0.03
difference → we need a larger real test set.<br>Reproduce: <code>python evaluation/midproject.py</code></p></div>
</div>
{foot(3)}</section>""")

# 4 ── error analysis
slides.append(f"""
<section class=slide>
<h1><small>Error analysis</small>{N_CLS_ERR} baseline errors in 7 categories</h1>
<table>
<tr><th style="width:44px">#</th><th style="width:300px">Category</th><th style="width:60px">N</th><th>Example (gold → predicted)</th></tr>
{err_rows()}
</table>
<p class=muted style="margin-top:8px">Keyword rules: 45 train + 12 dev + 5 Test B errors (rules are not trained, so train errors are fair);
TF-IDF: 3 Test B errors. Test A errors were not inspected (frozen). File: <code>reports/midproject_errors.csv</code></p>
{foot(4)}</section>""")

# 5 ── what the errors suggest
slides.append(f"""
<section class=slide>
<h1><small>Error analysis</small>What the errors tell us</h1>
<div class=cols>
<div>
<div class=box><h3>1 · The label is the action, not the topic (E1, E3, E5 = {CNT['E1'] + CNT['E3'] + CNT['E5']})</h3>
<p style="font-size:17px">“MAT 102 sınavı <u>Konferans Salonu yerine E-205</u>'te” is a room change, but every
cue word says exam. Bag-of-words counts cannot weigh “X yerine Y” against “sınav”. A contextual encoder can — this is
the strongest argument for BERTurk.</p></div>
<div class=box><h3>2 · Vocabulary gaps (E4, E7 = {CNT['E4'] + CNT['E7']})</h3>
<p style="font-size:17px">“Don't miss…”, “no lecture”, “ekle-sil”, “guest lecture”, “tuition waiver”, “ilişik kesme”:
the hand-written set uses words that never appear in training. TF-IDF fails here; pretrained subword knowledge may not.</p></div>
</div>
<div>
<div class=box><h3>3 · Ambiguous shared cues (E2, E6 = {CNT['E2'] + CNT['E6']})</h3>
<p style="font-size:17px">“başvuru” is both scholarship and registration; “Seminer Salonu” is a room, not a seminar.
TF-IDF learns these from data (0 errors on Test A) — so these are a rules problem, not a learning problem.</p></div>
<div class=box><h3>4 · Extraction (stage 02) on Test B: {N_EX_ERR} errors</h3>
<p style="font-size:17px">Date ranges without a year (“15-17 Mayıs” → wrong year, {CNT['X1']}), free-form places
(“Fen Fakültesi amfileri”, {CNT['X2']}), dotted time “17.30'da” ({CNT['X3']}). Micro-F1 is still 0.947.</p></div>
<div class="box q"><h3>So</h3><p style="font-size:17px">Expect BERTurk to help on categories 1 and 2 only, which exist
mainly outside the templates → measure on Test B and a real Test C, not on Test A.</p></div>
</div></div>
{foot(5)}</section>""")

# 6 ── method status
slides.append(f"""
<section class=slide>
<h1><small>Method status</small>BERTurk fine-tuning implemented and run on CPU</h1>
<div class=cols>
<div>
<div class=box><h3>The one controlled change</h3>
<p style="font-size:17.5px">Replace <b>TF-IDF features + LogReg</b> with a <b>fine-tuned BERTurk encoder</b>
(<code>dbmdz/bert-base-turkish-cased</code>). Same split, same input (title + text), same labels, same metric.</p></div>
<div class=box><h3>As implemented</h3>
<p style="font-size:16.5px">4 epochs · lr 3e-5 · batch 16 · max length 128 · weight decay 0.01 · seeds {BT_SEEDS} ·
<code>python evaluation/midproject.py --with-berturk</code><br>Compute: {BT_MIN} min per seed on a laptop CPU;
final runs (BERTurk + XLM-R, 3 seeds, larger train) ≈ 2 GPU-hours on Colab T4.</p></div>
</div>
<div>
<table>
<tr><th>Dev set (n = 84)</th><th>Macro-F1</th><th>Accuracy</th></tr>
<tr><td>Keyword rules</td><td>{f(kw['dev']['macro_f1'])}</td><td>{f(kw['dev']['accuracy'])}</td></tr>
<tr><td>TF-IDF + LogReg</td><td>{f(tf['dev']['macro_f1'])}</td><td>{f(tf['dev']['accuracy'])}</td></tr>
<tr class=hl><td>BERTurk (mean ± std, 3 seeds)</td><td>{BT_F1}</td><td>{BT_ACC}</td></tr>
</table>
<p style="font-size:16px;margin-top:12px"><b>First finding:</b> BERTurk is <b>below</b> TF-IDF on dev and unstable across seeds
(per seed: {", ".join(f"{s['macro_f1']:.3f}" for s in bt)}). Training loss is still 1.1–1.6 after 64 steps → under-trained, not
a ceiling effect. Next: 10–20 epochs, lr 5e-5, epochs chosen on dev, on GPU. Test sets untouched until Dec 9.</p>
<div class=box><h3>Rest of the pipeline (already working)</h3>
<p style="font-size:15.5px">Extraction rules: micro-F1 0.947 on Test B · Summaries: structured template ROUGE-L 0.953 vs lead-1 0.432 ·
QA (50 questions, end-to-end with predicted labels): 50/50 answers correct · Flask web app + CLI.</p></div>
</div></div>
{foot(6)}</section>""")

# 7 ── protocol + plan
slides.append(f"""
<section class=slide>
<h1><small>Plan to the final</small>Remaining experiments, owners, dates</h1>
<div class=cols>
<div style="flex:1.45">
<table style="font-size:15px">
<tr><th>Step</th><th>Owner</th><th>Date</th></tr>
<tr><td><s>Collect ≥ 300 real announcements</s> <b>done: 818 from 24 GTÜ departments</b></td><td>Helin S.</td><td>Nov 11 ✓</td></tr>
<tr><td>Double-annotate 150 GTÜ items (+ “other” class) → <b>Test C</b>, κ on 50, freeze</td><td>Helin S. + Nisanur K.</td><td>Nov 27</td></tr>
<tr><td>BERTurk, 3 seeds on GPU, choose epochs on dev</td><td>Cevahir A.</td><td>Nov 24</td></tr>
<tr><td>Ablation: XLM-R vs BERTurk; title removed</td><td>Cevahir A.</td><td>Dec 4</td></tr>
<tr><td>Fix extraction errors X1–X3; hybrid NER on real data</td><td>Nisanur K.</td><td>Dec 4</td></tr>
<tr><td>QA on 50 real student questions</td><td>Emre K.</td><td>Dec 7</td></tr>
<tr><td><b>One</b> test run of all systems on A/B/C + paired bootstrap</td><td>Cevahir A.</td><td>Dec 9</td></tr>
<tr><td>Error analysis of the method</td><td>Nisanur K.</td><td>Dec 11</td></tr>
<tr><td>Final report draft (paper format) + demo</td><td>Emre K. (all)</td><td>Dec 18</td></tr>
</table>
</div>
<div>
<div class=box><h3>Protocol (fixed today, before method results)</h3>
<p style="font-size:15.5px">Macro-F1 (primary), accuracy · frozen tests A, B (+C on Nov 27) · 3 seeds for neural models ·
95% bootstrap CI · <b>a gain is real only if ΔMacro-F1 ≥ 0.03 and the paired-bootstrap CI excludes 0</b>.</p></div>
<div class="box no"><h3>Biggest risk → fallback</h3>
<p style="font-size:15.5px">Real GTÜ text is far from our labels: median classifier confidence 0.89 on Test A vs
<b>0.37 on the 818 GTÜ items</b>, and many are none of our 6 classes. Labelling may be slow / κ low.<br>
<b>Fallback:</b> label 150 instead of 300, add an “other” class, report the minimum detectable difference.</p></div>
<div class=box><h3>Proposal feedback</h3><p style="font-size:15.5px">RQ narrowed to a testable yes/no; baselines + error
analysis added before any neural model.</p></div>
</div></div>
{foot(7)}</section>""")

# ───────────────────────── Project Card (appendix, 4 slides)
abstract = ("University announcements arrive through websites, e-mail and club channels, and students miss exam, "
            "room and deadline changes. We build a four-stage assistant (classification, information extraction, "
            "summarisation, cited question answering) for Turkish and English announcements and ask whether a fine-tuned "
            f"Turkish encoder is needed for the first stage. On {SP['train'] + SP['dev'] + SP['test']} labelled synthetic "
            f"announcements a TF-IDF + logistic-regression baseline reaches {f(tf['test']['macro_f1'])} Macro-F1, but only "
            f"{f(tf['challenge']['macro_f1'])} on 24 hand-written announcements; most baseline errors confuse the topic of an "
            "announcement with the action it asks for. Fine-tuned BERTurk reaches <span class=gapmark>[GAP: Test B / Test C "
            "Macro-F1 ± CI]</span>, which <span class=gapmark>[GAP: does / does not]</span> exceed our 0.03 threshold. "
            "The extraction rules reach 0.947 micro-F1 and end-to-end QA answers 50/50 questions.")
abstract_words = len(abstract.replace("<span class=gapmark>", "").replace("</span>", "").split())

card = []
card.append(f"""
<section class="slide card">
<h1><small>Appendix A · Project Card 1/4</small>Fields 1–3</h1>
<div class=fields>
<div class="fld wide"><h4>1 · Title and abstract</h4>
<p><b>Working title:</b> Do We Need BERTurk to Sort Campus Announcements? Linear Baselines vs. Fine-tuned Encoders
on Out-of-Template Turkish Notices</p>
<p><b>Draft abstract ({abstract_words} words):</b> {abstract}</p></div>
<div class=fld><h4>2 · Research question</h4>
<p><b>RQ:</b> Does fine-tuning BERTurk classify campus announcements more accurately (Macro-F1) than TF-IDF + logistic
regression on announcements written outside our generation templates?</p>
<p><b>“No” result:</b> ΔMacro-F1 &lt; 0.03 or the paired-bootstrap 95% CI includes 0 → the linear model suffices.</p>
<p><b>Changed:</b> proposal asked “How can NLP techniques turn scattered announcements into structured, searchable
information and answer questions?” — no falsifiable outcome, and the in-template test is saturated
(TF-IDF {f(tf['test']['macro_f1'])}). The system is unchanged; the question is narrowed to stage 01 on out-of-template text.</p></div>
<div class=fld><h4>3 · Contributions (testable claims)</h4>
<ul>
<li><b>C1.</b> On out-of-template announcements (Test B/C), fine-tuned BERTurk beats TF-IDF + LogReg by ≥ 0.03 Macro-F1
(paired bootstrap). Refuted if the CI includes 0.</li>
<li><b>C2.</b> The gain comes from “topic-vs-action” errors (E1/E3/E5): BERTurk removes ≥ 50% of them while E2/E6
errors are already 0 for TF-IDF.</li>
<li><b>C3.</b> A Turkish-only encoder (BERTurk) is at least as good as multilingual XLM-R on the 20% English
announcements (ablation). Refuted if XLM-R is ≥ 0.03 better on English items.</li>
</ul></div>
</div>
{foot("A1", "Project Card (graded, not presented)")}</section>""")

card.append(f"""
<section class="slide card">
<h1><small>Appendix B · Project Card 2/4</small>Fields 4–6</h1>
<div class=fields>
<div class="fld wide"><h4>4 · Related work</h4>
<ul>
<li>Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. <i>NAACL-HLT</i>. — We fine-tune a BERT-style encoder on a small 6-class domain task, not on general benchmarks.</li>
<li>Schweter, S. (2020). BERTurk – BERT models for Turkish. Zenodo, doi:10.5281/zenodo.3770924. — Our method model; we test whether its gains hold on short, mixed TR/EN campus notices.</li>
<li>Conneau, A., Khandelwal, K., Goyal, N., et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. <i>ACL</i>. — XLM-R is our ablation for the English share of announcements.</li>
<li>Wang, S., Manning, C. D. (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification. <i>ACL</i>. — Motivates a strong linear baseline; we add character n-grams for Turkish suffixes.</li>
<li>Joulin, A., Grave, E., Bojanowski, P., Mikolov, T. (2017). Bag of Tricks for Efficient Text Classification. <i>EACL</i>. — Same claim that linear models are competitive; we test it on low-resource Turkish text.</li>
<li>Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. <i>NeurIPS</i>. — Our QA retrieves and filters structured fields and cites sources instead of training a generator.</li>
<li>Dror, R., Baumer, G., Shlomov, S., Reichart, R. (2018). The Hitchhiker's Guide to Testing Statistical Significance in NLP. <i>ACL</i>. — Basis for our paired-bootstrap decision rule.</li>
<li>Dodge, J., Ilharco, G., Schwartz, R., et al. (2020). Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping. arXiv:2002.06305. — Why we report 3 seeds for every neural model.</li>
</ul></div>
<div class=fld><h4>5 · Data</h4>
<p><b>Source:</b> synthetic “campus world” generator (<code>scripts/generate_dataset.py</code>, seed 42, 4 source types)
+ 24 hand-written real-style announcements (team-authored, CC BY 4.0) + <b>818 raw GTÜ announcements</b> from 24
departments (<code>scripts/collect_gtu.py</code>, public pages © GTÜ → we share URLs + labels only).</p>
<p><b>Counts:</b> generated 420 → clean_text 420 → dedup 420 → split train {SP['train']} / dev {SP['dev']} / test A
{SP['test']}; Test B 24. 335 TR / 85 EN; 1,292 entity spans. GTÜ: 818 collected (517 pages, 259 PDF titles, 42 links;
0 duplicates, 3 timeouts) → 150 to be labelled as Test C.</p>
<p><b>Example:</b> “1 Kasım 2026 tarihinde yapılması planlanan KIM107 Genel Kimya dersi … yapılmayacaktır.” → cancellation;
DATE 2026-11-01, COURSE KIM 107.</p>
<p><b>Guideline:</b> <code>docs/annotation_guidelines.md</code> (label = the student's required action). <b>Agreement:</b>
not measured yet — κ on a 50-item double-annotated sample of the real set, due Nov 27.</p></div>
<div class=fld><h4>6 · Method and baseline</h4>
<p><b>Baseline as run:</b> TF-IDF (word 1–2-grams + char 3–5-grams, folded Turkish) + logistic regression (C = 5);
also keyword rules, NB, SVM, majority.</p>
<p><b>Method as implemented:</b> BERTurk (<code>dbmdz/bert-base-turkish-cased</code>) fine-tuned, 4 epochs, lr 3e-5,
batch 16, max length 128; dev Macro-F1 {BT_F1} (3 seeds) vs 1.000 for TF-IDF — under-trained after 64 steps, so epochs will be tuned on dev.</p>
<p><b>The one change:</b> the text representation + classifier (sparse TF-IDF + LogReg → fine-tuned encoder); data,
input fields, split and metric are identical.</p>
<p><b>Compute:</b> baselines &lt; 10 s CPU; BERTurk {BT_MIN} min/seed on CPU. Needed: ≈ 2 GPU-hours (Colab T4) for
BERTurk + XLM-R × 3 seeds on the larger train set.</p></div>
</div>
{foot("A2", "Project Card (graded, not presented)")}</section>""")

card_res = "\n".join(
    f"<tr{' class=hl' if k == 'tfidf_logreg' else ''}><td>{SHORT[k]}</td><td>{f(B[k]['test']['macro_f1'])} {ci(B[k]['test'])}</td>"
    f"<td>{f(B[k]['challenge']['macro_f1'])} {ci(B[k]['challenge'])}</td><td class=gap>Dec 9</td></tr>"
    for k, n in NAMES.items())
card_err = "\n".join(f"<tr><td>{k}</td><td>{name}</td><td class=num style='font-size:13px'>{CNT[k]}</td>"
                     f"<td>{example(k, 95)}</td></tr>" for k, name, _ in CATS)
card.append(f"""
<section class="slide card">
<h1><small>Appendix C · Project Card 3/4</small>Fields 7–9</h1>
<div class=fields>
<div class=fld><h4>7 · Evaluation protocol (fixed Nov 11, before method results)</h4>
<p><b>Metric:</b> Macro-F1 over 6 classes (primary); accuracy (secondary). Extraction: micro P/R/F1 on (label, normalised value).</p>
<p><b>Frozen tests:</b> A = 84 synthetic (ids + sha256 2cafd21d7400 in <code>data/splits/</code>); B = 24 hand-written;
C = 150 real, frozen on Nov 27 before any model sees it. Each system is run on the tests once (Dec 9).</p>
<p><b>Spread:</b> 3 seeds (13, 42, 2026) for neural models, mean ± std; 95% bootstrap CI (1,000 resamples, seed 0) for all.</p>
<p><b>Smallest real difference:</b> ΔMacro-F1 ≥ 0.03 <i>and</i> paired-bootstrap 95% CI of Δ excludes 0.</p></div>
<div class=fld><h4>8 · Results so far (final results table)</h4>
<table><tr><th>System</th><th>Test A Macro-F1 [CI]</th><th>Test B Macro-F1 [CI]</th><th>Test C</th></tr>
{card_res}
<tr class=method><td>BERTurk (3 seeds)</td><td class=gap>Dec 9</td><td class=gap>Dec 9</td><td class=gap>Dec 9</td></tr>
<tr class=method><td>XLM-R base (ablation)</td><td class=gap>Dec 9</td><td class=gap>Dec 9</td><td class=gap>Dec 9</td></tr>
<tr class=method><td>BERTurk, no title (ablation)</td><td class=gap>Dec 9</td><td class=gap>Dec 9</td><td class=gap>Dec 9</td></tr>
</table>
<p style="margin-top:4px">Baselines are deterministic, so the CI comes from bootstrap; seed std applies to neural rows.</p></div>
<div class="fld wide"><h4>9 · Error analysis — {N_CLS_ERR} baseline classification errors (+ {N_EX_ERR} extraction errors)</h4>
<table><tr><th>#</th><th>Category</th><th>N</th><th>Example (gold → predicted)</th></tr>
{card_err}
<tr><td>X1–3</td><td>Extraction: year-less date range / free-form place / “17.30'da”</td><td class=num style='font-size:13px'>{CNT['X1']}/{CNT['X2']}/{CNT['X3']}</td><td><i>“konserler 15-17 Mayıs tarihlerinde Kampüs Açık Hava Tiyatrosu'nda”</i> → 2026 instead of 2027, place missed</td></tr>
</table></div>
</div>
{foot("A3", "Project Card (graded, not presented)")}</section>""")

card.append(f"""
<section class="slide card">
<h1><small>Appendix D · Project Card 4/4</small>Fields 10–12</h1>
<div class=fields>
<div class="fld wide"><h4>10 · Research lead section — {LEAD} (research lead)</h4>
<p><b>Uludoğan, G., Yirmibeşoğlu Balal, Z., Akkurt, F., Türker, M., Güngör, O., Üsküdarlı, S. (2024). TURNA: A Turkish
Encoder-Decoder Language Model for Enhanced Understanding and Generation. <i>Findings of ACL 2024</i>.</b> — Evaluates Turkish
models on general benchmarks and finds encoder models such as BERTurk strong on classification; we test whether that holds for
a tiny, noisy, mixed-language domain where a linear model is already near its ceiling.</p>
<p><b>Edwards, A., Camacho-Collados, J. (2024). Language Models for Text Classification: Is In-Context Learning Enough?
<i>LREC-COLING 2024</i>.</b> — Shows fine-tuned smaller models often beat prompted LLMs on classification (English); we ask the
prior question, whether fine-tuning beats a linear model at all, in Turkish, and can add a prompted Claude row as an extra comparison.</p>
<p><b>Ablation we will run:</b> (a) XLM-R base vs BERTurk with the identical recipe (published multilingual baseline);
(b) BERTurk without the title field; (c) learning curve with 25 / 50 / 100% of train. All on frozen tests A/B/C, 3 seeds.</p></div>
<div class=fld><h4>11 · Plan and risk</h4>
<p>Real data: done Nov 11 (818 GTÜ items) · Test C double annotation + κ, freeze — Helin S. + Nisanur K. Nov 27 · BERTurk GPU runs —
Cevahir A. Nov 24 · XLM-R / no-title ablation — Cevahir A. Dec 4 · extraction fixes + hybrid NER — Nisanur K. Dec 4 · QA on 50
real questions — Emre K. Dec 7 · single test run + paired bootstrap — Cevahir A. Dec 9 · method error analysis — Nisanur K.
Dec 11 · report draft — Emre K. (all) Dec 18.</p>
<p><b>Biggest risk:</b> real GTÜ text is far from our label set (median confidence 0.37 vs 0.89 on Test A; many items fit
none of the 6 classes), so labelling is slow or κ is low. <b>Fallback:</b> 150 labelled items, an added “other” class, and the
minimum detectable difference reported as a limitation.</p>
<p><b>Proposal feedback:</b> RQ made falsifiable with an explicit “no”; baselines and error analysis added first.</p></div>
<div class=fld><h4>12 · Reproducibility</h4>
<p><b>Repository:</b> {REPO}</p>
<p><b>One command (field 8):</b> <code>python evaluation/midproject.py</code> (add <code>--with-berturk</code> for the
method dev row; <code>python evaluation/error_analysis.py</code> for field 9).</p>
<p><b>Seeds:</b> data 42 · split 2026 · bootstrap 0 · neural 13 / 42 / 2026. <b>Versions:</b> Python 3.12.6, scikit-learn 1.9.1,
numpy 2.5.4, torch 2.15.0 (CPU), transformers 5.22.2.</p>
<p><b>Who did what:</b> Helin S. — data collection, guideline; Nisanur K. — research lead (related work, ablation design), extraction, error analysis; Cevahir A. — classifiers,
evaluation protocol, repository; Emre K. — summarisation, QA, web demo.</p>
<p><b>AI tools:</b> Claude (Anthropic) via Claude Code helped write evaluation scripts and draft slide text; every number was
produced by running the scripts above and checked by the team.</p></div>
</div>
{foot("A4", "Project Card (graded, not presented)")}</section>""")

html = f"<!doctype html><html><head><meta charset=utf-8><title>{TEAM} Mid-Project</title><style>{CSS}</style></head><body>" \
       + "".join(slides + card) + "</body></html>"
(OUT / "slides.html").write_text(html, encoding="utf-8")
print("abstract words:", abstract_words, "| slides:", len(slides), "+", len(card))

pdf = OUT / f"{TEAM.replace('#', 'X')}_MidProject.pdf"
browsers = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
exe = next((b for b in browsers if Path(b).exists()), None)
if exe:
    subprocess.run([exe, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    (OUT / "slides.html").as_uri()], check=True, timeout=120)
    print("PDF:", pdf)
else:
    print("open slides.html in a browser and print to PDF", file=sys.stderr)
