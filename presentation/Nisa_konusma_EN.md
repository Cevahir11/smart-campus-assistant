# Speaker Script — Nisanur Kara (≈ 6–7 min)

## Slide 1 — Research question
"Hello, we are building a Smart Campus Announcement Assistant. It classifies university announcements, extracts
dates, times, places and course codes, and answers students' questions with sources.

Our question: **does fine-tuning BERTurk classify announcements better than a TF-IDF + logistic-regression baseline,
on text written outside our templates?**
The answer is 'no' if the gain is below 0.03 Macro-F1 or its confidence interval includes zero.
We narrowed the proposal's 'how' question because it could not be falsified."

## Slide 2 — Data
"We have 420 labelled announcements in 6 classes, split 252 / 84 / 84; the test set is frozen.
These are synthetic, so we also wrote 24 real-style announcements as Test B.
And we collected **818 real announcements from 24 GTÜ departments** — not labelled yet."

## Slide 3 — Baseline
"TF-IDF + logistic regression scores **1.000** on test A — the synthetic test is saturated.
On hand-written Test B it drops to **0.874**. We report 95% bootstrap intervals; Test B's is wide because it has only
24 items."

## Slide 4 — Error analysis
"We grouped **65 baseline errors** into 7 categories. The biggest: when an exam's room changes, the model says
'exam' instead of 'room change' — 21 errors. Next, no keyword fires at all — 16 errors."

## Slide 5 — What the errors suggest
"The label is the action, not the topic. Bag-of-words models cannot read 'room A instead of room B'; a contextual
model like BERTurk can. Unseen words are the second problem. So we test BERTurk on out-of-template data."

## Slide 6 — Method status
"BERTurk is implemented and has been run. The only change is the model; data and metric are the same.
First dev result: **0.807 ± 0.134** — below TF-IDF and unstable across seeds, because it trained for only 64
steps. Next we train longer on a GPU."

## Slide 7 — Plan
"Every step has an owner and a date. By November 27 we label 150 GTÜ announcements as Test C. The ablation is
BERTurk vs XLM-R. On December 9 we run each system on the test sets once.
**Biggest risk:** real GTÜ text is far from our labels — median confidence drops from 0.89 to 0.37.
**Fallback:** label 150, add an 'other' class.
Thank you — questions are welcome."
