# Mid-Project Talk — Speaker Script (10 min, 7 slides)

About 9 minutes of speaking + 1 minute buffer. Suggested speakers are only a proposal — swap as you like.
Any member can be asked any question, so everyone should read the whole script.
Project Card pages (A1–A4) are graded, not presented.

---

## Slide 1 — Research question (≈ 1:15) · Cevahir

"Hello, we are building the Smart Campus Announcement Assistant. The system takes scattered university
announcements, classifies them, extracts the date, time, place and course code, writes a one-sentence summary,
and answers students' questions with cited sources.

Our research question is: **Does fine-tuning BERTurk classify campus announcements more accurately than a TF-IDF
plus logistic-regression baseline — measured with Macro-F1, on announcements written outside our generation
templates?**

We defined the 'no' result in advance: if BERTurk gains less than 0.03 Macro-F1, or the paired-bootstrap
confidence interval of the gain includes zero, the answer is no. Then a simple linear model is enough for this
stage, and we move our effort to extraction and question answering.

The question changed since the proposal. The old one was: 'How can NLP techniques turn scattered announcements
into structured, searchable information?' That is a 'how' question — it has no 'no' result, so it cannot be
falsified. Also, on our template-based test set TF-IDF already scores 1.000. We kept the system and narrowed the
question to the one comparison that is still open."

---

## Slide 2 — Data (≈ 1:15) · Helin

"Here is our data as loaded. We have 420 labelled announcements in 6 classes — exam, event, scholarship,
cancellation, room change and registration — 80% Turkish and 20% English. We split them stratified by class,
60/20/20: **252 train, 84 dev, 84 test.** The split uses seed 2026, the id list is committed to the repository,
and its hash is on the slide — so the test set is frozen.

We have a second test set: **Test B, 24 announcements we wrote by hand in real-world style.** No model was trained
on them.

On the right is a real example: 'KIM107 dersi iptal'. It is labelled cancellation; we extract the date and the
course code, and there is a reference summary.

We want to be honest about one point: train, dev and test A are **synthetic** — produced by our own generator — so
they share vocabulary with our rules. That is exactly why Test B exists, and why a real test set, Test C, is the
biggest item in our plan. And real data is no longer a promise: yesterday we collected **818 real announcements from 24
GTÜ departments** with our own collector; they are not labelled yet. Licence: our own text is CC BY 4.0; for the GTÜ
pages we will share only URLs and our labels."

---

## Slide 3 — Baseline results (≈ 1:30) · Cevahir

"All baselines are trained on the 252 training announcements. The majority class gets 0.059, keyword rules get
0.840 Macro-F1 on test A. **Our main baseline, TF-IDF plus logistic regression, gets 1.000 on test A.** So test A
is saturated — it cannot separate models.

The real finding is in the right-hand column: on the hand-written Test B the same model drops to **0.874**, a
0.126 drop. Out-of-template text — the closest thing to real data — is where the gap is.

About the spread: these models are deterministic, so changing the seed changes nothing. Instead we report a 95%
bootstrap confidence interval with 1,000 resamples of the test set. Test B's interval goes from 0.68 to 1.00 —
about 0.3 wide. With 24 items we cannot detect a 0.03 difference, which tells us we need a larger real test set.
The whole table is reproduced by one command: `python evaluation/midproject.py`."

---

## Slide 4 — Error analysis: categories (≈ 1:15) · Nisanur

"We read all **65 baseline errors** and grouped them into 7 categories. The keyword rules are not trained, so their
errors on the training split are fair to analyse; we did not look at test A, because it is frozen.

The largest category is **E1, 21 errors: 'topic beats action'.** When an exam's room changes, the label is room
change, but the text says 'exam', so the model predicts exam.
Second, **E4, 16 errors:** no keyword fires at all — for example 'Don't miss', 'no lecture', 'ekle-sil'.
**E2, 13 errors:** the word 'başvuru', 'application', appears in both scholarship and registration notices.
E3 is registration about exams, E5 is events that mention exams, and E6 is a room name like 'Seminer Salonu'
being read as an event. **E7, 3 errors:** TF-IDF's mistakes on Test B, caused by words it never saw in training."

---

## Slide 5 — What the errors tell us (≈ 1:15) · Nisanur

"We draw four conclusions.

One: **the label is the action, not the topic.** In 'the MAT 102 exam will be held in E-205 instead of the
Conference Hall', every cue says exam, but what the student must do is go to a new room. A bag-of-words model cannot
weigh 'X instead of Y'; a contextual encoder can. This is the strongest argument for BERTurk.

Two: **vocabulary gaps.** Phrases like 'guest lecture', 'tuition waiver' or 'ilişik kesme' never appear in
training. A pretrained model may already carry that knowledge.

Three: ambiguous cues like 'başvuru' are learned from data by TF-IDF — it makes zero errors on test A. So that is a
rules problem, not a learning problem.

Four: for extraction there are 7 errors on Test B — date ranges without a year, free-form place names, and dotted
times like '17.30'. Micro-F1 is still 0.947.

So we expect BERTurk to help only in the first two categories, which appear mainly outside the templates. That is
why we measure on Test B and the real Test C, not on test A."

---

## Slide 6 — Method status (≈ 1:15) · Emre

"The method is implemented and has been run. **The one controlled change:** we replace TF-IDF features plus
logistic regression with a fine-tuned BERTurk encoder. The split, the input — title plus text —, the labels and the
metric are identical.

Settings: 4 epochs, learning rate 3e-5, batch size 16, three seeds. On a laptop CPU it took about
6 minutes per seed.

**Our first number on the dev set: BERTurk reaches 0.807 ± 0.134 Macro-F1 over three seeds — below TF-IDF's
1.000.** The seeds range from 0.62 to 0.92, so the model is very unstable. The training loss is still between 1.1
and 1.6 after only 64 steps, so this is under-training, not a real weakness of the model. That is exactly why we
report three seeds. Next we train for 10 to 20 epochs on a GPU and choose the number of epochs on the dev set. We do
not touch the test sets until December 9.

The rest of the pipeline already works: extraction reaches 0.947 micro-F1 on Test B, the structured summariser
reaches ROUGE-L 0.953 against 0.432 for the lead-1 baseline, and question answering — end to end, with predicted
labels — answers 50 out of 50 questions correctly. The Flask web app and the command-line tool are ready."

---

## Slide 7 — Plan to the final (≈ 1:15) · Helin (table) + Cevahir (protocol, risk)

"Every remaining step has an owner and a date. Data collection is already done: 818 real GTÜ announcements. By November 27, two of us annotate 150 of them
independently, adding an 'other' class, compute Cohen's kappa and freeze them as
Test C. BERTurk runs on a GPU with three seeds. As the ablation, by December 4 we compare BERTurk with multilingual
XLM-R and with a version without the title field. On December 9 we run every system on the test sets **exactly
once** and run the paired bootstrap. The final report, in paper format, is drafted by December 18.

We fixed the evaluation protocol today, before seeing any method result: a gain counts as real only if it is at
least 0.03 Macro-F1 and its confidence interval excludes zero.

**Our biggest risk** is that real GTÜ text is far from our labels: the classifier's median confidence is 0.89 on
test A but only **0.37 on the GTÜ announcements**, and many of them fit none of our six classes. So labelling may be
slow or agreement low. **The fallback:** label 150 instead of 300, add an 'other' class, and report the minimum
detectable difference as a limitation.

Thank you — we are happy to take questions."

---

## Likely questions and short answers

- **"Your data is synthetic — how meaningful are the results?"** → That is why test A saturates. The decision is
  made on Test B and the real Test C, which is frozen on Nov 27 before any model sees it.
- **"Why bootstrap instead of seeds?"** → The TF-IDF models are deterministic; seeds change nothing. Seed std is
  reported for BERTurk and XLM-R over 3 seeds.
- **"Where does the 0.03 threshold come from?"** → Chosen in advance as the smallest practically useful gain; the
  paired-bootstrap CI must also exclude zero (Dror et al., 2018).
- **"Why not analyse errors on test A?"** → It is frozen; inspecting it would turn it into a development set.
- **"What is the ablation?"** → BERTurk vs XLM-R with the same recipe, BERTurk without the title, and a learning
  curve with 25/50/100% of the training data.
- **"Who did what?"** → Helin: data and guideline; Nisanur: research lead (related work, ablation design), extraction and error analysis; Cevahir: classifiers and
  evaluation; Emre: summarisation, QA and the web demo.
- **"Why is BERTurk worse than TF-IDF?"** → Only 252 training examples and 64 optimisation steps; the training
  loss had not converged (1.1–1.6) and one seed collapsed to 0.62. This is under-training; we will tune epochs and
  learning rate on dev before the single test run. If it still does not beat TF-IDF by 0.03 on Test B/C, our answer
  to the research question is "no".
