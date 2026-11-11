# Speaker Script — Nisanur Kara · Team Syntax Error (≈ 7 min)

Tip: bold words are the ones to stress. Pause briefly at each "—".

## Slide 1 — Research question
"Good afternoon, we are team **Syntax Error**, and our project is the **Smart Campus Announcement Assistant**.

Every week students miss exam dates, room changes and deadlines, because announcements are scattered across
department pages, e-mails and club channels. Our system reads these announcements, sorts them into categories,
pulls out the date, time, place and course code, and answers questions like *'Is there an exam this week?'* — always
showing the source.

The question we want to answer is simple: **does a fine-tuned Turkish BERT model, BERTurk, classify these
announcements better than a classic TF-IDF baseline — on text written outside our templates?**
We also decided in advance what a **'no'** looks like: a gain smaller than **0.03 Macro-F1**, or a confidence
interval that includes zero.

In our proposal we asked a broader 'how' question. We narrowed it, because a 'how' question can never be proven
wrong."

## Slide 2 — Data
"So, what data do we test this on?

We have **420 labelled announcements** in six categories, split into **252 for training, 84 for development and
84 for testing** — and the test set is **frozen**. Here on the right you can see one real example: a chemistry class
cancellation, with its label and the extracted date and course code.

To be honest, these 420 are **synthetic**, so they share words with our rules. That is why we also wrote
**24 announcements by hand**, in real-world style — we call it **Test B**.

And since yesterday, we also have real data: **818 announcements from 24 GTÜ departments**, collected with our own
crawler. They are not labelled yet — that is our next step."

## Slide 3 — Baseline
"Let's look at the baseline results.

On the synthetic test set, TF-IDF with logistic regression gets a perfect **1.000** — which tells us this test set
is simply **too easy**.

The interesting number is on the right: on our hand-written Test B the same model drops to **0.874**. So the gap
appears exactly where the text looks like the real world.

Because these models are deterministic, we report **95% bootstrap confidence intervals** instead of seeds. Test B's
interval is wide, since it has only 24 items — so we clearly need a larger real test set."

## Slide 4 — Error analysis
"To understand that gap, we looked at **65 baseline errors** one by one and grouped them into seven categories.

The biggest group, with **21 errors**, is what we call *'topic beats action'*: when an exam's room changes, the
model sees the word *'exam'* and says **exam**, although the correct label is **room change**.
The second group, with **16 errors**, is announcements where no keyword fires at all."

## Slide 5 — What the errors suggest
"And this gives us our main insight: **the label is the action, not the topic.**

A word-counting model cannot understand a sentence like *'the exam will be held in room B instead of room A'*.
A contextual model like **BERTurk can** — and that is exactly why we are testing it.
The second problem is **unseen vocabulary**, and pretrained models may help there too.

So we expect BERTurk to make a difference **only on real, out-of-template text** — not on the easy synthetic set."

## Slide 6 — Method status
"Now, where are we with the method?

BERTurk is **implemented and has already been run**. The only thing we change is the model — the data, the split
and the metric stay exactly the same.

Our first development-set result is **0.807**, with a large spread across three seeds — so it is still **below**
TF-IDF. The reason is clear: it trained for only **64 steps** on a laptop CPU. Next, we train it properly on a GPU.

The rest of the pipeline already works: information extraction reaches **0.947 F1**, and question answering
answers **all 50** of our test questions correctly."

## Slide 7 — Plan to the final
"Finally, our plan to the final.

Every step has an **owner and a date**. By **November 27**, we will label **150 real GTÜ announcements** — that
becomes our real test set, **Test C**. As the ablation, we compare BERTurk with the multilingual **XLM-R**. And on
**December 9**, we run every system on the test sets **exactly once**.

Our **biggest risk** is that real GTÜ text is very different from our labels: the model's confidence drops from
**0.89 to 0.37**. Our **fallback** is to label 150 instead of 300, and to add an **'other'** category.

So by the final, we will be able to answer our question with **real data**: does BERTurk really help?

Thank you — we are happy to take your questions."
