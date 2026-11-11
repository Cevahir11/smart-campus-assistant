# Smart Campus Announcement Assistant — Evaluation Report

Dataset: 420 announcements, 50 QA questions, reference date 2026-11-09 (synthetic seed data)

## 1. Classification (held-out 20%)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| keyword_rules | 0.881 | 0.878 |
| tfidf_nb | 1.000 | 1.000 |
| tfidf_svm | 1.000 | 1.000 |
| tfidf_logreg | 1.000 | 1.000 |

Per-class F1 (tfidf_nb):

| Class | P | R | F1 | Support |
|---|---|---|---|---|
| exam | 1.000 | 1.000 | 1.000 | 18 |
| event | 1.000 | 1.000 | 1.000 | 15 |
| scholarship | 1.000 | 1.000 | 1.000 | 12 |
| cancellation | 1.000 | 1.000 | 1.000 | 13 |
| room_change | 1.000 | 1.000 | 1.000 | 12 |
| registration | 1.000 | 1.000 | 1.000 | 14 |

## 2. Information extraction

**rules**

| Entity | Precision | Recall | F1 |
|---|---|---|---|
| COURSE | 1.000 | 1.000 | 1.000 |
| DATE | 1.000 | 1.000 | 1.000 |
| LOCATION | 1.000 | 1.000 | 1.000 |
| TIME | 1.000 | 1.000 | 1.000 |
| micro | 1.000 | 1.000 | 1.000 |

**Hand-written challenge set (real-world style, not template-generated)**

_rules_

| Entity | Precision | Recall | F1 |
|---|---|---|---|
| COURSE | 1.000 | 1.000 | 1.000 |
| DATE | 0.929 | 0.929 | 0.929 |
| LOCATION | 1.000 | 0.889 | 0.941 |
| TIME | 1.000 | 0.909 | 0.952 |
| micro | 0.969 | 0.925 | 0.947 |

**Challenge-set classification**

| Model | Accuracy | Macro-F1 |
|---|---|---|
| keyword_rules | 0.792 | 0.801 |
| tfidf_logreg | 0.875 | 0.874 |

## 3. Summarization (ROUGE F1 vs reference one-sentence summaries)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| lead_1 | 0.471 | 0.310 | 0.432 |
| centroid_extractive | 0.503 | 0.331 | 0.465 |
| structured_template | 0.953 | 0.916 | 0.953 |

## 4. Question answering

| Mode | Answer accuracy | Retrieval hit@1 | N |
|---|---|---|---|
| rules | 1.000 | 1.000 | 50 |

By question type (rules): cancellation 6/6, event 5/5, exam_when 14/14, exam_where 8/8, list_event 2/2, list_exam 2/2, no_answer 3/3, registration 4/4, scholarship_deadline 6/6

## Notes

- Scores on the synthetic seed data are optimistic: the rules and templates share a vocabulary. The challenge set and, above all, the planned 300–500 manually annotated real announcements are the numbers to report.
- QA runs end to end on *predicted* categories (5-fold out-of-fold) and *extracted* fields, not on gold labels.
