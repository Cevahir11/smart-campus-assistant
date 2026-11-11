# Per-class precision / recall / F1 (frozen mid-project split)

## keyword_rules — Test A (synthetic, n=84)

Accuracy 0.857 · Macro-P 0.903 · Macro-R 0.851 · Macro-F1 0.840

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| exam | 0.750 | 1.000 | 0.857 | 18 |
| event | 1.000 | 1.000 | 1.000 | 15 |
| scholarship | 0.667 | 1.000 | 0.800 | 12 |
| cancellation | 1.000 | 1.000 | 1.000 | 13 |
| room_change | 1.000 | 0.750 | 0.857 | 12 |
| registration | 1.000 | 0.357 | 0.526 | 14 |

Confusions (gold->pred: n): room_change->exam: 3, registration->scholarship: 6, registration->exam: 3

## keyword_rules — Test B (hand-written, n=24)

Accuracy 0.792 · Macro-P 0.907 · Macro-R 0.792 · Macro-F1 0.801

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| exam | 0.444 | 1.000 | 0.615 | 4 |
| event | 1.000 | 0.500 | 0.667 | 4 |
| scholarship | 1.000 | 1.000 | 1.000 | 4 |
| cancellation | 1.000 | 1.000 | 1.000 | 4 |
| room_change | 1.000 | 0.500 | 0.667 | 4 |
| registration | 1.000 | 0.750 | 0.857 | 4 |

Confusions (gold->pred: n): event->exam: 2, room_change->exam: 2, registration->exam: 1

## tfidf_logreg — Test A (synthetic, n=84)

Accuracy 1.000 · Macro-P 1.000 · Macro-R 1.000 · Macro-F1 1.000

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| exam | 1.000 | 1.000 | 1.000 | 18 |
| event | 1.000 | 1.000 | 1.000 | 15 |
| scholarship | 1.000 | 1.000 | 1.000 | 12 |
| cancellation | 1.000 | 1.000 | 1.000 | 13 |
| room_change | 1.000 | 1.000 | 1.000 | 12 |
| registration | 1.000 | 1.000 | 1.000 | 14 |

## tfidf_logreg — Test B (hand-written, n=24)

Accuracy 0.875 · Macro-P 0.892 · Macro-R 0.875 · Macro-F1 0.874

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| exam | 0.800 | 1.000 | 0.889 | 4 |
| event | 1.000 | 0.750 | 0.857 | 4 |
| scholarship | 0.750 | 0.750 | 0.750 | 4 |
| cancellation | 0.800 | 1.000 | 0.889 | 4 |
| room_change | 1.000 | 1.000 | 1.000 | 4 |
| registration | 1.000 | 0.750 | 0.857 | 4 |

Confusions (gold->pred: n): event->cancellation: 1, scholarship->exam: 1, registration->scholarship: 1
