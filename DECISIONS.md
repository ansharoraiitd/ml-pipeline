# Decision Log
A record of every technical choice I made and why.

---
## Week 3 — Real Data: Adult Census Income
### Why this dataset for the flagship project
Real missing data (encoded as "?", not NaN), genuine class
imbalance (~24% positive), mixed categorical/numerical features,
and a known redundant-feature trap (education vs education-num) —
exactly the mess ml-fundamentals deliberately avoided. This
dataset carries through feature engineering, tuning, explainability,
and the capstone, so every later week builds on the same base.