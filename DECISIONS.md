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

---
## Capstone Decision (Week 5-6)
### What we're building
Extending THIS repo (not a new one) into a deployed system:
FastAPI service serving the tuned Week 3 model, returning both a
prediction and a per-prediction SHAP explanation. Streamlit
frontend as the user-facing form + result display.

### Why extend ml-pipeline instead of starting fresh
The cleaning, feature engineering, tuning, and imbalance decisions
already documented here ARE the capstone's foundation. A new repo
would orphan that context. The capstone is this project's natural
conclusion, not a separate one.