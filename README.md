# Customer Churn Prediction & Dimensionality Reduction

Two applied machine-learning analyses in R: predicting whether a telecom customer
will churn, and testing whether PCA dimensionality reduction loses information
when clustering congressional voting records.

## Part 1 — T-Mobile Customer Churn Prediction

T-Mobile asked whether a customer will leave the service, based on their usage
pattern. The `mlc_churn` dataset covers **5,000 customers** with 20 features:
state, account length, area code, international plan, voice-mail plan, number of
voice-mail messages, total day/evening/night/international minutes, calls and
charges, and number of customer-service calls. The target is `churn`:
**707 customers churned (14.1%)**, 4,293 stayed.

The data was split **75% train / 25% test**, and three classifiers were trained
on the training set and evaluated on the test set, each with a confusion matrix,
accuracy, sensitivity, and specificity:

| Model | Approach |
|---|---|
| Logistic regression | Linear model on the usage features |
| Naive Bayes | Probabilistic classifier |
| K-nearest neighbors | Distance-based classifier |

The confusion matrices and evaluation metrics are shown as R output in the
full report (`report/Churn_Prediction_Report.pdf`).

## Part 2 — PCA and K-Means on 1984 Congress Votes

Does PCA, as a dimensionality-reduction method, lose much information? The
test case is the 1984 U.S. House voting record: **435 members** (267 Republican,
168 Democrat) across **16 roll-call votes**, coded 1 = yea, −1 = nay, 0 = abstain.

- **PCA:** scatter plots of PC1 vs PC2, PC1 vs PC3, and PC2 vs PC3, with the
  PC1–PC2 plot color-coded by party — the two parties are visibly separable
  in the first two components.
- **K-means (k = 2) on the full 16-dimensional data:** clusters match the
  actual party labels with **88.28% accuracy**.
- **K-means (k = 2) on PC1 and PC2 only:** clusters match the actual party
  labels with **87.82% accuracy**.

Reducing the data to two principal components costs only 0.46 percentage
points of clustering accuracy, so PCA retains essentially all of the
information that separates the parties — a simpler, lower-dimensional analysis
without meaningfully compromising performance.

## Repository Layout

```
├── README.md
├── code/                  # R scripts (tmobile.r, votes-last.r) — to be added
├── data/
│   ├── mlc_churn.csv                          # 5,000 customers, 20 features + churn
│   ├── congress_1984_votes.csv                # 435 members × 16 votes (1 / -1 / 0)
│   └── congress_1984_party_affiliations.csv   # Party label per member
└── report/
    ├── Churn_Prediction_Report.pdf
    └── Churn_Prediction_Report.docx
```

## Running the Analysis

The analysis is written in R. Once the scripts in `code/` are added, run them
from the repository root so the relative `data/` paths resolve. The report
contains the full R output, including the confusion matrices and the PCA and
k-means plots.
