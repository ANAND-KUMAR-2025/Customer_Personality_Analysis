 # Customer Personality Analysis

End-to-end customer segmentation system using unsupervised machine learning.
Segments 2,200+ customers into buyer personas using K-Means clustering, validated
with Elbow and Silhouette methods, visualised via PCA — all connected to an
interactive multi-tab dashboard.

## What it does

- Preprocesses raw customer data: null imputation, IQR outlier removal,
  feature engineering (TotalSpend, FamilySize, Customer_Tenure), StandardScaler normalisation
- Runs Elbow + Silhouette analysis over k=2..10 to select optimal clusters
- Fits K-Means (k=4) on 10 engineered features to discover buyer segments
- Reduces to 2D with PCA for visual cluster inspection
- Profiles each cluster into named marketing personas with spend and channel breakdowns
- Exports structured JSON consumed by a standalone interactive dashboard

## Personas discovered

| Segment             | Customers  | Avg Income | Avg Spend |
|---------------------|------------|------------|---------|
| 👑 Premium Loyalist | 443        | $81,946    | $989    |
| 🏠 Frugal Family    | 578          | $58,562 | $528 |
| 🚀 Rising Achiever  | 664 | $55,676 | $482 |
| 👻 Passive Visitor  | 518 | $38,760 | $367 |

## Tech Stack

- **ML**: scikit-learn (KMeans, PCA, StandardScaler, silhouette_score)
- **Data**: pandas, numpy
- **Dashboard**: Vanilla JS, Chart.js, HTML5 Canvas (custom PCA scatter)
- **Output**: JSON pipeline → static frontend, no server required

## Run it

pip install pandas numpy scikit-learn
python backend/pipeline.py
# open dashboard.html in browser

## Dashboard tabs

- **Overview** — KPIs, spend by category, cluster distribution, income histogram
- **Personas** — per-segment cards, income vs spend scatter, profile table
- **PCA Scatter** — interactive 2D projection with hover tooltips
- **Spend Analysis** — product breakdown, stacked %, channel patterns
- **Clustering** — elbow curve, silhouette scores, pipeline summary
```

 
