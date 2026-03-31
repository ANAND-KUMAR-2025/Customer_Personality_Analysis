"""
Customer Personality Analysis - Full ML Pipeline
Generates synthetic data, preprocesses, clusters, and outputs JSON results.
"""

import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

np.random.seed(42)

# ─── 1. Generate Synthetic Dataset ──────────────────────────────────────────

def generate_data(n=2240):
    """Mimics the UCI Marketing Campaign dataset structure."""
    ages = np.random.normal(45, 12, n).clip(18, 80).astype(int)
    incomes = np.where(
        ages < 35,
        np.random.normal(40000, 10000, n),
        np.where(ages < 55,
                 np.random.normal(65000, 20000, n),
                 np.random.normal(55000, 15000, n))
    ).clip(5000, 150000)

    education = np.random.choice(
        ["Basic", "2n Cycle", "Graduation", "Master", "PhD"],
        n, p=[0.04, 0.08, 0.50, 0.22, 0.16]
    )
    marital = np.random.choice(
        ["Single", "Together", "Married", "Divorced", "Widow"],
        n, p=[0.22, 0.25, 0.38, 0.10, 0.05]
    )
    kidhome = np.random.choice([0, 1, 2], n, p=[0.55, 0.35, 0.10])
    teenhome = np.random.choice([0, 1, 2], n, p=[0.60, 0.30, 0.10])

    recency = np.random.randint(0, 100, n)
    num_web_visits = np.random.randint(1, 20, n)
    num_purchases = np.random.randint(0, 25, n)

    # Spending correlated with income
    income_factor = (incomes / 65000)
    mnt_wines = (income_factor * np.random.normal(300, 200, n)).clip(0, 1500).astype(int)
    mnt_fruits = (income_factor * np.random.normal(25, 30, n)).clip(0, 200).astype(int)
    mnt_meat = (income_factor * np.random.normal(180, 150, n)).clip(0, 1000).astype(int)
    mnt_fish = (income_factor * np.random.normal(40, 45, n)).clip(0, 300).astype(int)
    mnt_sweets = (income_factor * np.random.normal(25, 30, n)).clip(0, 250).astype(int)
    mnt_gold = (income_factor * np.random.normal(45, 50, n)).clip(0, 350).astype(int)

    num_deals = np.random.randint(0, 15, n)
    num_catalog = np.random.randint(0, 12, n)
    num_store = np.random.randint(0, 13, n)
    num_web = np.random.randint(0, 14, n)

    accepted_cmp = np.random.binomial(1, 0.15, n)
    complain = np.random.binomial(1, 0.01, n)

    enrollment_years = np.random.randint(2012, 2023, n)

    df = pd.DataFrame({
        "Age": ages,
        "Income": incomes.astype(int),
        "Education": education,
        "Marital_Status": marital,
        "Kidhome": kidhome,
        "Teenhome": teenhome,
        "Recency": recency,
        "MntWines": mnt_wines,
        "MntFruits": mnt_fruits,
        "MntMeatProducts": mnt_meat,
        "MntFishProducts": mnt_fish,
        "MntSweetProducts": mnt_sweets,
        "MntGoldProds": mnt_gold,
        "NumDealsPurchases": num_deals,
        "NumWebPurchases": num_web,
        "NumCatalogPurchases": num_catalog,
        "NumStorePurchases": num_store,
        "NumWebVisitsMonth": num_web_visits,
        "AcceptedCmp": accepted_cmp,
        "Complain": complain,
        "Year_Customer": enrollment_years,
    })

    # Inject ~1% nulls in Income
    null_idx = np.random.choice(n, int(n * 0.01), replace=False)
    df.loc[null_idx, "Income"] = np.nan

    return df


# ─── 2. Preprocessing ────────────────────────────────────────────────────────

def preprocess(df):
    df = df.copy()

    # Handle nulls
    df["Income"].fillna(df["Income"].median(), inplace=True)

    # Remove outliers (IQR on income)
    Q1, Q3 = df["Income"].quantile(0.25), df["Income"].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df["Income"] >= Q1 - 1.5 * IQR) & (df["Income"] <= Q3 + 1.5 * IQR)]

    # Feature Engineering
    df["TotalSpend"] = (df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"]
                        + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"])
    df["TotalPurchases"] = (df["NumWebPurchases"] + df["NumCatalogPurchases"]
                            + df["NumStorePurchases"] + df["NumDealsPurchases"])
    df["FamilySize"] = df["Kidhome"] + df["Teenhome"]
    df["Customer_Tenure"] = 2024 - df["Year_Customer"]

    # Encode categoricals
    edu_map = {"Basic": 0, "2n Cycle": 1, "Graduation": 2, "Master": 3, "PhD": 4}
    df["Education_Enc"] = df["Education"].map(edu_map)

    marital_map = {"Married": 1, "Together": 1, "Single": 0, "Divorced": 0, "Widow": 0}
    df["IsCouple"] = df["Marital_Status"].map(marital_map)

    # Features for clustering
    features = [
        "Age", "Income", "TotalSpend", "TotalPurchases",
        "Recency", "FamilySize", "Customer_Tenure",
        "Education_Enc", "IsCouple", "NumWebVisitsMonth"
    ]
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled, features


# ─── 3. Elbow + Silhouette ───────────────────────────────────────────────────

def elbow_analysis(X_scaled, max_k=10):
    inertias, silhouettes = [], []
    ks = list(range(2, max_k + 1))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    return ks, inertias, silhouettes


# ─── 4. Clustering ───────────────────────────────────────────────────────────

def cluster(X_scaled, n_clusters=4):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return labels


# ─── 5. PCA ──────────────────────────────────────────────────────────────────

def run_pca(X_scaled, n_components=2):
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X_scaled)
    return coords, pca.explained_variance_ratio_.tolist()


# ─── 6. Persona Labeling ─────────────────────────────────────────────────────

PERSONA_TEMPLATES = [
    {"name": "Premium Loyalist",   "emoji": "👑", "color": "#F59E0B",
     "desc": "High income, high spend, loyal customers who respond well to premium offers."},
    {"name": "Frugal Family",      "emoji": "🏠", "color": "#3B82F6",
     "desc": "Mid income, larger families, deal-seekers with moderate spend."},
    {"name": "Rising Achiever",    "emoji": "🚀", "color": "#10B981",
     "desc": "Younger, growing income, high web activity, price-sensitive but aspirational."},
    {"name": "Passive Visitor",    "emoji": "👻", "color": "#8B5CF6",
     "desc": "High web visits but low conversions; disengaged from campaigns."},
]

def label_clusters(df_with_labels, n_clusters):
    """Assign personas based on cluster mean characteristics."""
    summary = df_with_labels.groupby("Cluster").agg(
        AvgIncome=("Income", "mean"),
        AvgSpend=("TotalSpend", "mean"),
        AvgAge=("Age", "mean"),
        AvgFamily=("FamilySize", "mean"),
        AvgRecency=("Recency", "mean"),
        Count=("Income", "count"),
    ).reset_index()

    # Sort by income desc to map personas
    summary = summary.sort_values("AvgIncome", ascending=False).reset_index(drop=True)
    persona_map = {}
    for i, row in summary.iterrows():
        p = PERSONA_TEMPLATES[i % len(PERSONA_TEMPLATES)]
        persona_map[int(row["Cluster"])] = p

    return summary, persona_map


# ─── 7. Main ─────────────────────────────────────────────────────────────────

def run_pipeline():
    print("🔄 Generating data...")
    df_raw = generate_data(2240)

    print("🧹 Preprocessing...")
    df, X_scaled, features = preprocess(df_raw)

    print("📐 Elbow analysis...")
    ks, inertias, silhouettes = elbow_analysis(X_scaled)

    print("🔍 Clustering (k=4)...")
    n_clusters = 4
    labels = cluster(X_scaled, n_clusters)
    df["Cluster"] = labels

    print("📉 PCA...")
    pca_coords, var_ratio = run_pca(X_scaled)
    df["PCA1"] = pca_coords[:, 0]
    df["PCA2"] = pca_coords[:, 1]

    print("🧠 Profiling personas...")
    cluster_summary, persona_map = label_clusters(df, n_clusters)

    # ── Build output JSON ─────────────────────────────────────────────────────

    # Elbow chart data
    elbow_data = [{"k": k, "inertia": round(i, 2), "silhouette": round(s, 4)}
                  for k, i, s in zip(ks, inertias, silhouettes)]

    # PCA scatter (sample 600 for performance)
    sample = df.sample(600, random_state=42)
    scatter_data = [
        {
            "x": round(float(r.PCA1), 3),
            "y": round(float(r.PCA2), 3),
            "cluster": int(r.Cluster),
            "persona": persona_map[int(r.Cluster)]["name"],
            "income": int(r.Income),
            "spend": int(r.TotalSpend),
            "age": int(r.Age),
        }
        for _, r in sample.iterrows()
    ]

    # Cluster profiles
    cluster_profiles = []
    for _, row in cluster_summary.iterrows():
        cid = int(row["Cluster"])
        p = persona_map[cid]
        cluster_profiles.append({
            "cluster_id": cid,
            "persona": p["name"],
            "emoji": p["emoji"],
            "color": p["color"],
            "description": p["desc"],
            "count": int(row["Count"]),
            "avg_income": int(row["AvgIncome"]),
            "avg_spend": int(row["AvgSpend"]),
            "avg_age": round(float(row["AvgAge"]), 1),
            "avg_family_size": round(float(row["AvgFamily"]), 2),
            "avg_recency": round(float(row["AvgRecency"]), 1),
        })

    # Spend breakdown per cluster
    spend_cols = ["MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]
    spend_breakdown = []
    for cid in df["Cluster"].unique():
        sub = df[df["Cluster"] == cid]
        entry = {"cluster": int(cid), "persona": persona_map[int(cid)]["name"]}
        for col in spend_cols:
            entry[col] = int(sub[col].mean())
        spend_breakdown.append(entry)

    # Income distribution per cluster (histogram bins)
    income_dist = {}
    bins = list(range(0, 160000, 20000))
    for cid in sorted(df["Cluster"].unique()):
        sub = df[df["Cluster"] == cid]["Income"]
        counts, _ = np.histogram(sub, bins=bins)
        income_dist[str(cid)] = {
            "persona": persona_map[int(cid)]["name"],
            "color": persona_map[int(cid)]["color"],
            "bins": [f"{b//1000}k-{(b+20000)//1000}k" for b in bins[:-1]],
            "counts": counts.tolist(),
        }

    # Channel preference
    channel_data = []
    for cid in sorted(df["Cluster"].unique()):
        sub = df[df["Cluster"] == cid]
        channel_data.append({
            "persona": persona_map[int(cid)]["name"],
            "color": persona_map[int(cid)]["color"],
            "web": round(float(sub["NumWebPurchases"].mean()), 2),
            "catalog": round(float(sub["NumCatalogPurchases"].mean()), 2),
            "store": round(float(sub["NumStorePurchases"].mean()), 2),
            "deals": round(float(sub["NumDealsPurchases"].mean()), 2),
        })

    # Summary KPIs
    kpis = {
        "total_customers": len(df),
        "avg_income": int(df["Income"].mean()),
        "avg_spend": int(df["TotalSpend"].mean()),
        "avg_age": round(float(df["Age"].mean()), 1),
        "pca_variance_explained": [round(v * 100, 1) for v in var_ratio],
        "best_k": int(ks[silhouettes.index(max(silhouettes))]),
        "best_silhouette": round(max(silhouettes), 4),
    }

    result = {
        "kpis": kpis,
        "elbow_data": elbow_data,
        "scatter_data": scatter_data,
        "cluster_profiles": cluster_profiles,
        "spend_breakdown": spend_breakdown,
        "income_distribution": income_dist,
        "channel_data": channel_data,
        "features_used": features,
    }

    out_path = "/home/claude/customer_personality/data/analysis_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"✅ Done! Results saved to {out_path}")
    return result


if __name__ == "__main__":
    run_pipeline()
