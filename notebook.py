"""
shopper_spectrum.py
---------------------
ONE FILE, everything: the training pipeline AND the Streamlit app.

Train the models (run this first):
    python shopper_spectrum.py

Then launch the app (loads the models the line above just saved):
    streamlit run shopper_spectrum.py

How one file does both: when Streamlit runs a script it sets up a "runtime"
that isn't there for a plain `python` run. We check for that at the bottom
of this file and branch into either train() or run_app() accordingly.

Steps below follow the project brief:
  STEP 1 - Load + explore the raw data
  STEP 2 - Clean it
  STEP 3 - Exploratory Data Analysis
  STEP 4 - RFM feature engineering + KMeans clustering
  STEP 5 - Label clusters from their own data (not a hardcoded guess)
  STEP 6 - Product recommendation model (item-based collaborative filtering)
  STEP 7 - Save the models for the Streamlit app
  STEP 8 - The Streamlit app itself
"""

import pickle
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D plotting)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# =============================================================
# CONFIG
# =============================================================
DATA_PATH = "online_retail.csv"      # dataset lives next to this script
RANDOM_STATE = 42
K_RANGE = range(2, 11)               # candidate cluster counts to try

# Number of clusters to use. k=4 maps to the brief's 4 named segments
# (High-Value / Regular / Occasional / At-Risk). Set to None to instead
# auto-pick whichever k scores highest on silhouette score.
FINAL_K = 4

KMEANS_MODEL_PATH = "kmeans_model.pkl"
SCALER_PATH = "scaler.pkl"
CLUSTER_LABELS_PATH = "cluster_labels.pkl"
PRODUCT_SIMILARITY_PATH = "product_similarity.pkl"
PRODUCT_POPULARITY_PATH = "product_popularity.pkl"


# =============================================================
# STEP 1: Load + explore the raw data
# =============================================================
def load_raw():
    return pd.read_csv(DATA_PATH, encoding="ISO-8859-1")


def explore(df):
    print("Shape:", df.shape)

    print("\nColumn dtypes:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isna().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    cancelled = df["InvoiceNo"].astype(str).str.startswith("C").sum()
    print("Cancelled invoices (InvoiceNo starts with 'C'):", cancelled)

    print("Rows with Quantity <= 0:", (df["Quantity"] <= 0).sum())
    print("Rows with UnitPrice <= 0:", (df["UnitPrice"] <= 0).sum())


# =============================================================
# STEP 2: Clean the data
# =============================================================
def clean(df):
    before = len(df)

    df = df.dropna(subset=["CustomerID"]).copy()
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df = df.drop_duplicates()

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    print(f"Cleaned: {before} rows -> {len(df)} rows ({before - len(df)} removed)")
    return df


# =============================================================
# STEP 3: Exploratory Data Analysis
# =============================================================
def run_eda(df, rfm):
    country_sales = df.groupby("Country")["InvoiceNo"].nunique().sort_values(ascending=False)
    plt.figure(figsize=(12, 5))
    country_sales.head(15).plot(kind="bar")
    plt.title("Transactions by Country (top 15)")
    plt.ylabel("Unique Transactions")
    plt.tight_layout()
    plt.show()

    top = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(12, 5))
    top.plot(kind="bar", color="teal")
    plt.title("Top 10 Selling Products")
    plt.ylabel("Quantity Sold")
    plt.tight_layout()
    plt.show()

    monthly = df.set_index("InvoiceDate").resample("ME")["TotalPrice"].sum()
    plt.figure(figsize=(12, 5))
    monthly.plot()
    plt.title("Monthly Sales Trend")
    plt.ylabel("Total Revenue")
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    per_txn = df["TotalPrice"]
    axes[0].hist(per_txn.clip(upper=per_txn.quantile(0.99)), bins=50)
    axes[0].set_title("Spend per Transaction (clipped at 99th pct)")
    per_customer = df.groupby("CustomerID")["TotalPrice"].sum()
    axes[1].hist(per_customer.clip(upper=per_customer.quantile(0.99)), bins=50, color="orange")
    axes[1].set_title("Total Spend per Customer (clipped at 99th pct)")
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(rfm["Recency"], bins=40)
    axes[0].set_title("Recency distribution")
    axes[1].hist(rfm["Frequency"].clip(upper=rfm["Frequency"].quantile(0.99)), bins=40)
    axes[1].set_title("Frequency distribution")
    axes[2].hist(rfm["Monetary"].clip(upper=rfm["Monetary"].quantile(0.99)), bins=40)
    axes[2].set_title("Monetary distribution")
    plt.tight_layout()
    plt.show()


# =============================================================
# STEP 4: RFM feature engineering + clustering
# =============================================================
def compute_rfm(df):
    """
    Recency   = days since each customer's most recent purchase
    Frequency = number of distinct invoices per customer
    Monetary  = total amount spent per customer
    """
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    )
    return rfm


def log_transform_for_clustering(recency, frequency, monetary):
    """
    Frequency and Monetary are usually heavily right-skewed (a few customers
    spend WAY more than everyone else). Left as-is, those outliers dominate
    the distance calculations KMeans uses, and clustering just splits
    "outliers" vs "everyone else" instead of finding real segments.

    log1p() squashes that skew before scaling. Recency is left alone since
    it's naturally bounded and not nearly as skewed.

    This same transform is applied both when training (in train()) and when
    the app makes a prediction on new Recency/Frequency/Monetary input.
    """
    return recency, np.log1p(frequency), np.log1p(monetary)


def find_best_k(rfm_scaled, k_range):
    """Try each k, track inertia (elbow) + silhouette score, return the best k."""
    ks = list(k_range)
    inertia = []
    sil_scores = []

    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(rfm_scaled)
        inertia.append(km.inertia_)
        sil_scores.append(silhouette_score(rfm_scaled, labels))

    best_k = ks[sil_scores.index(max(sil_scores))]
    return ks, inertia, sil_scores, best_k


def label_clusters(rfm_with_clusters):
    """
    Turn KMeans's arbitrary cluster numbers into business labels, based on
    each cluster's own average RFM values - NOT a hardcoded position. This
    keeps the labels correct no matter how KMeans happens to number things.
    """
    profile = rfm_with_clusters.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()

    profile["monetary_rank"] = profile["Monetary"].rank(ascending=False)
    profile["frequency_rank"] = profile["Frequency"].rank(ascending=False)
    profile["recency_rank"] = profile["Recency"].rank(ascending=True)
    profile["value_score"] = profile["monetary_rank"] + profile["frequency_rank"] + profile["recency_rank"]
    profile = profile.sort_values("value_score")

    ordered_ids = profile.index.tolist()
    n = len(ordered_ids)

    if n == 4:
        label_names = ["High-Value Customers", "Regular Customers",
                        "Occasional Shoppers", "At-Risk Customers"]
    else:
        label_names = [f"Segment {i + 1} (best to worst)" for i in range(n)]

    cluster_to_label = {cid: label_names[i] for i, cid in enumerate(ordered_ids)}
    return cluster_to_label, profile


# =============================================================
# STEP 6: Product recommendation model (item-based collaborative filtering)
# =============================================================
def build_product_similarity(df):
    customer_product = df.pivot_table(
        index="CustomerID", columns="Description", values="Quantity", aggfunc="sum"
    ).fillna(0)

    similarity = cosine_similarity(customer_product.T)
    return pd.DataFrame(similarity, index=customer_product.columns, columns=customer_product.columns)

def recommend_products(product_similarity_df, product_name, top_n=5):
    if product_name not in product_similarity_df.index:
        return []
    sim_scores = product_similarity_df[product_name].sort_values(ascending=False)
    return sim_scores.iloc[1:top_n + 1].index.tolist()


def find_exact_product(products, query):
    """Case-insensitive exact match (dataset descriptions are stored as-typed,
    sometimes with stray trailing spaces, so also strip before comparing)."""
    query_norm = query.strip().lower()
    for product in products:
        if product.strip().lower() == query_norm:
            return product
    return None


def resolve_product(products, query, popularity=None):
    """
    Turn free-text user input into one real product from the similarity
    matrix, so it can be fed into the item-based CF model below. Exact
    (case-insensitive) match wins; otherwise plain regex substring search
    (no fuzzy matching / NLP) finds candidates, e.g. "lunch box" matches
    LUNCH BOX I LOVE LONDON, DOLLY GIRL LUNCH BOX, etc. Among candidates,
    the best-selling one (highest total quantity ever sold) is picked as
    the closest match to the query - falls back to shortest description
    if no popularity data is available.
    """
    exact = find_exact_product(products, query)
    if exact:
        return exact

    query = query.strip()
    if not query:
        return None
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    candidates = [p for p in products if pattern.search(p)]
    if not candidates:
        return None
    if popularity is not None:
        return max(candidates, key=lambda p: popularity.get(p, 0))
    return min(candidates, key=len)


# =============================================================
# TRAINING PIPELINE - runs every step in order, saves the models
# =============================================================
def train():
    print("=" * 60)
    print("STEP 1: LOAD + EXPLORE RAW DATA")
    print("=" * 60)
    df = load_raw()
    explore(df)

    print("\n" + "=" * 60)
    print("STEP 2: CLEAN DATA")
    print("=" * 60)
    df = clean(df)

    print("\n" + "=" * 60)
    print("STEP 4: RFM FEATURE ENGINEERING")
    print("=" * 60)
    rfm = compute_rfm(df)
    print(rfm.describe())

    print("\n" + "=" * 60)
    print("STEP 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    run_eda(df, rfm)

    # log-transform Frequency/Monetary so a handful of big-spending outliers
    # don't dominate the clustering (see log_transform_for_clustering above)
    rfm_for_clustering = pd.DataFrame(index=rfm.index)
    (rfm_for_clustering["Recency"],
     rfm_for_clustering["Frequency"],
     rfm_for_clustering["Monetary"]) = log_transform_for_clustering(
        rfm["Recency"], rfm["Frequency"], rfm["Monetary"]
    )

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_for_clustering)

    print("\nSearching for the best number of clusters (elbow + silhouette)...")
    ks, inertia, sil_scores, best_k = find_best_k(rfm_scaled, K_RANGE)

    plt.figure(figsize=(6, 4))
    plt.plot(ks, inertia, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(ks, sil_scores, marker="o", color="red")
    plt.title("Silhouette Scores")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.tight_layout()
    plt.show()

    chosen_k = FINAL_K if FINAL_K else best_k
    print(f"\nBest k by silhouette score: {best_k}")
    print(f"Using k = {chosen_k}  (set FINAL_K at the top of this file to override)")

    print("\n" + "=" * 60)
    print("STEP 5: FIT FINAL KMEANS MODEL + LABEL CLUSTERS")
    print("=" * 60)
    kmeans = KMeans(n_clusters=chosen_k, random_state=RANDOM_STATE, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    final_silhouette = sil_scores[ks.index(chosen_k)] if chosen_k in ks else None
    print(f"Final model: k={chosen_k}, inertia={kmeans.inertia_:.1f}"
          + (f", silhouette={final_silhouette:.3f}" if final_silhouette is not None else ""))

    cluster_to_label, profile = label_clusters(rfm)
    print("\nCluster profiles (average RFM values):")
    print(profile[["Recency", "Frequency", "Monetary"]])
    print("\nCluster -> label mapping:")
    for cid, label in cluster_to_label.items():
        print(f"  Cluster {cid}: {label}")

    profile[["Recency", "Frequency", "Monetary"]].plot(
        kind="bar", subplots=True, figsize=(8, 8), layout=(3, 1), legend=False
    )
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(rfm["Recency"], rfm["Frequency"], rfm["Monetary"],
               c=rfm["Cluster"], cmap="tab10", alpha=0.5, s=10)
    ax.set_xlabel("Recency")
    ax.set_ylabel("Frequency")
    ax.set_zlabel("Monetary")
    ax.set_title("Customer Clusters (RFM space)")
    plt.tight_layout()
    plt.show()

    # STEP 7: save the models for the Streamlit app
    pickle.dump(kmeans, open(KMEANS_MODEL_PATH, "wb"))
    pickle.dump(scaler, open(SCALER_PATH, "wb"))
    pickle.dump(cluster_to_label, open(CLUSTER_LABELS_PATH, "wb"))
    print(f"\nSaved {KMEANS_MODEL_PATH}, {SCALER_PATH}, {CLUSTER_LABELS_PATH}")

    print("\n" + "=" * 60)
    print("STEP 6: PRODUCT RECOMMENDATION MODEL")
    print("=" * 60)
    product_similarity_df = build_product_similarity(df)
    pickle.dump(product_similarity_df, open(PRODUCT_SIMILARITY_PATH, "wb"))
    print(f"Saved {PRODUCT_SIMILARITY_PATH}  (shape: {product_similarity_df.shape})")

    # total quantity ever sold per product - used by the app to pick the
    # best-selling candidate when a search resolves to multiple products
    product_popularity = df.groupby("Description")["Quantity"].sum()
    pickle.dump(product_popularity, open(PRODUCT_POPULARITY_PATH, "wb"))
    print(f"Saved {PRODUCT_POPULARITY_PATH}")

    top_products = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(15).index
    sample_sim = product_similarity_df.loc[top_products, top_products]

    plt.figure(figsize=(10, 8))
    plt.imshow(sample_sim, cmap="viridis")
    plt.xticks(range(len(top_products)), top_products, rotation=90, fontsize=7)
    plt.yticks(range(len(top_products)), top_products, fontsize=7)
    plt.colorbar(label="Cosine similarity")
    plt.title("Product Similarity Heatmap (top 15 products)")
    plt.tight_layout()
    plt.show()

    example_product = top_products[0]
    print(f"\nExample recommendation for '{example_product}':")
    print(recommend_products(product_similarity_df, example_product))

    print("\nAll done. Run `streamlit run shopper_spectrum.py` to try the app.")


# =============================================================
# STEP 8: THE STREAMLIT APP
# =============================================================
@st.cache_resource
def load_artifacts():
    kmeans = pickle.load(open(KMEANS_MODEL_PATH, "rb"))
    scaler = pickle.load(open(SCALER_PATH, "rb"))
    cluster_labels = pickle.load(open(CLUSTER_LABELS_PATH, "rb"))
    product_similarity = pickle.load(open(PRODUCT_SIMILARITY_PATH, "rb"))
    product_popularity = pickle.load(open(PRODUCT_POPULARITY_PATH, "rb"))
    return kmeans, scaler, cluster_labels, product_similarity, product_popularity


def home_page():
    st.title("Shopper Spectrum")
    st.write("Customer Segmentation & Product Recommendation in E-Commerce")
    st.write(
        "Use the sidebar to open **Clustering** (find a customer's segment) "
        "or **Recommendation** (find similar products)."
    )


def clustering_page(kmeans, scaler, cluster_labels):
    st.title("Customer Segmentation")

    recency = st.number_input("Recency (days since last purchase)", min_value=0, step=1)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0, step=1)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, step=1.0)

    if st.button("Predict Segment"):
        # must match the log1p(Frequency)/log1p(Monetary) transform used
        # during training, or predictions will be meaningless
        r, f, m = log_transform_for_clustering(recency, frequency, monetary)
        features = pd.DataFrame([[r, f, m]], columns=["Recency", "Frequency", "Monetary"])
        features_scaled = scaler.transform(features)
        cluster = kmeans.predict(features_scaled)[0]
        label = cluster_labels.get(cluster, "Unknown")

        st.success(f"This customer belongs to: {label}")


def recommendation_page(product_similarity, product_popularity):
    st.title("Product Recommender")

    query = st.text_input("Enter Product Name")

    if st.button("Get Recommendations"):
        query = query.strip()
        if not query:
            st.warning("Type a product name first.")
            return

        # regex-resolve free text (e.g. "lunch box") to one real product,
        # then let the item-based CF cosine-similarity model do the actual
        # recommending - the regex step never produces the results itself
        anchor = resolve_product(product_similarity.index, query, product_popularity)
        if anchor is None:
            st.error("No products found matching that name.")
            return

        recs = recommend_products(product_similarity, anchor, top_n=5)
        if anchor.strip().lower() != query.lower():
            st.info(f"Matched '{query}' to product: **{anchor.strip()}**")

        st.subheader("Recommended Products")
        for prod in recs:
            with st.container(border=True):
                st.markdown(f"🛍️ **{prod.strip()}**")


def run_app():
    # must be the very first Streamlit command in the whole script run -
    # even the cache-loading spinner below counts as a command, so this
    # has to come before load_artifacts()
    st.set_page_config(page_title="Shopper Spectrum", layout="wide")

    kmeans, scaler, cluster_labels, product_similarity, product_popularity = load_artifacts()

    # sidebar navigation - matches the Home / Clustering / Recommendation
    # pages from the project brief's UI mockup
    page = st.sidebar.radio(
        "Navigation", ["Home", "Clustering", "Recommendation"], label_visibility="collapsed"
    )

    if page == "Home":
        home_page()
    elif page == "Clustering":
        clustering_page(kmeans, scaler, cluster_labels)
    elif page == "Recommendation":
        recommendation_page(product_similarity, product_popularity)


# =============================================================
# ENTRY POINT
# -----------------------------------------------------------
# `streamlit run shopper_spectrum.py` sets up a Streamlit runtime that a
# plain `python shopper_spectrum.py` doesn't have - that's how we tell the
# two use-cases apart from the same file.
# =============================================================
if __name__ == "__main__":
    if st.runtime.exists():
        run_app()
    else:
        train()
