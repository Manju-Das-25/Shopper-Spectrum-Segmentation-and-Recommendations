import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# ============================
# 2. Load Dataset
# ============================
df = pd.read_csv("OnlineRetail.csv", encoding="ISO-8859-1")
print("Initial Shape:", df.shape)
print(df.head())

# ============================
# 3. Data Preprocessing
# ============================
df = df.dropna(subset=["CustomerID"])
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
print("Cleaned Shape:", df.shape)

# ============================
# 4. Exploratory Data Analysis
# ============================
country_sales = df.groupby("Country")["InvoiceNo"].nunique().sort_values(ascending=False)
plt.figure(figsize=(12,5))
country_sales.plot(kind="bar")
plt.title("Transactions by Country")
plt.ylabel("Unique Transactions")
plt.show()

top_products = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(12,5))
top_products.plot(kind="bar", color="teal")
plt.title("Top 10 Selling Products")
plt.ylabel("Quantity Sold")
plt.show()

monthly_sales = df.set_index("InvoiceDate").resample("M")["TotalPrice"].sum()
plt.figure(figsize=(12,5))
monthly_sales.plot()
plt.title("Monthly Sales Trend")
plt.ylabel("Total Revenue")
plt.show()

# ============================
# 5. RFM Feature Engineering
# ============================
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "InvoiceNo": "count",
    "TotalPrice": "sum"
})
rfm.columns = ["Recency", "Frequency", "Monetary"]
print("RFM Sample:")
print(rfm.head())

# ============================
# 6. Standardization
# ============================
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# ============================
# 7. Clustering (KMeans)
# ============================
inertia = []
sil_scores = []
K = range(2, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    inertia.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(rfm_scaled, kmeans.labels_))

plt.figure(figsize=(6,4))
plt.plot(K, inertia, marker="o")
plt.title("Elbow Method for KMeans")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.show()

plt.figure(figsize=(6,4))
plt.plot(K, sil_scores, marker="o", color="red")
plt.title("Silhouette Scores")
plt.xlabel("k")
plt.ylabel("Score")
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)
print(rfm.groupby("Cluster").mean())

pickle.dump(kmeans, open("kmeans_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

# ============================
# 8. Recommendation System
# ============================
customer_product = df.pivot_table(
    index="CustomerID",
    columns="Description",
    values="Quantity",
    aggfunc="sum"
).fillna(0)

product_similarity = cosine_similarity(customer_product.T)
product_similarity_df = pd.DataFrame(
    product_similarity,
    index=customer_product.columns,
    columns=customer_product.columns
)
pickle.dump(product_similarity_df, open("product_similarity.pkl", "wb"))

def recommend_products(product_name, top_n=5):
    if product_name not in product_similarity_df.index:
        return ["❌ Product not found in database."]
    sim_scores = product_similarity_df[product_name].sort_values(ascending=False)[1:top_n+1]
    return sim_scores.index.tolist()

print("\nExample Recommendation:")
print(recommend_products("WHITE HANGING HEART T-LIGHT HOLDER"))
