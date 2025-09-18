import streamlit as st
import pickle
import numpy as np

# Load models
kmeans = pickle.load(open("kmeans_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
product_similarity = pickle.load(open("product_similarity.pkl", "rb"))

st.set_page_config(page_title="Shopper Spectrum", layout="centered")
st.title("🛒 Shopper Spectrum")
st.write("Customer Segmentation & Product Recommendation in E-Commerce")

# ============================
# Product Recommendation
# ============================
st.header("🎯 Product Recommendation")
product_name = st.text_input("Enter a product name:")
if st.button("Get Recommendations"):
    if product_name not in product_similarity.index:
        st.error("❌ Product not found! Try another one.")
    else:
        sim_scores = product_similarity[product_name].sort_values(ascending=False)[1:6]
        st.success("### Recommended Products:")
        for i, prod in enumerate(sim_scores.index.tolist()):
            st.write(f"{i+1}. {prod}")

# ============================
# Customer Segmentation
# ============================
st.header("🔍 Customer Segmentation")
recency = st.number_input("Recency (days since last purchase)", min_value=0)
frequency = st.number_input("Frequency (#transactions)", min_value=0)
monetary = st.number_input("Monetary (total spend)", min_value=0)

if st.button("Predict Cluster"):
    features = np.array([[recency, frequency, monetary]])
    features_scaled = scaler.transform(features)
    cluster = kmeans.predict(features_scaled)[0]
    cluster_labels = {
        0: "High-Value Customers 🏆",
        1: "Regular Customers 👍",
        2: "Occasional Shoppers 🛍",
        3: "At-Risk Customers ⚠"
    }
    st.success(f"Predicted Customer Segment: **{cluster_labels.get(cluster, 'Unknown')}**")
