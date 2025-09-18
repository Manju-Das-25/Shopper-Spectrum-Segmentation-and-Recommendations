# Shopper-Spectrum-Segmentation-and-Recommendations
Shopper Spectrum

Customer Segmentation and Product Recommendations in E-Commerce

---

## 📌 Overview
This project uses **RFM analysis** and **collaborative filtering** to:
- Segment customers (High-Value, Regular, Occasional, At-Risk)
- Recommend similar products based on purchase history

It includes a **Streamlit app** for real-time predictions and recommendations.

---

## 📂 Files
- `notebook.py` → Data processing, clustering, recommendation model building
- `app.py` → Streamlit app
- `requirements.txt` → Dependencies
- `kmeans_model.pkl`, `scaler.pkl`, `product_similarity.pkl` → Saved models

---

## 🛠 Setup

1. Install dependencies
```bash
pip install -r requirements.txt

python notebook.py

streamlit run app.py
