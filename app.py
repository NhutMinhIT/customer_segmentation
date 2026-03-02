import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

st.set_page_config(page_title="Customer Segmentation Demo", layout="wide")

st.title("🛍️ Customer Segmentation (Clustering) Demo")
st.caption(
    "Mall Customers dataset • Normalize • Feature Engineering • K-Means vs Hierarchical • Silhouette + Visualization"
)

# -----------------------------
# Sidebar - Input controls
# -----------------------------
st.sidebar.header("⚙️ Cấu hình")

data_path = st.sidebar.text_input("Đường dẫn CSV", value="data/Mall_Customers.csv")

model_type = st.sidebar.selectbox(
    "Chọn model", ["K-Means", "Hierarchical (Agglomerative)"]
)

use_gender = st.sidebar.checkbox("Dùng Gender (encode 0/1)", value=False)
use_feature_ratio = st.sidebar.checkbox(
    "Tạo feature spending_score_per_income", value=True
)

k = st.sidebar.slider("Số cụm (K)", min_value=2, max_value=10, value=5)

auto_search_k = st.sidebar.checkbox("Tự tìm K tốt nhất (silhouette)", value=True)


# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


try:
    df = load_data(data_path)
except Exception as e:
    st.error(f"Không đọc được file CSV. Kiểm tra path nhé. Lỗi: {e}")
    st.stop()

# Standardize columns
df = df.rename(
    columns={"Annual Income (k$)": "income", "Spending Score (1-100)": "spending_score"}
)

if "Gender" in df.columns and use_gender:
    df["gender"] = df["Gender"].map({"Male": 1, "Female": 0})
elif "gender" in df.columns and not use_gender:
    df = df.drop(columns=["gender"], errors="ignore")

if use_feature_ratio:
    df["spending_score_per_income"] = df["spending_score"] / (df["income"] + 1e-6)
else:
    df = df.drop(columns=["spending_score_per_income"], errors="ignore")

# Feature selection
base_features = []
if "Age" in df.columns:
    base_features.append("Age")
base_features += ["income", "spending_score"]
if use_feature_ratio:
    base_features.append("spending_score_per_income")
if use_gender:
    base_features.append("gender")

missing = [c for c in base_features if c not in df.columns]
if missing:
    st.error(f"Thiếu cột cần thiết: {missing}")
    st.stop()

X = df[base_features].copy()

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# -----------------------------
# Utility: search best K
# -----------------------------
def find_best_k_kmeans(Xs, k_min=2, k_max=10):
    best = (None, -1.0, None)
    scores = []
    for kk in range(k_min, k_max + 1):
        m = KMeans(n_clusters=kk, random_state=42, n_init="auto")
        labels = m.fit_predict(Xs)
        s = silhouette_score(Xs, labels)
        scores.append((kk, s))
        if s > best[1]:
            best = (kk, s, labels)
    return best, scores


def find_best_k_hier(Xs, k_min=2, k_max=10):
    best = (None, -1.0, None)
    scores = []
    for kk in range(k_min, k_max + 1):
        m = AgglomerativeClustering(n_clusters=kk, linkage="ward")
        labels = m.fit_predict(Xs)
        s = silhouette_score(Xs, labels)
        scores.append((kk, s))
        if s > best[1]:
            best = (kk, s, labels)
    return best, scores


# -----------------------------
# Train + Evaluate
# -----------------------------
scores_list = None

if auto_search_k:
    if model_type == "K-Means":
        (best_k, best_s, best_labels), scores_list = find_best_k_kmeans(X_scaled, 2, 10)
    else:
        (best_k, best_s, best_labels), scores_list = find_best_k_hier(X_scaled, 2, 10)
    k_used = best_k
    labels = best_labels
else:
    k_used = k
    if model_type == "K-Means":
        model = KMeans(n_clusters=k_used, random_state=42, n_init="auto")
        labels = model.fit_predict(X_scaled)
    else:
        model = AgglomerativeClustering(n_clusters=k_used, linkage="ward")
        labels = model.fit_predict(X_scaled)
    best_s = silhouette_score(X_scaled, labels)

df["cluster"] = labels

# -----------------------------
# Main layout
# -----------------------------
colA, colB = st.columns([1.2, 1])

with colA:
    st.subheader("📌 Tổng quan")
    st.write("**Features dùng để clustering:**", base_features)
    st.write(f"**Model:** {model_type}  |  **K sử dụng:** {k_used}")
    st.metric("Silhouette score", f"{best_s:.4f}")

    st.markdown("### 📊 Cluster profile (mean)")
    profile = df.groupby("cluster")[base_features].mean().round(2)
    st.dataframe(profile, width='stretch')

with colB:
    st.subheader("🔎 Silhouette theo K")
    if scores_list:
        chart_df = pd.DataFrame(scores_list, columns=["K", "Silhouette"])
        st.line_chart(chart_df.set_index("K"))
    else:
        st.info("Bật 'Tự tìm K tốt nhất' để xem biểu đồ silhouette theo K.")

st.divider()

# PCA plot
st.subheader("🧭 Visualization (PCA 2D)")

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)

fig = plt.figure()
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"{model_type} - PCA 2D (K={k_used})")
st.pyplot(fig)

# Dendrogram for hierarchical
if model_type.startswith("Hierarchical"):
    st.subheader("🌲 Dendrogram (Ward linkage)")
    Z = linkage(X_scaled, method="ward")
    fig2 = plt.figure(figsize=(10, 4))
    dendrogram(Z, truncate_mode="lastp", p=12)
    plt.xlabel("Cluster size (truncated)")
    plt.ylabel("Distance")
    st.pyplot(fig2)

st.divider()

st.subheader("🗂️ Data preview")
st.dataframe(df.head(20), width='stretch')
