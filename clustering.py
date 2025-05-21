import chromadb
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np
from scipy.spatial.distance import pdist

# --- ChromaDBからデータ取得 ---
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="object_metadata")
results = collection.get(include=["documents", "metadatas"])
captions = results["documents"]
metadatas = results["metadatas"]

# --- 文章の埋め込み ---
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(captions, show_progress_bar=True)

# --- クラスタリング ---
clusterer = hdbscan.HDBSCAN(
    metric='euclidean',
    min_cluster_size=10,
    min_samples=10,
    cluster_selection_method='eom',
    alpha=1.0
)
labels = clusterer.fit_predict(embeddings)

# --- 次元削減 ---
pca = PCA(n_components=3)
reduced = pca.fit_transform(embeddings)

# --- カラーマップをクラスタ数に応じて作成 ---
unique_labels = np.unique(labels)
n_clusters = len(unique_labels[unique_labels >= 0])
palette = sns.color_palette("tab10", n_clusters)
colors = np.array([palette[label] if label >= 0 else (0.5, 0.5, 0.5) for label in labels])  # -1はノイズ

# --- 3Dプロット ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    reduced[:, 0],
    reduced[:, 1],
    reduced[:, 2],
    c=colors,
    s=20,
    alpha=0.8
)

# 注釈（必要に応じて調整）
for i, metadata in enumerate(metadatas):
    ax.text(reduced[i, 0], reduced[i, 1], reduced[i, 2], str(i), fontsize=1)

ax.set_title("Sentence Embedding Clusters (HDBSCAN)")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")

plt.tight_layout()
plt.show()

# --- 各クラスタの凝集度を計算（平均距離） ---
cluster_compactness = {}
for label in unique_labels:
    if label == -1:
        continue  # ノイズ除外
    cluster_embeddings = embeddings[labels == label]
    if len(cluster_embeddings) < 2:
        continue
    avg_distance = np.mean(pdist(cluster_embeddings))
    cluster_compactness[label] = avg_distance

# --- 最も凝集度が高いクラスタを特定 ---
most_compact_label = min(cluster_compactness, key=cluster_compactness.get)
print(f"\n最も凝集度が高いクラスタ: {most_compact_label}")
print(f"平均距離: {cluster_compactness[most_compact_label]:.4f}")

# --- 該当クラスタの画像パスを出力 ---
print("\n該当クラスタの画像パス:")
for i, label in enumerate(labels):
    if label == most_compact_label:
        print(metadatas[i].get("path", f"(index {i} に 'path' が見つかりません)"))