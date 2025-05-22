import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import os
import shutil

# --- ChromaDBからデータ取得 ---
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="sentence_embeddings")
results = collection.get(include=["documents", "metadatas"])
captions = results["documents"]
metadatas = results["metadatas"]

def get_appropriate_cluster_count(embeddings, min_clusters=5, max_clusters=30):
    """
    シルエットスコアを用いて適切なクラスタ数を決定する関数
    """
    range_n_clusters = range(min_clusters, max_clusters + 1)
    scores = []

    for n in range_n_clusters:
        clusterer = AgglomerativeClustering(n_clusters=n, linkage='ward')
        labels = clusterer.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        scores.append(score)

    best_n_clusters = range_n_clusters[np.argmax(scores)]
    return best_n_clusters

# --- 文章の埋め込み ---
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(captions, show_progress_bar=True)

# --- シルエットスコアで最適なクラスタ数を推定 ---
range_n_clusters = range(6, 30)
scores = []

# --- 最適なクラスタ数を決定 ---
best_n_clusters = get_appropriate_cluster_count(embeddings, min_clusters=6, max_clusters=30)
print(f"\n選択されたクラスタ数: {best_n_clusters}")

# --- 最終クラスタリング ---
clusterer = AgglomerativeClustering(n_clusters=best_n_clusters, linkage='ward')
labels = clusterer.fit_predict(embeddings)

for i, label in enumerate(labels):
    print(f"Index: {i}, Label: {label}, Metadata: {metadatas[i]}")

# --- PCAで次元削減（3次元） ---
pca = PCA(n_components=3)
reduced = pca.fit_transform(embeddings)

# --- カラーマップ作成 ---
unique_labels = np.unique(labels)
palette = sns.color_palette("tab10", len(unique_labels))
colors = np.array([palette[label] for label in labels])

# --- 3Dプロット表示 ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], c=colors, s=20, alpha=0.8)

for i, metadata in enumerate(metadatas):
    ax.text(reduced[i, 0], reduced[i, 1], reduced[i, 2], str(i), fontsize=1)

ax.set_title("Sentence Embedding Clusters (Agglomerative Clustering)")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")
plt.tight_layout()
plt.show()

# --- 各クラスタの凝集度を計算（平均距離） ---
cluster_compactness = {}
for label in unique_labels:
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

# --- クラスタごとに ./result/{cluster_id}/ に画像をコピー ---
output_base = "./result"

for i, label in enumerate(labels):
    image_path = f"./imgs/{metadatas[i].get('path')}"
    if image_path is None or not os.path.exists(image_path):
        print(f"[警告] 画像が見つかりません: {image_path}")
        continue

    output_dir = os.path.join(output_base, str(label))
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{i}_{os.path.basename(image_path)}"
    output_path = os.path.join(output_dir, filename)
    shutil.copy(image_path, output_path)

print("画像の分類コピーが完了しました。")