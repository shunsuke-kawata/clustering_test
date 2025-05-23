import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from chroma_db_manager import ChromaDBManager

def find_optimal_clusters(embeddings: np.ndarray, min_clusters: int = 6, max_clusters: int = 30):
    best_score = -1
    best_k = min_clusters
    scores = []

    print("Evaluating silhouette scores for different cluster sizes...")

    for k in range(min_clusters, max_clusters + 1):
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(embeddings)

        if len(set(labels)) == 1:
            continue  # silhouette_score fails if there's only one cluster

        score = silhouette_score(embeddings, labels)
        scores.append((k, score))

        print(f"Clusters: {k}, Silhouette Score: {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k

    return best_k, scores

def cluster_images_with_optimal_k():
    # ChromaDBから文ベクトルを取得
    manager = ChromaDBManager("sentence_embeddings")
    all_data = manager.get_all()

    embeddings = all_data["embeddings"]
    metadatas = all_data["metadatas"]

    if len(embeddings) == 0 or len(metadatas) == 0:
        print("No data available.")
        return

    X = np.array(embeddings)

    # 最適なクラスタ数を探索
    best_k, score_list = find_optimal_clusters(X, min_clusters=6, max_clusters=30)
    print(f"\nOptimal number of clusters: {best_k}")

    # 最終クラスタリング
    model = AgglomerativeClustering(n_clusters=best_k)
    labels = model.fit_predict(X)

    # 画像をフォルダに分類
    output_base = "./clusters_agglo"
    os.makedirs(output_base, exist_ok=True)

    for i, label in enumerate(labels):
        cluster_dir = os.path.join(output_base, f"cluster_{label}")
        os.makedirs(cluster_dir, exist_ok=True)

        path = metadatas[i]["path"]
        src = f"./imgs/{path}"
        dst = os.path.join(cluster_dir, os.path.basename(path))

        if os.path.exists(src):
            shutil.copy(src, dst)

    print("Images clustered and copied successfully.")

if __name__ == "__main__":
    cluster_images_with_optimal_k()