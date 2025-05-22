from chroma_db_manager import ChromaDBManager

from embeddings_manager.sentence_embeddings_manager import SentenceEmbeddingsManager
from embeddings_manager.image_embeddings_manager import ImageEmbeddingsManager

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

from collections import defaultdict
import numpy as np  # get_appropriate_cluster_count で np.argmax を使うために必要

def print_clusters(results, labels, label_name="クラスタ"):
    clusters = defaultdict(list)

    for i, label in enumerate(labels):
        doc = results["documents"][i] if results.get("documents") else "No Document"
        meta = results["metadatas"][i] if results.get("metadatas") else {}
        path = meta.get("path", "No Path")
        clusters[label].append(f"{path} | {doc}")  # 最初の50文字だけ表示

    for cluster_id, items in clusters.items():
        print(f"\n--- {label_name} {cluster_id} ---")
        for item in items:
            print(item)

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

def main():
    # ChromaDBの初期化
    sentence_db_manager = ChromaDBManager("sentence_embeddings")
    image_db_manager = ChromaDBManager("image_embeddings")

    # 埋め込みマネージャーの初期化
    sentence_embeddings_manager = SentenceEmbeddingsManager()
    image_embeddings_manager = ImageEmbeddingsManager()

    # データベースの取得
    results_sentence = sentence_db_manager.get_all()
    appropriate_sentence_cluster_count = get_appropriate_cluster_count(results_sentence["embeddings"])
    
    sentence_clusterer = AgglomerativeClustering(n_clusters=appropriate_sentence_cluster_count, linkage='ward')
    sentence_labels = sentence_clusterer.fit_predict(results_sentence["embeddings"])

    print_clusters(results_sentence, sentence_labels, label_name="Sentence Cluster")

    results_image = image_db_manager.get_all()
    appropriate_image_cluster_count = get_appropriate_cluster_count(results_image["embeddings"])
    
    image_clusterer = AgglomerativeClustering(n_clusters=appropriate_image_cluster_count, linkage='ward')
    image_labels = image_clusterer.fit_predict(results_image["embeddings"])

    # print_clusters(results_image, image_labels, label_name="Image Cluster")
    
if __name__ == "__main__":
    main()

    
    
    
    
    
    

