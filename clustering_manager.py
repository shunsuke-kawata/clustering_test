from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from chroma_db_manager import ChromaDBManager
import uuid
import re

class InitClustering:
    
    def __init__(self, chroma_db: ChromaDBManager, images_folder_path: str, output_base_path: str = './results'):
        def _is_valid_path(path: str) -> bool:
            if not isinstance(path, str) or not path.strip():
                return False

            # 条件 3: 必ず ./ ../ / のいずれかで始まる
            if not (path.startswith("./") or path.startswith("../") or path.startswith("/")):
                return False

            # 条件 4: 最後は / で終わってはいけない
            if path.endswith("/"):
                return False

            # 条件 5: 危険文字の除外
            if re.search(r'[<>:"|?*]', path):
                return False

            return True

        if not (_is_valid_path(images_folder_path) and _is_valid_path(output_base_path)):
            raise ValueError(f" Error Folder Path: {images_folder_path}")
        
        self._chroma_db = chroma_db
        self._images_folder_path = Path(images_folder_path)
        self._output_base_path = Path(output_base_path)
    
    @property
    def chroma_db(self) -> ChromaDBManager:
        return self._chroma_db

    @property
    def images_folder_path(self) -> Path:
        return self._images_folder_path
    
    @property
    def output_base_path(self) -> Path:
        return self._output_base_path
    
    def get_optimal_cluster_num(self, embeddings: list[float], min_cluster_num: int = 5, max_cluster_num: int = 30) -> tuple[int, float]:
        embeddings_np = np.array(embeddings)
        best_score = -1
        best_k = min_cluster_num
        scores = []

        for k in range(min_cluster_num, max_cluster_num + 1):
            model = AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(embeddings_np)

            if len(set(labels)) == 1:
                continue 

            score = silhouette_score(embeddings_np, labels)
            scores.append((k, score))

            if score > best_score:
                best_score = score
                best_k = k

        return best_k, float(best_score)
    
    def clustering(self, chroma_db_data: dict[str, list], cluster_num: int, output: bool = False):
        embeddings_np = np.array(chroma_db_data['embeddings'])
        result_uuids_dict = {
            i: {'folder_id': str(uuid.uuid4()), 'ids': []}
            for i in range(cluster_num)
        }

        model = AgglomerativeClustering(n_clusters=cluster_num)
        labels = model.fit_predict(embeddings_np)

        if output:
            if self._output_base_path.exists():
                shutil.rmtree(self._output_base_path)
            self._output_base_path.mkdir(parents=True, exist_ok=True)

            for cluster_id, info in result_uuids_dict.items():
                (self._output_base_path / info['folder_id']).mkdir(parents=True, exist_ok=True)

        for i, label in enumerate(labels):
            result_uuids_dict[label]['ids'].append(chroma_db_data['ids'][i])

            if output:
                image_name = chroma_db_data['metadatas'][i].get('path')
                origin_image_path = self._images_folder_path / image_name

                if not origin_image_path.exists():
                    print(f"[警告] 画像が見つかりません: {origin_image_path}")
                    continue

                output_dir = self._output_base_path / result_uuids_dict[label]['folder_id']
                filename = f"{i}_{origin_image_path.name}"
                shutil.copy(origin_image_path, output_dir / filename)

        return result_uuids_dict

if __name__ == "__main__":
    cl_module = InitClustering(
        chroma_db=ChromaDBManager('sentence_embeddings'),
        images_folder_path='./imgs',
        output_base_path='./results'
    )
    
    all_sentence_data = cl_module.chroma_db.get_all()
    cluster_num, _ = cl_module.get_optimal_cluster_num(embeddings=all_sentence_data['embeddings'])
    cluster_result = cl_module.clustering(chroma_db_data=all_sentence_data, cluster_num=cluster_num, output=True)