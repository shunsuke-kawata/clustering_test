import json
import os
import uuid
import chromadb
from embeddings_manager.sentence_embeddings_manager import SentenceEmbeddingsManager
from embeddings_manager.image_embeddings_manager import ImageEmbeddingsManager

class ChromaDBManager:
    def __init__(self, colection_name:str,path="./chroma_db"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=colection_name)

    def add(self, documents:list[str], metadatas:list[dict],embeddings:list[list[float]] = None):
        ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        kwargs = {"ids": ids, "documents": documents, "metadatas": metadatas}
        if embeddings:
            kwargs["embeddings"] = embeddings
        self.collection.upsert(**kwargs)

    def get_all(self):
        results = self.collection.get(include=["documents", "metadatas","embeddings"],limit=None)
        return results
    
    def update(self, ids:list[str], documents:list[str], metadatas:list[dict],embeddings:list[list[float]] = None):

        kwargs = {"ids": ids, "documents": documents, "metadatas": metadatas}
        if embeddings:
            kwargs["embeddings"] = embeddings
        self.collection.upsert(**kwargs)
    
    def delete(self, ids:list[str]):
        self.collection.delete(ids=ids)

    #埋め込み表現でクエリを発行する
    def query_by_embeddings(self, query_embeddings:list[list[float]], n_results:int = 10):
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=["documents", "metadatas", "distances","embeddings"]
        )
        return results

if __name__ == "__main__":
    
    image_embeddings_manager = ImageEmbeddingsManager()
    sentence_embeddings_manager = SentenceEmbeddingsManager()
    # Example usage
    sentence_db_manager = ChromaDBManager("sentence_embeddings")
    
    with open('captions_20250522_013210.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    succeed_data = [item for item in data if item['is_success']]
    
    sentence_db_manager.add(
        documents=[item["caption"] for item in succeed_data],
        embeddings=[sentence_embeddings_manager.sentence_to_embedding(item["caption"]) for item in succeed_data],
        metadatas=[{
            "path": item["path"],
            "is_success": item["is_success"]
        } for item in succeed_data]
    )
    
    image_db_manager = ChromaDBManager("image_embeddings")
    image_db_manager.add(
        documents=[item["caption"] for item in succeed_data],
        embeddings=[sentence_embeddings_manager.sentence_to_embedding(f"./imgs/{item['path']}") for item in succeed_data],
        metadatas=[{
            "path": item["path"],
            "is_success": item["is_success"]
        } for item in succeed_data]
    )
    
    