from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# 検索キーワード
query_text = "Its hypernym is container."

# 埋め込みモデル（学習時と同じものを使う）
model = SentenceTransformer("all-MiniLM-L6-v2")

# クエリをベクトルに変換
query_embedding = model.encode([query_text]).tolist()

client = PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="object_metadata")
# 類似検索の実行
results = collection.query(
    query_embeddings=query_embedding,
    n_results=40,  # 上位5件を取得
    include=["documents", "metadatas", "distances"]
)

# 検索結果の出力
for i, doc in enumerate(results["documents"][0]):
    print(f"\nRank {i+1}")
    print(f"Caption: {doc}")
    print(f"Metadata: {results['metadatas'][0][i]}")
    print(f"Distance: {results['distances'][0][i]:.4f}")