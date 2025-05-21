import json
import chromadb
from chromadb.config import Settings
import uuid

# JSONファイルからデータを読み込む
with open('captions_20250522_013210.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


# キャプションを抽出して埋め込みを作成
succeed_data = [item for item in data if item['is_success']]


# ChromaDBのローカルクライアントを初期化
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="object_metadata")


collection.upsert(
    ids=[str(uuid.uuid4()) for _ in range(len(succeed_data))],  # 一意なIDが必要
    metadatas=[{
        "path": datum["path"],
        "is_success": datum["is_success"]
    } for datum in succeed_data],
    documents=[datum["caption"] for datum in succeed_data],
)
