from chroma_db_manager import ChromaDBManager
import json
import uuid
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from tqdm import tqdm

# JSONデータの読み込み
with open('captions_20250522_013210.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
succeed_data = [item for item in data if item['is_success']]

# ChromaDB クラスで初期化
sentence_db = ChromaDBManager("sentence_embeddings")
image_db = ChromaDBManager("image_embeddings")

# キャプションの登録
sentence_db.add(
    documents=[item["caption"] for item in succeed_data],
    metadatas=[{
        "path": item["path"],
        "is_success": item["is_success"]
    } for item in succeed_data]
)

# ResNet18モデルの初期化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()
model.to(device)

# 画像前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 画像のベクトル化と登録
documents = []  # 不要でも空で渡す必要あり
embeddings = []
metadatas = []

for item in tqdm(succeed_data, desc="画像をベクトル化中"):
    image_path = os.path.join("./imgs", item["path"])
    if not os.path.exists(image_path):
        print(f"[警告] 画像が見つかりません: {image_path}")
        continue
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(image_tensor).squeeze().cpu().numpy()
        embeddings.append(embedding.tolist())
        metadatas.append({
            "path": item["path"],
            "is_success": item["is_success"]
        })
        documents.append("")  # 必須フィールドなので空文字列を入れる
    except Exception as e:
        print(f"[エラー] {image_path} の処理に失敗: {e}")

image_db.add(
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings
)
