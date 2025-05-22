from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18

class ImageEmbeddingsManager:
    def __init__(self ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet18(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.model.to(self.device)

        # 画像前処理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    
    def image_to_embedding(self, image_path:str)-> list[float]:
        """
        画像のパスを受け取り、埋め込みベクトルを生成する（単一画像用）
        """
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.model(image_tensor).squeeze().cpu().numpy()
            return embedding.tolist()
        except Exception as e:
            
            return None
