import faiss
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class Memory:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda"):
        """
        使用 HuggingFace Transformers 的 CLIPModel
        """
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        # CLIP 输出向量维度
        text_dim = self.model.config.projection_dim
        img_dim = self.model.config.projection_dim
        self.dim = text_dim + img_dim

        self.index = faiss.IndexFlatL2(self.dim)
        self.memory_data = []  # 存储 memory 原始信息和向量

    def encode_text(self, text_list):
        inputs = self.processor(text=text_list, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()

    def encode_image(self, img: Image.Image):
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()

    def build_memory(self, memories):
        """
        memories: List[Dict]
        每条 memory 包含：
            - "img": PIL.Image.Image 对象（可选）
            - "object_text": 文本描述
        """
        vectors = []
        for mem in memories:
            # image feature
            img_obj = mem.get("img", None)
            if img_obj is not None:
                img_feat = self.encode_image(img_obj).flatten()
            else:
                img_feat = np.zeros(self.model.config.projection_dim, dtype=np.float32)

            # text/object feature
            text_tokens = mem.get("object_text", None)
            if text_tokens is not None:
                text_feat = self.encode_text([text_tokens]).flatten()
            else:
                text_feat = np.zeros(self.model.config.projection_dim, dtype=np.float32)

            # 合并向量
            combined = np.concatenate([img_feat, text_feat])
            vectors.append(combined)
            self.memory_data.append(mem)

        vectors = np.stack(vectors).astype(np.float32)
        self.index.add(vectors)

    def query(self, query_text=None, query_image=None, top_k=5):
        """
        query_text: str
        query_image: PIL.Image.Image 对象
        返回的结果里直接包含 PIL.Image.Image
        """
        img_feat = np.zeros(self.model.config.projection_dim, dtype=np.float32)
        text_feat = np.zeros(self.model.config.projection_dim, dtype=np.float32)

        if query_text:
            text_feat = self.encode_text([query_text]).flatten()
        if query_image:
            img_feat = self.encode_image(query_image).flatten()

        query_vec = np.concatenate([img_feat, text_feat]).astype(np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_vec, top_k)
        results = [self.memory_data[i] for i in indices[0] if i != -1]
        return results, distances[0]

    def clear(self):
        self.index.reset()
        self.memory_data = []
