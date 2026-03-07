import os
import faiss
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

IMAGE_ROOT = "/your/image/root/path"  # Update this to your actual image root path same as in config.yaml (config.data.ydata_path)

class Memory:
    def __init__(self, model_name=None, device="cuda", model=None, processor=None):
        self.device = device

        if model is not None and processor is not None:
            self.model = model
            self.processor = processor
        else:
            if model_name is None:
                raise ValueError("Either (model_name) or (model, processor) must be provided.")
            self.model = CLIPModel.from_pretrained(model_name).to(device)
            self.processor = CLIPProcessor.from_pretrained(model_name)

        self.model.eval()

        dim = self.model.config.projection_dim

        self.text_index = faiss.IndexFlatL2(dim)
        self.image_index = faiss.IndexFlatL2(dim)

        self.text_memory_data = []
        self.image_memory_data = []


    def encode_text(self, text_list):
        inputs = self.processor(
            text=text_list, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
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
        for mem in memories:
            text_tokens = mem.get("object_text", None)
            if text_tokens:
                text_feat = self.encode_text([text_tokens]).flatten()
                self.text_index.add(text_feat.reshape(1, -1))
                self.text_memory_data.append(mem)

            img_obj = mem.get("img", None)
            if img_obj:
                img_feat = self.encode_image(img_obj).flatten()
                self.image_index.add(img_feat.reshape(1, -1))
                self.image_memory_data.append(mem)
                
    def query(self, query_text, top_k=5):
        query_vector = self.encode_text([query_text])  # shape: (1, dim)

        if len(self.text_memory_data) > 0:
            text_dists, text_indices = self.text_index.search(query_vector, top_k)
            text_indices = [self.text_memory_data[i] for i in text_indices[0] if i != -1]
        else:
            text_indices = []

        if len(self.image_memory_data) > 0:
            image_dists, image_indices = self.image_index.search(query_vector, top_k)
            image_indices = [self.image_memory_data[i] for i in image_indices[0] if i != -1]
        else:
            image_indices = []

        results = {id(mem): mem for mem in text_indices + image_indices}
        return list(results.values())

    def clear(self):
        self.text_index.reset()
        self.image_index.reset()
        self.text_memory_data = []
        self.image_memory_data = []

_global_memory_tool = None

def get_memory_tool():
    global _global_memory_tool
    if _global_memory_tool is None:
        model_name = "openai/clip-vit-large-patch14"
        _global_memory_tool = Memory(model_name=model_name)
    return _global_memory_tool

def retrieve_memories(memory, query_text=None):
    top_k = 3
    image_root = IMAGE_ROOT
    resize_size = (360, 360)
    base_memory_tool = get_memory_tool()
    memory_tool = Memory.__new__(Memory)
    memory_tool.device = base_memory_tool.device
    memory_tool.model = base_memory_tool.model
    memory_tool.processor = base_memory_tool.processor

    dim = memory_tool.model.config.projection_dim
    memory_tool.text_index = faiss.IndexFlatL2(dim)
    memory_tool.image_index = faiss.IndexFlatL2(dim)
    memory_tool.text_memory_data = []
    memory_tool.image_memory_data = []


    all_memories = []
    for mem in memory:
        mem_copy = mem.copy()
        img_path = mem.get("img_path")
        if img_path:
            full_path = os.path.join(image_root, img_path)
            if os.path.exists(full_path):
                with Image.open(full_path) as img:
                    img = img.convert("RGB")
                    img = img.resize(resize_size, Image.BILINEAR)
                    mem_copy["img"] = img.copy()
        mem_copy["object_text"] = str(mem.get("object"))
        all_memories.append(mem_copy)

    memory_tool.clear()
    memory_tool.build_memory(all_memories)

    results = memory_tool.query(query_text, top_k=top_k)
    return [{"img": r.get("img"), "objects": r.get("object_text")} for r in results]
