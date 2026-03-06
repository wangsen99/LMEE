from PIL import Image
import numpy as np
import base64
from io import BytesIO
import os
import time
import logging
import torch
from typing import Optional
import warnings
from qwen_vl_utils import process_vision_info
import open_clip
import faiss
import re
import difflib
import random
warnings.filterwarnings("ignore")

def l2norm(v):
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-10)

class KnowledgeBase:
    def __init__(self, device="cuda"):
        self.text_embedder, self.image_model, self.preprocess = None, None, None
        text = "openai/clip-vit-large-patch14"
        visual = "openai/clip-vit-large-patch14"
        text_model_name = text.split("/")[-1]
        visual_model_name = visual.split("/")[-1]
        if text_model_name == visual_model_name and visual_model_name in ["clip-vit-large-patch14"]:
            from transformers import CLIPProcessor, CLIPModel
            self.text_embedder = self.image_model = CLIPModel.from_pretrained(text).to(device)
            self.preprocess = CLIPProcessor.from_pretrained(text)

        self.text_index = faiss.IndexFlatL2(768)
        self.image_index = faiss.IndexFlatL2(768)
        self.data = []
        self.device = device


    def add_to_knowledge_base(self, text=None, image=None, name=None):
        text_vector = image_vector = None
        if text:
            text_inputs = self.preprocess(
                text=text, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)

            text_vector = self.text_embedder.get_text_features(**text_inputs)
            text_vector = text_vector.cpu().detach().numpy()
            text_vector = l2norm(text_vector)

            self.text_index.add(text_vector)

        if image is not None and image.size > 0:
            image_inputs = self.preprocess(images=image, return_tensors="pt").to(self.device)

            image_vector = self.image_model.get_image_features(**image_inputs)
            image_vector = image_vector.cpu().detach().numpy()
            image_vector = l2norm(image_vector)

            self.image_index.add(image_vector)

        self.data.append({
            "rgb_id": name,
            "image": image,
            "text": text,
            "text_vector": text_vector,
            "image_vector": image_vector
        })

    def search_text(self, query_text, top_k=5):
        text_inputs = self.preprocess(
            text=query_text, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        query_vector = self.text_embedder.get_text_features(**text_inputs)
        query_vector = query_vector.cpu().detach().numpy()

        query_vector = l2norm(query_vector)

        image_dists, image_indices = self.image_index.search(query_vector, top_k)
        image_indices = image_indices[0]

        results = [self.data[i] for i in image_indices if i != -1]
        return results, image_indices.tolist()


    def search_text_to_pair(self, query_text, top_k=5):
        text_inputs = self.preprocess(
            text=query_text, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        query_vector = self.text_embedder.get_text_features(**text_inputs)
        query_vector = query_vector.cpu().detach().numpy()

        query_vector = l2norm(query_vector)

        text_dists, text_indices = self.text_index.search(query_vector, top_k)
        image_dists, image_indices = self.image_index.search(query_vector, top_k)

        text_results = [self.data[i] for i in text_indices[0] if i != -1]
        image_results = [self.data[i] for i in image_indices[0] if i != -1]

        all_results = {r["rgb_id"]: r for r in text_results + image_results}
        return list(all_results.values()), list(all_results.keys())


    def search_image_to_pair(self, query_text, query_image, top_k=5):

        # ---- Image → normalize ----
        image_inputs = self.preprocess(images=query_image, return_tensors="pt").to(self.device)
        query_vector_img = self.image_model.get_image_features(**image_inputs)
        query_vector_img = query_vector_img.cpu().detach().numpy()
        query_vector_img = l2norm(query_vector_img)

        # ---- Text → normalize ----
        text_inputs = self.preprocess(
            text=query_text, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        query_vector_txt = self.text_embedder.get_text_features(**text_inputs)
        query_vector_txt = query_vector_txt.cpu().detach().numpy()
        query_vector_txt = l2norm(query_vector_txt)

        image_dists, image_indices = self.image_index.search(query_vector_img, top_k)
        text_dists, text_indices = self.text_index.search(query_vector_txt, top_k)

        text_results = [self.data[i] for i in text_indices[0] if i != -1]
        image_results = [self.data[i] for i in image_indices[0] if i != -1]

        all_results = {r["rgb_id"]: r for r in text_results + image_results}
        return list(all_results.values()), list(all_results.keys())


    def clear(self):
        self.index.reset()
        self.data = []
        
def call_qwen25(sys_prompt, contents, model, processor):
    """
    contents: list of (text, optional image_base64)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    messages = []

    if sys_prompt is not None and len(sys_prompt.strip()) > 0:
        messages.append({"role": "system", "content": [{"type": "text", "text": sys_prompt}]})

    for c in contents:
        if len(c) == 2:
            messages.append({"role": "user", "content":[{"type":"text","text":c[0]},{"type":"image","image":f"data:image;base64,{c[1]}"}]})
        else:
            messages.append({"role": "user", "content":[{"type":"text","text":c[0]}]})

    # Prepare text and images
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    # Generate output
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
    # Trim prompt
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]

# encode tensor images to base64 format
def encode_tensor2base64(img):
    img = Image.fromarray(img)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return img_base64

def format_explore_prompt_object(
    question,
    egocentric_imgs,
    frontier_imgs,
    snapshot_imgs,
    snapshot_classes,
    snapshot_crops,
    egocentric_view=False,
    use_snapshot_class=True,
    image_goal=None,
):
    # ---------------- System Prompt ----------------
    sys_prompt = (
        "You are an embodied agent exploring an indoor environment.\n"
        "Your goal is to analyze previously visited areas (Memories) and unexplored regions (Frontiers) "
        "to determine the best next step to locate or find the target object described in the task instruction.\n"
        "RULES:\n"
        "1) Your whole reply MUST contain exactly two sections in this order:\n"
        "  A THOUGHT section that begins with 'THOUGHT:' (uppercase) followed by your reasoning.\n"
        "  A single ANSWER line that begins with 'ANSWER:' (uppercase) and provides exactly ONE decision.\n"
        "2) THOUGHT content rules:\n"
        "  For each Memory i, include a short analysis of each Object j.\n"
        "  For each Frontier i, describe its spatial direction or likelihood.\n"
        "3) The ANSWER line MUST be one of these exact formats:\n"
        "  ANSWER: Memory i, Object j\n"
        "  ANSWER: Frontier i\n"
        "  where i and j are plain integers.\n"
        "  IMPORTANT: When choosing a Memory, you MUST specify both the memory index and the object index.\n"
        "4) Do NOT invent or hallucinate new object indices.\n"
        "5) After analyzing all options, select the most promising Memory Object, or the Frontier that maximizes discovery.\n"
    )

    content = []

    # ---------------- Definitions ----------------
    definitions = (
        "Definitions:\n"
        "Memory: A focused observation of several objects. It contains a full image of the cluster of objects, and separate image crops of each object. "
        + "Choosing a memory means that the object asked in the task is within the cluster of objects that the memory represents, and you will choose that object as the result of task instruction. "
        + "Therefore, if you choose a memory, you should also choose the object in the memory that you think is the result of task completion.\n"
        + "Frontier: An unexplored region that could provide new information or a closer view of the target object. Selecting a frontier means that you will further explore that direction.\n"
        + "Indices start at 0. Always reference indices. Do NOT write names instead of indices.\n"
    )
    content.append((definitions,))

    # ---------------- Example 1: choose Frontier ----------------
    example1 = (
        "Example 1 — choose a Frontier:\n"
        "Task Instruction: Find the red kettle and bring it to the kitchen counter.\n"
        "Memories available:\n"
        "Memory 0: [living room with sofa, table, lamp]\n"
        "  Object 0: sofa\n"
        "  Object 1: table\n"
        "Memory 1: [partial kitchen with counter and a blue kettle]\n"
        "  Object 0: counter\n"
        "  Object 1: kettle (blue)\n"
        "Frontiers available:\n"
        "Frontier 0: doorway leading deeper into kitchen\n"
        "Frontier 1: corridor to another room\n"
        "THOUGHT:\n"
        "- Memory 0: contains sofa, table, lamp — none match the target 'red kettle'. Ignored.\n"
        "- Memory 1: contains counter and blue kettle — color mismatch, not the target.\n"
        "- Frontier 0: leads deeper into the kitchen, likely toward locations where a red kettle could be found.\n"
        "- Frontier 1: leads to another room, less relevant.\n"
        "Decision: Frontier 0 maximizes the chance to find the target object.\n"
        "ANSWER: Frontier 0\n"
    )
    content.append((example1,))

    # ---------------- Example 2: choose Memory object ----------------
    example2 = (
        "Example 2 — choose a Memory object (correct strict format):\n"
        "Task Instruction: Locate the refrigerator in the environment.\n"
        "Memories available:\n"
        "Memory 0: [kitchen with large white refrigerator beside a counter]\n"
        "  Object 0: counter (wooden, brown)\n"
        "  Object 1: refrigerator (white, tall, metallic handle)\n"
        "Memory 1: [living room with sofa and TV, no refrigerator visible]\n"
        "  Object 0: sofa (gray)\n"
        "  Object 1: television (black)\n"
        "Frontiers available:\n"
        "Frontier 0: hallway leading to the living room\n"
        "Frontier 1: door leading outside\n"
        "THOUGHT:\n"
        "- Memory 0: Object 1 matches the refrigerator — strong match.\n"
        "- Memory 1: contains sofa and TV — not relevant.\n"
        "- Frontier 0: leads to living room already explored.\n"
        "- Frontier 1: leads outside, irrelevant.\n"
        "Decision: Memory 0, Object 1 is the correct match.\n"
        "ANSWER: Memory 0, Object 1\n"

    )
    content.append((example2,))

    # ---------------- Actual Task ----------------
    task_text = f"Task Instruction: {question}"
    if image_goal is not None:
        content.append((f"{task_text}\nImage goal:" , image_goal))
        content.append(("\n",))
    else:
        content.append((task_text + "\n",))
    content.append(("Select the Memory or Frontier that would help complete the task.\n",))

    # ----------- Memories Section -----------
    content.append(("The followings are all the Memories that you can choose:\n",))
    if len(snapshot_imgs) == 0:
        content.append(("No Memory is available.\n",))
    else:
        for i, rgb_id in enumerate(snapshot_imgs.keys()):
            content.append((f"Memory {i}: ", snapshot_imgs[rgb_id]))
            for j, obj_name in enumerate(snapshot_classes[rgb_id]):
                content.append((f"Object {j}: {obj_name} ", snapshot_crops[rgb_id][j]))
                content.append(("\n",))
            content.append(("\n",))

    # ----------- Frontiers Section -----------
    content.append(("The followings are all the Frontiers that you can explore:\n",))
    if len(frontier_imgs) == 0:
        content.append(("No Frontier is available.\n",))
    else:
        for i, img in enumerate(frontier_imgs):
            content.append((f"Frontier {i}: ", img))
            content.append(("\n",))

    # ----------- Final Instruction -----------
    final_text = (
        "You MUST output the THOUGHT section followed by ONE valid ANSWER line:\n"
        "   ANSWER: Memory i, Object j  (when selecting a specific detected object)\n"
        "   ANSWER: Frontier i  (when deciding to explore a new region)\n"
    )
    content.append((final_text,))

    return sys_prompt, content

def format_explore_prompt_qa(
    question,
    egocentric_imgs,
    snapshot_imgs,
    snapshot_classes,
    egocentric_view=False,
    use_snapshot_class=True,
    image_goal=None,
    answer_type=None,
):
    sys_prompt = (
        "Task: You are an agent in an indoor scene tasked with answering questions by observing the surroundings and recalling memories.\n"
    )
    sys_prompt += (
        "Definitions:\n"
        "Memory: A focused observation of several objects. Choosing a Memory means that this memory image contains enough information for you to answer the question. "
        "If you choose a Memory, you need to directly give an answer to the question. "
        "If you don't have enough information to give an answer, then state that the information is insufficient.\n"
    )

    content = []

    text = f"Question: {question}"
    if image_goal is not None:
        content.append((text, image_goal))
        content.append(("\n",))
    else:
        content.append((text + "\n",))

    text = "Select the Memory that would help you answer the question.\n"
    content.append((text,))

    if egocentric_view:
        text = "The following is the egocentric view of the agent in forward direction: "
        content.append((text, egocentric_imgs[-1]))
        content.append(("\n",))

    text = "The followings are all the Memories that you can choose (followed with contained object classes):\n"
    text += (
        "Please note that the contained classes may not be accurate (wrong or missing classes) due to the limitation of the object detection model. "
        "So you still need to utilize the images to make decisions.\n"
    )
    content.append((text,))

    if len(snapshot_imgs) == 0:
        content.append(("No Memory is available.\n",))
    else:
        for i, rgb_id in enumerate(snapshot_imgs.keys()):
            content.append((f"Memory {i}: ", snapshot_imgs[rgb_id]))
            if use_snapshot_class and rgb_id in snapshot_classes:
                objects_str = ", ".join(snapshot_classes[rgb_id])
                content.append((f"Objects: {objects_str}\n",))

    if answer_type == "choice":
        text = (
            "Please provide your answer in the following format: "
            "'Memory i\n[Answer]', where i is the index of the memory you choose. "
            "[Answer] must be one of the options provided in the question (use the letter corresponding to your choice, e.g., A, B, C, D).\n"
        )
    else:
        text = (
            "Please provide your answer in the following format: "
            "'Memory i\n[Answer]', where i is the index of the memory you choose. "
            "For instance, if you choose the first memory, you can return "
            "'Memory 0\nThe fruit bowl is on the kitchen counter.'.\n"
            "Note that when answering, (1) you should give a direct and natural answer that can be understood by others, "
            "and avoid phrases like 'in this memory' or 'on the left of the image'; "
            "(2) you can refer to all provided memories to reason, but you must select one most relevant memory to base your answer on.\n"
        )
    content.append((text,))

    return sys_prompt, content


def get_retrieval_query_from_llm(question, model, processor, frontier_imgs=None, image_goal=None):
    sys_prompt = (
        "You are an intelligent embodied navigation assistant with a visual memory system.\n"
        "Your goal is to generate a retrieval query that can be used to search a memory database of past visual observations.\n"
        "You will receive a task instruction and possibly one or more images (e.g., frontier images or goal images).\n\n"
        "Guidelines for writing the retrieval query:\n"
        "1. The query must be *specific* — mention concrete object names, shapes, spatial relations, or room types.\n"
        "2. Focus on the information *most relevant for immediate navigation or memory lookup* — e.g., which region to check, which object to find.\n"
        "3. Avoid vague or generic phrases like 'find it', 'the target', or 'search around'.\n"
        "4. Output must strictly follow the format below:\n"
        "THOUGHT: concise reasoning about how you derive the query\n"
        "QUERY: one short, specific retrieval phrase suitable for searching the memory\n"
    )

    content = []
    # ------------------format an example-------------------------
    example = (
        "Here is an example:\n"
        "Task Instruction: Find the mug on the kitchen counter and bring it to the dining table.\n"
        "Frontier observations are available. Consider them implicitly in your reasoning.\n"
        "THOUGHT: The task involves a mug and a kitchen counter. To retrieve relevant memory, I will focus on previous observations showing a mug specifically on a kitchen counter.\n"
        "QUERY: mug on kitchen counter\n\n"

        "Example for task with an image:\n"
        "Task Instruction: Find the object highlighted in the given image.\n"
        "Image shows a small kettle on the left side of a wooden kitchen counter.\n"
        "THOUGHT: To find this object in memory, I should describe it using its type, and position in the scene. I will include that it is a kettle on the left side of a wooden kitchen counter.\n"
        "QUERY: small kettle on left side of wooden kitchen counter\n\n"
    )
    content.append((example,))

    # ------------------Task to solve----------------------------
    prompt = (
        f"Now given a new task, generate ONE specific retrieval query to help the agent recall relevant past observations.\n"
        f"Task Instruction: {question}\n"
    )
    if image_goal is not None:
        content.append((prompt, image_goal))
        content.append(("\n",))
    else:
        content.append((prompt + "\n",))

    if frontier_imgs is not None and len(frontier_imgs) > 0:
        prompt = (
            "Frontier observations are available. Consider them implicitly in your reasoning.\n"
        )
        content.append((prompt, frontier_imgs))

    # ------------------Call model-------------------------------
    response = call_qwen25(sys_prompt, content, model, processor)

    if response is None:
        return {"thought": "fallback", "query": question}

    # ------------------Parse response----------------------------
    lines = [l.strip() for l in response.split("\n") if l.strip()]
    thought, query = None, None

    for l in lines:
        l_strip = l.strip()
        l_lower = l_strip.lower()

        if l_lower.startswith("thought:") and thought is None:
            thought = l_strip[len(l_strip.split(':')[0]) + 1:].strip()
        elif l_lower.startswith("query:") and query is None:
            query = l_strip[len(l_strip.split(':')[0]) + 1:].strip()

        if thought and query:
            break

    # fallback
    if not (thought and query):
        return {"thought": "fallback", "query": question}

    return {"thought": thought, "query": query}

def memory_prefiltering(
    question,
    mem_base,
    egocentric_imgs,
    snapshot_full_imgs,
    snapshot_crops,
    snapshot_classes,
    snapshot_clusters,
    seen_classes,
    top_k=5,
    top_k_classes=3,
    image_goal=None,
    image_np=None,
    frontier_imgs=None,
    model=None,
    processor=None,
    verbose=False,
):

    response = get_retrieval_query_from_llm(question, model, processor, frontier_imgs=frontier_imgs, image_goal=image_goal)
    query = response["query"]
    if verbose:
        logging.info(f"QUERY: {query}")

    if image_goal is not None:
        results, _ = mem_base.search_image_to_pair(query_text=query, query_image=image_np, top_k=top_k)
    else:
        results, _ = mem_base.search_text_to_pair(query_text=query, top_k=top_k)
    
    if verbose:
        rgb_ids = [item['rgb_id'] for item in results]
        logging.info(f"Memory name list: {rgb_ids}")

    cluster_index_map = {k: i for i, k in enumerate(snapshot_clusters.keys())}
    keep_index = [cluster_index_map[sample["rgb_id"]] for sample in results if sample["rgb_id"] in cluster_index_map]

    keep_snapshot_id = [list(snapshot_classes.keys())[i] for i in keep_index]
    snapshot_classes = {rgb_id: snapshot_classes[rgb_id] for rgb_id in keep_snapshot_id}

    keep_index_snapshot = {}
    for rgb_id in keep_snapshot_id:
        keep_index_snapshot[rgb_id] = [i for i in range(len(snapshot_classes[rgb_id]))]
        snapshot_classes[rgb_id] = [
            snapshot_classes[rgb_id][i] for i in keep_index_snapshot[rgb_id]
        ]
    return snapshot_classes, keep_index, keep_index_snapshot

def get_step_info(step, model, processor, verbose=False):
    # 1 get question data
    question = step["question"]
    image_goal = None
    image_np = None
    if "task_type" in step and step["task_type"] == "image":
        with open(step["image"], "rb") as image_file:
            image_goal = base64.b64encode(image_file.read()).decode("utf-8")
        image_np = np.array(Image.open(step["image"]).convert("RGB"))

    # 2 get step information(egocentric, frontier, memory)
    # 2.1 get egocentric views
    egocentric_imgs = []
    if step.get("use_egocentric_views", False):
        for egocentric_view in step["egocentric_views"]:
            egocentric_imgs.append(encode_tensor2base64(egocentric_view))

    # 2.2 get frontiers
    frontier_imgs = []
    for frontier in step["frontier_imgs"]:
        frontier_imgs.append(encode_tensor2base64(frontier))

    # 2.3 get snapshots and memory base
    mem_base = KnowledgeBase()
    snapshot_classes = {}  # rgb_id -> list of classes
    snapshot_full_imgs = {}  # rgb_id -> full img (base64)
    snapshot_crops = {}  # rgb_id -> list of crops (base64)
    snapshot_clusters = {}  # rgb_id -> list of clusters (obj ids or classes)
    obj_map = step["obj_map"]
    seen_classes = set()
    for i, rgb_id in enumerate(step["snapshot_imgs"].keys()):
        snapshot_img = step["snapshot_imgs"][rgb_id]["full_img"]
        snapshot_full_imgs[rgb_id] = encode_tensor2base64(snapshot_img)
        snapshot_crops[rgb_id] = [
            encode_tensor2base64(crop_data["crop"])
            for crop_data in step["snapshot_imgs"][rgb_id]["object_crop"]
        ]
        snapshot_class = [
            crop_data["obj_class"]
            for crop_data in step["snapshot_imgs"][rgb_id]["object_crop"]
        ]
        cluster_class = [
            obj_map[int(obj_id)] for obj_id in step["snapshot_objects"][rgb_id]
        ]
        # remove duplicates
        seen_classes.update(sorted(list(set(snapshot_class))))
        snapshot_classes[rgb_id] = snapshot_class
        snapshot_clusters[rgb_id] = cluster_class

        # load memorybase
        unique_class = list(set(snapshot_class))
        snapshot_text = ", ".join(unique_class)
        mem_base.add_to_knowledge_base(image=snapshot_img, text=snapshot_text, name=rgb_id)

    # 3 prefiltering, note that we need the obj_id_mapping
    keep_index = list(range(len(snapshot_full_imgs)))
    keep_index_snapshot = {
        rgb_id: list(range(len(snapshot_crops[rgb_id]))) for rgb_id in snapshot_crops
    }
    if step.get("use_prefiltering") is True:
        n_prev_snapshot = len(snapshot_full_imgs)

        snapshot_classes, keep_index, keep_index_snapshot = memory_prefiltering(
            question=question,
            mem_base=mem_base,
            egocentric_imgs=egocentric_imgs,
            snapshot_full_imgs=snapshot_full_imgs,
            snapshot_crops=snapshot_crops,
            snapshot_classes=snapshot_classes,
            snapshot_clusters=snapshot_clusters,
            seen_classes=seen_classes,
            top_k=step.get("top_k_memories", 5),
            image_goal=image_goal,
            image_np=image_np,
            frontier_imgs=frontier_imgs,
            model=model,
            processor=processor,
            verbose=verbose,
        )
        # filter full imgs and crops according to keep_index_snapshot (keep_snapshot_ids)
        # keep_index_snapshot keys are rgb_id (base64 keys) — reconstruct snapshot_full_imgs accordingly
        snapshot_full_imgs = {
            rgb_id: snapshot_full_imgs[rgb_id] for rgb_id in keep_index_snapshot.keys()
        }
        for rgb_id in list(snapshot_crops.keys()):
            if rgb_id in keep_index_snapshot:
                snapshot_crops[rgb_id] = [
                    snapshot_crops[rgb_id][i] for i in keep_index_snapshot[rgb_id]
                ]
            else:
                snapshot_crops.pop(rgb_id, None)

        if verbose:
            logging.info(
                f"Prefiltering memory: {n_prev_snapshot} -> {len(snapshot_full_imgs)}"
            )

    return (
        question,
        image_goal,
        egocentric_imgs,
        frontier_imgs,
        snapshot_full_imgs,
        snapshot_classes,
        snapshot_crops,
        keep_index,
        keep_index_snapshot,
    )

def explore_step(step, cfg, model, processor, verbose=False):
    step["use_prefiltering"] = cfg.prefiltering
    step["top_k_memories"] = cfg.top_k_memories
    (
        question,
        image_goal,
        egocentric_imgs,
        frontier_imgs,
        snapshot_full_imgs,
        snapshot_classes,
        snapshot_crops,
        snapshot_id_mapping,
        snapshot_crop_mapping,
    ) = get_step_info(step, model, processor, verbose)

    sys_prompt, content = format_explore_prompt_object(
        question,
        egocentric_imgs,
        frontier_imgs,
        snapshot_full_imgs,
        snapshot_classes,
        snapshot_crops,
        egocentric_view=step.get("use_egocentric_views", False),
        use_snapshot_class=True,
        image_goal=image_goal,
    )

    if verbose:
        logging.info("Input prompt:")
        message = ""
        for c in content:
            if "example" in c[0].lower():
                continue
            if "definitions" in c[0].lower():
                continue
            message += c[0]
            if len(c) == 2:
                message += f"[{c[1][:10]}...]"
        logging.info(message)

    retry_bound = 1
    final_response = None
    final_reason = None
    for _ in range(retry_bound):
        response = call_qwen25(sys_prompt, content, model, processor)

        if response is None:
            print("call_qwen25 returns None, retrying")
            continue

        if verbose:
            logging.info("Response:")
            logging.info(response)
            
        lines = [l.strip() for l in response.split("\n") if l.strip()]
        reason, answer = None, None
        for l in lines:
            l_strip = l.strip()
            l_lower = l_strip.lower()
            if l_lower.startswith("thought:") and reason is None:
                reason = l_strip[len(l_strip.split(':')[0]) + 1:].strip()
            elif l_lower.startswith("answer:") and answer is None:
                answer = l_strip[len(l_strip.split(':')[0]) + 1:].strip().lower()
            if reason and answer:
                break
        try:
            choice_type, choice_id = answer.split(",")[0].strip().split(" ")
        except Exception as e:
            print(e)
            continue

        response_valid = False
        if (
            choice_type == "memory"
            and choice_id.isdigit()
            and 0 <= int(choice_id) < len(snapshot_full_imgs)
        ):
            try:
                object_choice_type, object_choice_id = (
                    answer.split(",")[1].strip().split(" ")
                )
            except Exception as e:
                print(f"Error in splitting response")
                object_choice_type = "object"
                object_choice_id = "100"
            if (
                object_choice_type == "object"
                and object_choice_id.isdigit()
                and 0
                <= int(object_choice_id)
                < len(list(snapshot_crop_mapping.values())[int(choice_id)])
            ):
                response_valid = True
            else:
                new_obj_id = "0"
                answer = f"memory {choice_id}, object {new_obj_id}"
                response_valid = True
                logging.info("Answer:")
                logging.info(answer)
        elif (
            choice_type == "frontier"
            and choice_id.isdigit()
            and 0 <= int(choice_id) < len(frontier_imgs)
        ):
            response_valid = True

        if response_valid:
            final_response = answer
            final_reason = reason
            break

    return (
        final_response,
        snapshot_id_mapping,
        snapshot_crop_mapping,
        final_reason,
        len(snapshot_full_imgs),
    )

def explore_step_qa(step, cfg, model, processor, verbose=False):
    step["use_prefiltering"] = cfg.prefiltering
    step["top_k_memories"] = cfg.top_k_memories
    (
        question,
        image_goal,
        egocentric_imgs,
        frontier_imgs,
        snapshot_full_imgs,
        snapshot_classes,
        snapshot_crops,
        snapshot_id_mapping,
        snapshot_crop_mapping,
    ) = get_step_info(step, model, processor, verbose)

    sys_prompt, content = format_explore_prompt_qa(
        question,
        egocentric_imgs,
        snapshot_full_imgs,
        snapshot_classes,
        egocentric_view=step.get("use_egocentric_views", False),
        use_snapshot_class=True,
        image_goal=image_goal,
        answer_type=step.get("answer_type", None)
    )

    if verbose:
        logging.info("Input prompt:")
        message = ""
        for c in content:
            if "example" in c[0].lower():
                continue
            if "definitions" in c[0].lower():
                continue
            message += c[0]
            if len(c) == 2:
                message += f"[{c[1][:10]}...]"
        logging.info(message)

    retry_bound = 1
    final_response = None
    final_reason = None
    for _ in range(retry_bound):
        full_response = call_qwen25(sys_prompt, content, model, processor)

        if full_response is None:
            print("call_qwen25 returns None, retrying")
            continue

        full_response = full_response.strip()
        if "\n" in full_response:
            full_response = full_response.split("\n")
            response, reason = full_response[0], full_response[-1]
            response, reason = response.strip(), reason.strip()
        else:
            response = full_response
            reason = ""
        response = response.lower()
        try:
            choice_type, choice_id = response.split(" ")
        except Exception as e:
            print(f"Error in splitting response: {response}")
            print(e)
            continue

        response_valid = False
        if (
            choice_type == "memory"
            and choice_id.isdigit()
            and 0 <= int(choice_id) < len(snapshot_full_imgs)
        ):
            response_valid = True

        if response_valid:
            final_response = response
            final_reason = reason
            break

    return final_response, snapshot_id_mapping, final_reason, len(snapshot_full_imgs)