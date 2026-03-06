import os
import json
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
import io
import base64
from tqdm import tqdm
import re
# ---------------------------
# Utility functions
# ---------------------------
def pil_to_base64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_difficulty(item):
    """
    Determine difficulty level.
    1. Prefer 'difficulty' key if present.
    2. Otherwise infer from image_path (e.g., 'easy', 'medium', 'hard').
    3. Default to 'unknown' if not found.
    """
    if "difficulty" in item:
        return item["difficulty"]

    path = item.get("image_path", "").lower()
    if "easy" in path:
        return "easy"
    elif "medium" in path:
        return "medium"
    elif "hard" in path:
        return "hard"
    else:
        return "unknown"


def evaluate_open_answer(model, processor, image_path, question, gt_answer, pred_answer, device):
    """
    Use a multimodal model to grade an open-ended prediction against the ground truth.
    The model should output an integer score 1–5.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[Warning] Cannot open image: {image_path}. Error: {e}")
        return 1  # worst score if image fails

    img_b64 = pil_to_base64(img)

    system_prompt = (
        "You are an AI assistant who will help me to evaluate the response given the question and the correct answer.\n"
        "To mark a response, you should output a single integer between 1 and 5 (including 1, 5).\n"
        "5 means that the response perfectly matches the answer.\n"
        "1 means that the response is completely different from the answer.\n"
        "Your Turn:\n"
        f"Question: {question}\n"
        f"Answer: {gt_answer}\n"
        f"Response: {pred_answer}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": "Here is the image related to the question."},
            {"type": "image", "image": f"data:image;base64,{img_b64}"}
        ]}
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages, image_patch_size=processor.image_processor.patch_size)

    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
        do_resize=False
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print("Score:", output_text)

    # Try to extract integer score from output
    for token in output_text.split():
        if token.strip().isdigit():
            score = int(token.strip())
            if 1 <= score <= 5:
                return score
    return 1  # default lowest if parsing fails

def normalize_choice(ans):
    if ans is None:
        return None
    ans = ans.strip().upper()
    match = re.search(r'\b([A-Z])\b', ans)
    return match.group(1) if match else None

# ---------------------------
# Main evaluation
# ---------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="results/exp_eval_lmee/lmee_answer.json", help="Path to the QA JSON file.")
    parser.add_argument("--root_dir", type=str, default="../data/LMEE-Bench/task_test", help="Root directory for image paths.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for model.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load data
    with open(args.json_path, "r") as f:
        data = json.load(f)

    print('Question Num:', len(data))

    # Load Qwen model for open-ended scoring
    pretrained = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    model = AutoModelForImageTextToText.from_pretrained(
        pretrained, torch_dtype="auto", device_map="cuda:0"
    )
    processor = AutoProcessor.from_pretrained(pretrained)

    # Collect statistics
    category_scores = {}
    category_acc = {}

    difficulty_scores = {}
    difficulty_acc = {}

    open_scores = []
    choice_acc = []

    print("Evaluating predictions...")
    for item in tqdm(data):
        category = item["category"]
        ans_type = item["answer_type"]
        gt_answer = item["open_answer"] if ans_type == "open" else item["answer"]
        pred_answer = item.get("pred_answer", "None")
        image_path = os.path.join(args.root_dir, item["image_path"])
        difficulty = get_difficulty(item)

        # Initialize per-category stats
        if category not in category_scores:
            category_scores[category] = []
            category_acc[category] = []

        # Initialize per-difficulty stats
        if difficulty not in difficulty_scores:
            difficulty_scores[difficulty] = []
            difficulty_acc[difficulty] = []

        # Evaluate
        if ans_type == "choice":
            pred = normalize_choice(pred_answer)
            gt = normalize_choice(gt_answer)

            acc = 1 if pred == gt else 0
            category_acc[category].append(acc)
            difficulty_acc[difficulty].append(acc)
            choice_acc.append(acc)
        else:
            score = evaluate_open_answer(model, processor, image_path, item["question"], gt_answer, pred_answer, device)
            score = 100.0 * (score - 1.0) / 4.0  # normalize to 0–100
            category_scores[category].append(score)
            difficulty_scores[difficulty].append(score)
            open_scores.append(score)

    # ---------------------------
    # Print category-wise results
    # ---------------------------
    print("\n===== Evaluation Results by Category =====")
    for category in category_acc:
        avg_acc = np.mean(category_acc[category]) * 100 if len(category_acc[category]) > 0 else None
        avg_score = np.mean(category_scores[category]) if len(category_scores[category]) > 0 else None

        print(f"{category}: ", end="")
        if avg_acc is not None:
            print(f"Accuracy = {avg_acc:.2f}%, ", end="")
        if avg_score is not None:
            print(f"Open QA Score = {avg_score:.2f}, ", end="")

    # ---------------------------
    # Print difficulty-wise results
    # ---------------------------
    print("\n===== Evaluation Results by Difficulty =====")
    for diff in difficulty_acc:
        avg_acc = np.mean(difficulty_acc[diff]) * 100 if len(difficulty_acc[diff]) > 0 else None
        avg_score = np.mean(difficulty_scores[diff]) if len(difficulty_scores[diff]) > 0 else None

        print(f"{diff}: ", end="")
        if avg_acc is not None:
            print(f"Accuracy = {avg_acc:.2f}%, ", end="")
        if avg_score is not None:
            print(f"Open QA Score = {avg_score:.2f}, ", end="")

    # ---------------------------
    # Overall summary
    # ---------------------------
    print("\n===== Overall Summary =====")
    if len(choice_acc) > 0:
        print(f"Overall Choice Accuracy: {np.mean(choice_acc) * 100:.2f}%")
    if len(open_scores) > 0:
        print(f"Overall Open QA Score: {np.mean(open_scores):.2f}")


if __name__ == "__main__":
    main()
