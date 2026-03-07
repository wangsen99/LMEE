# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import List, Dict
from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(predict: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    return 1.0 if format_match else 0.0

def accuracy_reward(predict: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0

def is_number(s):
    try:
        float(s)  # Will handle both int and float strings
        return True
    except ValueError:
        return False
    
def similarity_score(a, b):
    if a == b:
        return 1.0
    if a == 0 or b == 0:
        return 0.0
    return 1 - (abs(a - b) / max(abs(a), abs(b)))

def compute_score(predicts: List[str], 
                   ground_truths: List[str], 
                   labels: List[str], 
                   qa_category: List[str], 
                   qa_answer: List[str], 
                   notool: List[int],
                   w_action: float = 0.4, 
                   w_frontier: float = 0.4, 
                   w_answer: float = 0.1, 
                   w_format: float = 0.1) -> List[Dict[str, float]]:

    scores = []
    
    frontier2action = {
        "0": "turn_left",
        "1": "move_forward",
        "2": "turn_right"
    }

    for pred, gt_action, gt_label, q_cat, q_ans, nt in zip(predicts, ground_truths, labels, qa_category, qa_answer, notool):
        total_score = 0.0
        action_score = 0.0
        frontier_score = 0.0
        answer_score = 0.0
        format_score = 0.0
        consistency = 1.0
        tool_used = 0

        # ----------------- 解析预测 -----------------
        pred_action = None
        pred_label = None
        pred_answer = None

        ans_match = re.findall(r'ACTION:\s*(.*?)(?=\n|$)', pred)
        if ans_match:
            pred_action = ans_match[-1].strip()

        frontier_match = re.findall(r'FRONTIER:\s*(\d+)', pred)
        if frontier_match:
            pred_label = frontier_match[-1].strip()

        answer_match = re.findall(r'ANSWER:\s*(.*?)(?=\n|$)', pred)
        if answer_match:
            pred_answer = answer_match[-1].strip()

        # ----------------- 动作奖励 -----------------
        if pred_action is not None and pred_action == gt_action.strip():
            action_score = 1.0

        # ----------------- 前沿奖励 -----------------
        if pred_label is not None and pred_label == str(gt_label):
            frontier_score = 1.0

        # ----------------- 一致性惩罚 -----------------
        if pred_action is not None and pred_label is not None:
            expected_action = frontier2action.get(pred_label, None)
            if expected_action is not None and expected_action != pred_action:
                consistency = 0.5

        # ----------------- 格式奖励（分段式） -----------------
        format_parts = 0
        if pred_action is not None:
            format_parts += 1
        if pred_label is not None:
            format_parts += 1
        if pred_answer is not None:
            format_parts += 1
        format_score = format_parts / 3.0  # 1个=0.33，2个=0.67，3个=1.0

        # ----------------- 答案奖励 -----------------
        if q_cat == "Task Progress":
            gt_objects = [o.strip() for o in q_ans.split(",") if o.strip()]
            pred_objects = [o.strip() for o in pred_answer.split(",") if o.strip()] if pred_answer else []

            tp = len(set(pred_objects) & set(gt_objects))
            fp = len(set(pred_objects) - set(gt_objects))
            fn = len(set(gt_objects) - set(pred_objects))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            answer_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            if q_ans is not None and pred_answer is not None:
                answer_score = 1.0 if pred_answer.strip().upper() == q_ans.strip().upper() else 0.0

        # ----------------- notool 调整 -----------------
        if nt == 1:
            answer_score *= 0.5
            format_score *= 0.5
            action_score *= 0.6
            frontier_score *= 0.6
        else:
            tool_used = 1
            answer_score *= 1.2
            format_score *= 1.2
            action_score *= 1.2
            frontier_score *= 1.2

        # ----------------- 综合加权 -----------------
        total_score = (w_action * action_score * consistency +
                       w_frontier * frontier_score * consistency +
                       w_answer * answer_score +
                       w_format * format_score)
        total_score = max(0.0, min(1.0, total_score))

        scores.append({
            "overall": total_score,
            "action": action_score,
            "frontier": frontier_score,
            "answer": answer_score,
            "format": format_score,
            "consistency": consistency,
            "tool_used": tool_used
        })

    return scores

# def compute_score(predicts: List[str], 
#                    ground_truths: List[str], 
#                    labels: List[str], 
#                    qa_category: List[str], 
#                    qa_answer: List[str], 
#                    notool: List[int],
#                    w_action: float = 0.2, 
#                    w_frontier: float = 0.2, 
#                    w_answer: float = 0.4, 
#                    w_format: float = 0.2) -> List[Dict[str, float]]:
#     """
#     平滑化奖励版：防止模型放弃输出结构化答案
#     """

#     scores = []
    
#     frontier2action = {
#         "0": "turn_left",
#         "1": "move_forward",
#         "2": "turn_right"
#     }

#     for pred, gt_action, gt_label, q_cat, q_ans, nt in zip(predicts, ground_truths, labels, qa_category, qa_answer, notool):
#         # 初始化
#         action_score = frontier_score = answer_score = format_score = 0.0
#         base_reward = 0.05  # 保底奖励，防止梯度为零
#         consistency = 1.0
#         tool_used = 0

#         # ---------- 解析预测 ----------
#         pred_action = None
#         pred_label = None
#         pred_answer = None

#         ans_match = re.findall(r'ACTION:\s*(.*?)(?=\n|$)', pred)
#         if ans_match:
#             pred_action = ans_match[0].strip()

#         frontier_match = re.findall(r'FRONTIER:\s*(\d+)', pred)
#         if frontier_match:
#             pred_label = frontier_match[0].strip()

#         answer_match = re.findall(r'ANSWER:\s*(.*?)(?=\n|$)', pred)
#         if answer_match:
#             pred_answer = answer_match[0].strip()

#         # ---------- 格式奖励 ----------
#         format_parts = sum(x is not None for x in [pred_action, pred_label, pred_answer])
#         # 使用平滑函数 (sigmoid-like)
#         format_score = 1 / (1 + np.exp(-1.5 * (format_parts - 1)))  # 0→0.18, 1→0.5, 2→0.82, 3→0.95

#         # ---------- 动作奖励 ----------
#         if pred_action is not None:
#             action_score = 1.0 if pred_action == gt_action.strip() else 0.2  # 部分错误仍给低奖励

#         # ---------- 前沿奖励 ----------
#         if pred_label is not None:
#             frontier_score = 1.0 if pred_label == str(gt_label) else 0.2

#         # ---------- 一致性约束（平滑化） ----------
#         if pred_action and pred_label:
#             expected_action = frontier2action.get(pred_label, None)
#             if expected_action and expected_action != pred_action:
#                 consistency = 0.6  # 不再是 0.1，而是线性惩罚

#         # ---------- 答案奖励 ----------
#         if q_cat == "Task Progress":
#             gt_objects = [o.strip() for o in q_ans.split(",") if o.strip()]
#             pred_objects = [o.strip() for o in pred_answer.split(",") if o.strip()] if pred_answer else []
#             if len(gt_objects) > 0:
#                 tp = len(set(pred_objects) & set(gt_objects))
#                 precision = tp / max(len(pred_objects), 1)
#                 recall = tp / len(gt_objects)
#                 answer_score = 0.5 * precision + 0.5 * recall  # 用平均替代 F1，更平滑
#         else:
#             if q_ans and pred_answer:
#                 answer_score = 1.0 if pred_answer.strip().upper() == q_ans.strip().upper() else 0.3  # 错误也给0.3

#         # ---------- 工具使用调整 ----------
#         if nt == 1:  # 没有使用工具
#             tool_used = 0
#             action_score *= 0.8
#             frontier_score *= 0.8
#             answer_score *= 0.9
#             format_score *= 0.9
#         else:
#             tool_used = 1
#             action_score *= 1.05
#             frontier_score *= 1.05
#             answer_score *= 1.1
#             format_score *= 1.05

#         # ---------- 组合奖励 ----------
#         total_score = base_reward + (
#             w_action * action_score * consistency +
#             w_frontier * frontier_score * consistency +
#             w_answer * answer_score +
#             w_format * format_score
#         )

#         # ---------- 平滑归一化 ----------
#         total_score = max(0.0, min(1.0, total_score))

#         scores.append({
#             "overall": total_score,
#             "action": action_score,
#             "frontier": frontier_score,
#             "answer": answer_score,
#             "format": format_score,
#             "consistency": consistency,
#             "tool_used": tool_used
#         })

#     return scores


