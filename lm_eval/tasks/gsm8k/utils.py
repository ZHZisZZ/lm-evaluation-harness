"""
Utilities for GSM8K tasks in lm-evaluation-harness.

Answer extraction logic is aligned with parse_gsm_answers in
/mnt/petrelfs/fanyuyu/fyy/d1/eval/parse_and_get_acc.py.

Run from repo root with: python -m lm_eval.tasks.gsm8k.utils (no main).
"""

import re
from typing import Any, Dict, List, Optional


def extract_ground_truth_from_doc(doc: dict) -> Optional[float]:
    """
    Extract the numeric ground-truth answer from a GSM8K doc.
    GSM8K answer format is "reasoning #### number" (e.g. "steps #### 42").
    """
    raw = doc.get("answer")
    if not raw or not isinstance(raw, str):
        return None
    parts = raw.split("####")
    if len(parts) < 2:
        return None
    num_str = parts[-1].strip().replace(",", "")
    try:
        return float(num_str)
    except ValueError:
        return None


def extract_gsm_answer(raw_generation: str) -> Optional[float]:
    """
    Extract a numeric answer from model output using the same logic as
    parse_gsm_answers in parse_and_get_acc.py: prefer \\boxed{...}, then <answer>...</answer>.
    """
    if not raw_generation or not isinstance(raw_generation, str):
        return None
    parsed_answer = None

    boxed_matches = re.findall(r"\\boxed{(.*?)}", raw_generation)
    if boxed_matches:
        for boxed_content in boxed_matches:
            boxed_content = boxed_content.strip()
            if (
                boxed_content
                and boxed_content != "..."
                and not re.match(r"^\.+$", boxed_content)
            ):
                try:
                    parsed_answer = float(boxed_content)
                    break
                except ValueError:
                    numbers = re.findall(r"-?\d+\.?\d*", boxed_content)
                    if numbers:
                        try:
                            parsed_answer = float(numbers[0])
                            break
                        except ValueError:
                            pass

    if parsed_answer is None:
        answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
            if answer_text:
                try:
                    parsed_answer = float(answer_text)
                except ValueError:
                    numbers = re.findall(r"-?\d+\.?\d*", answer_text)
                    if numbers:
                        try:
                            parsed_answer = float(numbers[-1])
                        except ValueError:
                            pass

    return parsed_answer


def process_results(doc: dict, results: List[Any]) -> Dict[str, int]:
    """
    process_results for GSM8K reasoning format: extract answer from generation
    and compare to ground truth using extract_gsm_answer logic.
    """
    if not results or not results[0]:
        return {"exact_match": 0}
    raw_generation = results[0] if isinstance(results[0], str) else str(results[0])
    ground_truth = extract_ground_truth_from_doc(doc)
    parsed = extract_gsm_answer(raw_generation)
    is_correct = (
        parsed is not None and ground_truth is not None and parsed == ground_truth
    )
    return {"exact_match": 1 if is_correct else 0}
