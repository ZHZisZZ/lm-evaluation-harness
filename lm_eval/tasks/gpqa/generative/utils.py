import random
import re
from typing import Dict, List, Optional

import datasets


def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        choices = [
            preprocess(doc["Incorrect Answer 1"]),
            preprocess(doc["Incorrect Answer 2"]),
            preprocess(doc["Incorrect Answer 3"]),
            preprocess(doc["Correct Answer"]),
        ]

        random.shuffle(choices)
        correct_answer_index = choices.index(preprocess(doc["Correct Answer"]))

        out_doc = {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "choices": [choices[0], choices[1], choices[2], choices[3]],
            "answer": f"({chr(65 + correct_answer_index)})",
        }
        return out_doc

    return dataset.map(_process_doc)


def extract_choice_eval360(response: str) -> Optional[str]:
    leading_answer = re.match(r"^\s*([A-D])(?:\b|(?=[A-Z]))", response)
    if leading_answer:
        return leading_answer.group(1).upper()

    patterns = (
        r"(?is)(?:\*+\s*)?(?:final\s+answer|correct\s+answer|answer)(?:\s*\*+)?"
        r"\s*(?:is|:|=)?\s*(?:\*+\s*)?\(?\s*([A-D])\s*\)?"
        r"(?=\s*(?:[\).,:;-]|\*+|\s|$))",
        r"(?is)the\s+correct\s+answer\s+is\s*(?:\*+\s*)?\(?\s*([A-D])\s*\)?",
        r"(?im)^\s*(?:\*+\s*)?\(?\s*([A-D])\s*\)?(?:\s*[\).,:;-].*)?$",
    )
    for pattern in patterns:
        matches = re.findall(pattern, response)
        if matches:
            return matches[-1].upper()
    return None


def process_results_eval360(doc: dict, results: List[str]) -> Dict[str, int]:
    prediction = extract_choice_eval360(results[0])
    target = str(doc["ground_truth"]).strip().upper()
    return {"exact_match": int(prediction == target)}
