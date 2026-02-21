import re
import evaluate as hf_evaluate

from lm_eval.tasks.humaneval.sanitize_utils import sanitize

try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [
        [
            doc["prompt"] + (r if r.find("```") == -1 else r[: r.find("```")])
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]


def build_predictions_llada(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    processed = []

    for resp_list, doc in zip(resps, docs):
        new_list = []
        for resp in resp_list:

            # extract code block
            blocks = re.findall(r"```(?:[a-zA-Z0-9_+-]*)?\s*\n?(.*?)(?:```|$)", resp, flags=re.DOTALL)
            if len(blocks) >= 1:
                resp = blocks[0]

            # strip leading whitespace
            resp = resp.lstrip()
            new_list.append(resp)

        processed.append(new_list)

    return processed


def build_predictions_llada_fastdllm(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [
        [
            sanitize(
                doc["prompt"] + "\n" + r.split('```python\n', 1)[-1].split('```')[0],
                doc["entry_point"],
            )
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]


def pass_at_k_dream(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]

    processed_predictions = []
    for preds in predictions:
        processed_preds = []
        for p in preds:
            processed_preds.append(p.strip("```")[0] if "```" in p else p)
        processed_predictions.append(processed_preds)

    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


def build_predictions_instruct_dream(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [
        [
            sanitize(
                doc["prompt"] + "\n" + r.split('```python\n', 1)[-1].split('```')[0],
                doc["entry_point"]
            )
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]
