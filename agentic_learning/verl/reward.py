import re
from typing import Any

from agentic_learning.verl.utils.math_eval import math_scorer
from agentic_learning.verl.utils.math_eval import math_scorer


def last_boxed_only_string(string: str) -> str:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return ""

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx + 7: right_brace_idx] if right_brace_idx is not None else ""

def last_text_only_string(string: str) -> str:
    """Extract the last LaTeX text expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last text expression or None if not found
    """
    idx = string.rfind("\\text{")
    if idx < 0:
        return ""

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx + 6: right_brace_idx] if right_brace_idx is not None else ""

def science_scorer(prediction: str, ground_truth: str) -> float:
    if prediction.startswith("("):
        prediction = prediction[1:]
    if prediction.endswith(")"):
        prediction = prediction[:-1]

    if "\\text" in prediction:
        prediction = last_text_only_string(prediction)
    if "." in prediction:
        prediction = prediction.split(".")[0]

    if "\\text" in ground_truth:
        ground_truth = last_text_only_string(ground_truth)
    
    if prediction.lower() == ground_truth.lower():
        return 1.0
    else:
        return 0.0

def reasongym_scorer(prediction: str, ground_truth: str) -> float:
    if prediction == ground_truth:
        return 1.0
    else:
        return 0.0

def compute_score(
    data_source: str,
    solution_str: str, 
    ground_truth: str, 
    extra_info: dict[str, Any],
) -> dict[str, Any]:
    if data_source == "math":
        result = math_scorer(solution_str, ground_truth)
        score = result["score"]
        prediction = result["predicted_answer"]
    elif data_source == "science":
        prediction = last_boxed_only_string(solution_str)
        score = science_scorer(prediction, ground_truth)
    elif data_source == "puzzle":
        prediction = last_boxed_only_string(solution_str)
        prediction = last_boxed_only_string(solution_str)
        score = reasongym_scorer(prediction, ground_truth)
    else:
        raise ValueError("Unknown data source")

    return {
        "score": score,
        "acc": score,
        "prediction": prediction,
        "ground_truth": ground_truth
    }