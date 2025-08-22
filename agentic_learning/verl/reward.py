import re
from typing import Any

from antlr4.PredictionContext import PredictionContextFromRuleContext
from latex2sympy2_extended import NormalizationConfig
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
# from agentic_learning.verl.utils.qwen_math_eval import math_scorer


# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]

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

def normalize(text: str) -> str:
    text = text.split("=")[-1]

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        text = text.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        text = text.replace(expr, "")

    # Extract and normalize LaTeX math
    text = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", text)
    text = re.sub(r"(\\text\{)(.*?)(\})", "\\2", text)
    text = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", text)
    text = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", text)
    text = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", text)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    text = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", text)
    text = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", text)
    text = text.replace("$", "")

    # Normalize numbers
    if text.replace(",", "").isdigit():
        text = text.replace(",", "")

    return text.strip()

def math_scorer(prediction: str, ground_truth: str, timeout: int = 10) -> float:
    if prediction is None:
        prediction = ""

    prediction = normalize(prediction)
    ground_truth = normalize(ground_truth)

    prediction = f"\\boxed{{{prediction}}}"
    ground_truth = f"\\boxed{{{ground_truth}}}"

    parsed_ground_truth = parse(
        ground_truth, 
        extraction_config=[LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()],
        parsing_timeout=timeout
    )
    if len(parsed_ground_truth) == 0:
        parsed_ground_truth_with_env = f'${ground_truth}$'
        parsed_ground_truth = parse(
            parsed_ground_truth_with_env,
            extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
            parsing_timeout=timeout
        )

    if len(parsed_ground_truth) != 0:
        parsed_prediction = parse(
            prediction,
            extraction_config=[
                LatexExtractionConfig(
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                    normalization_config=NormalizationConfig(),
                )
            ],
            parsing_timeout=timeout
        )
        if verify(parsed_prediction, parsed_ground_truth, timeout_seconds=timeout):
            return 1.0
        else:
            return 0.0
    return 1.0

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

def sudoku_scorer(prediction: str, ground_truth: str) -> float:
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
    prediction = last_boxed_only_string(solution_str)
    
    if data_source == "math":
        score = math_scorer(prediction, ground_truth, 5)
    elif data_source == "science":
        score = science_scorer(prediction, ground_truth)
    elif data_source == "puzzle":
        score = reasongym_scorer(prediction, ground_truth)
    else:
        raise ValueError("Unknown data source")

    return {
        "score": score,
        "prediction": prediction,
        "ground_truth": ground_truth
    }