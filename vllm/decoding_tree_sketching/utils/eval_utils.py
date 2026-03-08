"""
Answer extraction and evaluation utilities for multiple benchmark datasets.

Supported datasets:
- AIME 2024/2025: \\boxed{} extraction with numeric fallback
- GPQA Diamond: Multiple-choice (A/B/C/D) extraction
- LiveBench Reasoning: Flexible answer extraction (XML, boxed, bold, etc.)
"""

import json
import os
import re
from typing import Optional


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def get_examples(split):
    path = os.path.join("data/", f"{split}.jsonl")
    examples = read_jsonl(path)

    for ex in examples:
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"] + "<|endoftext|>")

    print(f"{len(examples)} {split} examples")
    return examples


ANS_RE_gt = re.compile(r"#### (\-?[0-9\.\,]+)")
ANS_RE_qwq = re.compile(r"boxed\{(.*?)\}")

INVALID_ANS = "[invalid]"

# GPQA multiple-choice answer pattern
ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*\$?([A-D])\$?"


def extract_answer_gt(completion):
    match = ANS_RE_gt.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def extract_answer_qwq(completion):
    match = ANS_RE_qwq.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "").replace("%", "").replace("\\", "").replace("$", "")
        return match_str
    else:
        return INVALID_ANS


def extract_answer_llm(text):
    """
    Extract the last number-containing string from text and clean it
    to only include digits, decimal points, and a leading sign.

    Returns:
        Cleaned number string, or "INVALID_ANS" if not found.
    """
    number_strings = re.findall(r'\S*\d+\S*', text)

    if not number_strings:
        return "INVALID_ANS"

    last_number_string = number_strings[-1]

    cleaned_number = ''.join(char for char in last_number_string
                           if char.isdigit() or char in '.-')

    if cleaned_number.count('.') > 1:
        first_dot_index = cleaned_number.index('.')
        cleaned_number = cleaned_number[:first_dot_index + 1] + \
                        cleaned_number[first_dot_index + 1:].replace('.', '')

    if cleaned_number.startswith('-'):
        cleaned_number = '-' + cleaned_number[1:].replace('-', '')
    else:
        cleaned_number = cleaned_number.replace('-', '')

    return cleaned_number


def extract_all_boxed_content(text):
    results = []
    start = 0

    while True:
        start = text.find(r"\boxed{", start)
        if start == -1:
            break

        brace_count = 0
        result = []
        i = start

        while i < len(text):
            char = text[i]
            result.append(char)

            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1

            if brace_count == 0 and result[-1] == '}':
                break

            i += 1

        results.append(''.join(result))
        start = i + 1

    return results


def is_correct(model_completion, gt_example):
    gt_answer = extract_answer_gt(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer_gt(model_completion) == gt_answer


def fmt_float(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s.replace(".", "p") if s else "0"


def extract_livebench_answer(text: str) -> Optional[str]:
    if not text:
        return None

    solution_matches = re.findall(r"<solution>(.*?)</solution>", text, re.DOTALL | re.IGNORECASE)
    if solution_matches:
        return solution_matches[-1].strip()

    boxed_matches = re.findall(r"\\boxed\{(.*?)\}", text)
    if boxed_matches:
        return boxed_matches[-1].strip()

    bold_matches = re.findall(r"\*{2,}(.*?)\*{2,}", text, re.DOTALL)
    if bold_matches:
        return bold_matches[-1].strip().strip('*')

    lines = text.strip().split('\n')
    for line in reversed(lines):
        ans_match = re.search(r"(?:^|\.\s*|The\s+)(?:Answer|answer)\s*(?:is|:)\s*(.*?)(?:\.|$)", line, re.IGNORECASE)
        if ans_match:
            candidate = ans_match.group(1).strip()
            if len(candidate) < 100:
                return candidate
    try:
        num_ans = extract_answer_llm(text)
        if num_ans:
            return num_ans
    except:
        pass

    return None


def check_livebench_match(model_ans, gt_ans):
    def normalize_list(s):
        if s is None: return []
        s = str(s).replace("<solution>", "").replace("</solution>", "")
        s = s.replace("\n", ",")
        items = [item.strip() for item in s.split(",") if item.strip()]
        return [i.lower() for i in items]

    normalized_model = normalize_list(model_ans)
    normalized_gt = normalize_list(gt_ans)
    return normalized_model == normalized_gt


def extract_gpqa_answer(text: str) -> Optional[str]:
    """Extract GPQA multiple-choice answer (A/B/C/D) from text."""
    if not text:
        return None

    lines = text.strip().split('\n')
    for line in reversed(lines):
        match = re.search(ANSWER_PATTERN_MULTICHOICE, line)
        if match:
            return match.group(1).upper()

    all_matches = re.findall(r"\b([A-D])\b", text)
    if all_matches:
        return all_matches[-1].upper()

    return None
