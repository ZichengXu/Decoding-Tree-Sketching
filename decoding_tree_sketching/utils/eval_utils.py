import json
import os
import re

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
        # Remove all possible %, commas, backslashes, and dollar signs
        match_str = match_str.replace(",", "").replace("%", "").replace("\\", "").replace("$", "")
        return match_str
    else:
        return INVALID_ANS

def extract_answer_llm(text):
    """
    Extracts the last number-containing string from the text and cleans it 
    to only include digits, decimal points, and a leading sign.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Cleaned number string, or "INVALID_ANS" if not found.
    """
    # Match strings containing digits
    number_strings = re.findall(r'\S*\d+\S*', text)
    
    if not number_strings:
        return "INVALID_ANS"
    
    # Get the last matched string
    last_number_string = number_strings[-1]
    
    # Keep only digits, decimal points, and signs
    cleaned_number = ''.join(char for char in last_number_string 
                           if char.isdigit() or char in '.-')
    
    # Handle potential multiple decimal points or signs
    # Only keep the first decimal point
    if cleaned_number.count('.') > 1:
        first_dot_index = cleaned_number.index('.')
        cleaned_number = cleaned_number[:first_dot_index + 1] + \
                        cleaned_number[first_dot_index + 1:].replace('.', '')
    
    # Only keep the leading sign
    if cleaned_number.startswith('-'):
        cleaned_number = '-' + cleaned_number[1:].replace('-', '')
    else:
        cleaned_number = cleaned_number.replace('-', '')
    
    return cleaned_number

def extract_all_boxed_content(text):
    results = []
    start = 0

    while True:
        # Find the next occurrence of \boxed{
        start = text.find(r"\boxed{", start)
        if start == -1:
            break  # No more \boxed{ found

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

            # Stop when the braces are balanced
            if brace_count == 0 and result[-1] == '}':
                break

            i += 1

        # Append the matched content
        results.append(''.join(result))
        start = i + 1  # Move past the current match to find the next

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
