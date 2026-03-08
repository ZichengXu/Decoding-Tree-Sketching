#!/usr/bin/env python3
"""
Benchmark runner for DTS (Decoding Tree Sketching) experiments using vLLM.

Supports two modes:
- DTS mode: Entropy-guided tree decoding with branching and majority vote
- Standard mode: Standard vLLM autoregressive generation (baseline)

Usage (DTS Greedy, 1 trace):
  python run_benchmark.py dts \
    --model_name 1.5B --dataset_name aime24 \
    -e 2.5 -k 3 -a 48 -m 32768 -t 0.6 -s 0 --num_traces 1 -n 5

Usage (DTS Stable, 8 traces with majority vote):
  python run_benchmark.py dts \
    --model_name 1.5B --dataset_name aime24 \
    -e 2.5 -k 3 -a 48 -m 32768 -t 0.6 -s 0 --num_traces 8 -n 5

Usage (Standard baseline):
  python run_benchmark.py standard \
    --model_name 1.5B --dataset_name aime24 \
    -t 0.6 -s 0 -n 5
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import json
import random
import sys
import re
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any
from collections import Counter

import numpy as np
import yaml
from datasets import load_dataset

from vllm import LLM, SamplingParams, TokensPrompt

from decoding_tree_sketching.tree_decoder import TreeDecoder, majority_vote_answers
from decoding_tree_sketching.utils.eval_utils import (
    extract_answer_llm,
    extract_answer_qwq,
    is_float,
    extract_livebench_answer,
    fmt_float,
    check_livebench_match,
    extract_gpqa_answer,
    ANSWER_PATTERN_MULTICHOICE,
)


# ─── GPQA multichoice template ───

QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()


# ─── Answer extraction & evaluation ───


def make_answer_extractor(dataset_name: str):
    """Create a dataset-specific answer extractor function."""

    def extractor(text: str) -> str:
        if dataset_name in ["aime24", "aime25"]:
            ans = extract_answer_qwq(text)
            if not ans or ans == "[invalid]":
                ans = extract_answer_llm(text)
            return ans or ""
        elif dataset_name == "gpqa_diamond":
            return extract_gpqa_answer(text) or ""
        elif dataset_name == "livebench_reasoning":
            return extract_livebench_answer(text) or ""
        else:
            return text.strip()

    return extractor


def evaluate_answer(dataset_name: str, response: str, gt_raw):
    """Evaluate a single response against ground truth.

    Returns:
        (accept, llm_answer, gt_answer)
    """
    if dataset_name == "gpqa_diamond":
        llm_answer = extract_gpqa_answer(response)
        gt_answer = str(gt_raw).strip().upper()
        accept = llm_answer == gt_answer
    elif dataset_name == "livebench_reasoning":
        gt_answer = str(gt_raw)
        llm_answer = extract_livebench_answer(response)
        accept = check_livebench_match(llm_answer, gt_answer)
    elif dataset_name in ["aime24", "aime25"]:
        gt_answer = str(gt_raw)
        llm_answer = extract_answer_qwq(response)
        if not llm_answer or llm_answer == "[invalid]":
            llm_answer = extract_answer_llm(response)
        accept = False
        if is_float(gt_answer) and is_float(llm_answer):
            try:
                accept = int(round(float(gt_answer))) == int(round(float(llm_answer)))
            except Exception:
                accept = False
    else:
        gt_answer = str(gt_raw)
        llm_answer = extract_answer_qwq(response) or extract_answer_llm(response) or ""
        accept = llm_answer.strip() == gt_answer.strip()

    return accept, llm_answer, gt_answer


# ─── Main ───


def main():
    parser = argparse.ArgumentParser(
        description="DTS / Standard Benchmark Runner (vLLM)"
    )

    parser.add_argument(
        "mode",
        choices=["dts", "standard"],
        help="'dts' for tree decoding or 'standard' for autoregressive baseline",
    )

    # Model and dataset
    parser.add_argument(
        "--model_name", type=str, default="1.5B",
        choices=["1.5B", "7B", "phi-4-mini-reasoning", "qwen30p6"],
        help="Model config key from config.yaml",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="aime24",
        choices=["aime24", "aime25", "gpqa_diamond", "livebench_reasoning"],
        help="Dataset config key from config.yaml",
    )
    parser.add_argument(
        "--config_file", type=str, default="../configs/config.yaml",
        help="Path to config.yaml",
    )

    # DTS parameters
    parser.add_argument("-e", "--entropy_threshold", type=float, default=2.5,
                        help="Branch when entropy H <= this")
    parser.add_argument("-v", "--varentropy_threshold", type=float, default=1.5,
                        help="Branch when varentropy V > this")
    parser.add_argument("-k", "--branch_top_k", type=int, default=3,
                        help="Number of children at each branch point")
    parser.add_argument("-a", "--max_active_hyps", type=int, default=48,
                        help="Freeze branching at this many hypotheses")
    parser.add_argument("--num_traces", type=int, default=8,
                        help="Traces to collect (1=Greedy, 8=Stable)")
    parser.add_argument("--entropy_temp", type=float, default=0.6,
                        help="Entropy temperature (default 0.6)")
    parser.add_argument("--early_stop_min_ratio", type=float, default=0.4,
                        help="Early stop if top answer ratio >= this")
    parser.add_argument("--early_stop_patience", type=int, default=4,
                        help="Min finished traces before early stop check")

    # Shared parameters
    parser.add_argument("-m", "--max_new_tokens", type=int, default=32768,
                        help="Maximum tokens to generate")
    parser.add_argument("-t", "--temperature", type=float, default=0.6,
                        help="Sampling temperature")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="Initial random seed")
    parser.add_argument("-n", "--num_trials", type=int, default=1,
                        help="Number of trials (seeds)")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional tag for output filename")

    # vLLM engine settings
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=40960)
    parser.add_argument("--top_logprobs", type=int, default=20,
                        help="Top logprobs to request (vLLM max=20)")
    parser.add_argument("--enforce_eager", action="store_true", default=False,
                        help="Disable CUDA graphs (recommended for DTS)")

    args = parser.parse_args()

    # ─── Load config ───
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    model_config = config["models"][args.model_name]
    dataset_config = config["datasets"][args.dataset_name]
    output_config = config["output"]

    model_name_hf = model_config["model_name"]

    print("=" * 60)
    print(f"Mode:               {args.mode}")
    print(f"Model:              {model_name_hf} ({args.model_name})")
    print(f"Dataset:            {args.dataset_name}")
    print(f"Seed:               {args.seed}")
    print(f"Trials:             {args.num_trials}")
    print(f"Temperature:        {args.temperature}")
    print(f"Max new tokens:     {args.max_new_tokens}")
    if args.mode == "dts":
        print(f"Entropy threshold:  {args.entropy_threshold}")
        print(f"Varentropy thresh:  {args.varentropy_threshold}")
        print(f"Branch top-k:       {args.branch_top_k}")
        print(f"Max active hyps:    {args.max_active_hyps}")
        print(f"Num traces:         {args.num_traces}")
        print(f"Entropy temp:       {args.entropy_temp}")
    print("=" * 60)

    # ─── Load dataset ───
    dataset_name_hf = dataset_config["dataset_name"]
    dataset_split = dataset_config.get("split", "train")
    prompt_tail = dataset_config["prompt_tail"]
    prompt_key = dataset_config.get("prompt_key", "Problem")
    answer_key = dataset_config.get("answer_key", "Answer")

    print(f"\nLoading dataset: {dataset_name_hf} (split: {dataset_split})...")
    if dataset_name_hf == "Idavidrein/gpqa":
        test_examples = load_dataset(
            dataset_name_hf, "gpqa_diamond", split=dataset_split
        )
    else:
        test_examples = load_dataset(dataset_name_hf, split=dataset_split)
    test_examples = list(test_examples)
    print(f"Loaded {len(test_examples)} examples")

    # ─── Initialize model ───
    print(f"\nLoading model: {model_name_hf}...")

    if args.mode == "dts":
        decoder = TreeDecoder(
            model_name=model_name_hf,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            seed=args.seed,
            enforce_eager=args.enforce_eager,
        )
        tokenizer = decoder.tokenizer
    else:
        llm = LLM(
            model=model_name_hf,
            trust_remote_code=True,
            enable_prefix_caching=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            seed=args.seed,
        )
        tokenizer = llm.get_tokenizer()

        # Collect stop token IDs for standard mode
        stop_token_ids = []
        if tokenizer.eos_token_id is not None:
            stop_token_ids.append(tokenizer.eos_token_id)
        for tok_str in ["<|end|>", "<|endoftext|>"]:
            tid = tokenizer.convert_tokens_to_ids(tok_str)
            if tid is not None and tid != tokenizer.unk_token_id:
                stop_token_ids.append(tid)

    print("Model loaded successfully")

    # ─── Setup ───
    answer_extractor_fn = make_answer_extractor(args.dataset_name)
    seeds = [args.seed + i for i in range(args.num_trials)]

    all_accuracy = []
    all_branch_events = []
    all_time = []
    all_token_num = []
    answers = []
    all_detailed_candidates = []

    # ─── Trial loop ───
    for trial_idx, s in enumerate(seeds):
        print(f"\n{'─'*50}")
        print(f"  Trial {trial_idx + 1}/{args.num_trials} (seed={s})")
        print(f"{'─'*50}")

        random.seed(s)
        np.random.seed(s)

        correct = 0
        num_tokens = 0
        branch_events = 0
        all_responses = []
        all_stats = []
        gt_answers = []
        selected_indices = []

        t0 = time.time()

        for index, example in enumerate(test_examples):
            # ── Format prompt ──
            prompt_content = ""

            if args.dataset_name == "livebench_reasoning":
                prompt_content = str(example[prompt_key][0])
                gt_answers.append(str(example[answer_key]))

            elif args.dataset_name == "gpqa_diamond":
                question_text = example[prompt_key]
                correct_opt = example[answer_key]

                options = [
                    example[answer_key],
                    example["Incorrect Answer 1"],
                    example["Incorrect Answer 2"],
                    example["Incorrect Answer 3"],
                ]
                options = [
                    str(opt).strip() if opt is not None else "" for opt in options
                ]
                correct_opt_cleaned = str(correct_opt).strip()

                random.shuffle(options)

                try:
                    correct_index = options.index(correct_opt_cleaned)
                except ValueError:
                    print(
                        f"  [WARN] Correct answer not found in shuffled options "
                        f"for example {index}, skipping"
                    )
                    continue

                correct_letter_gt = ["A", "B", "C", "D"][correct_index]
                gt_answers.append(correct_letter_gt)

                prompt_content = QUERY_TEMPLATE_MULTICHOICE.format(
                    Question=question_text,
                    A=options[0],
                    B=options[1],
                    C=options[2],
                    D=options[3],
                )

            else:
                prompt_content = example[prompt_key] + prompt_tail
                gt_answers.append(example[answer_key])

            # ── Apply chat template ──
            messages = [{"role": "user", "content": prompt_content}]

            if args.mode == "dts":
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            else:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            # ── Generate ──
            if args.mode == "dts":
                out = decoder.generate(
                    text,
                    entropy_threshold=args.entropy_threshold,
                    varentropy_threshold=args.varentropy_threshold,
                    branch_top_k=args.branch_top_k,
                    max_active_hyps=args.max_active_hyps,
                    max_new_tokens=args.max_new_tokens,
                    entropy_temp=args.entropy_temp,
                    temperature=args.temperature,
                    num_traces=args.num_traces,
                    top_logprobs=args.top_logprobs,
                    answer_extractor=answer_extractor_fn,
                    seed=s,
                )

                traces = out.get("traces", [])
                global_stats = out.get("stats", {})

                # Re-extract answers from all traces
                for tr in traces:
                    ans = answer_extractor_fn(tr["text"])
                    tr["answer"] = ans

                # Majority vote to select winner
                winner_answer = majority_vote_answers(traces)

                # Build trace info for detailed output
                all_traces_info = []
                for tr in traces:
                    all_traces_info.append({
                        "text": tr["text"],
                        "answer": tr.get("answer"),
                        "token_count": tr.get("length", len(tr.get("tokens", []))),
                        "logprob": tr.get("logprob"),
                    })

                # Select best trace (first one matching winner answer)
                best_trace = None
                if winner_answer:
                    for tr in traces:
                        if tr.get("answer") == winner_answer:
                            best_trace = tr
                            break
                if best_trace is None and traces:
                    best_trace = traces[0]

                if best_trace:
                    best_text = best_trace["text"]
                    gen_len = best_trace.get("length", 0)
                else:
                    best_text = ""
                    gen_len = 0

                branch_ev = global_stats.get("branch_events", 0)

                all_responses.append(best_text)
                all_stats.append({
                    "generated_len": gen_len,
                    "branch_events": branch_ev,
                    "num_candidates": len(traces),
                    "total_branches_created": global_stats.get(
                        "total_branches_created", 0
                    ),
                    "all_candidate_traces": all_traces_info,
                })
                selected_indices.append(index)

            elif args.mode == "standard":
                params = SamplingParams(
                    temperature=args.temperature,
                    max_tokens=args.max_new_tokens,
                    stop_token_ids=stop_token_ids,
                    seed=s,
                )

                outputs = llm.generate([text], params)
                out = outputs[0].outputs[0]
                gen_text = out.text
                gen_len = len(out.token_ids)

                all_responses.append(gen_text)
                all_stats.append({
                    "generated_len": gen_len,
                    "branch_events": 0,
                })
                selected_indices.append(index)

            # Progress
            if (index + 1) % 5 == 0 or index == len(test_examples) - 1:
                elapsed = time.time() - t0
                print(f"  [{index + 1}/{len(test_examples)}] elapsed={elapsed:.0f}s")

        t1 = time.time()
        trial_time = t1 - t0

        # ── Evaluate ──
        print("\n  Evaluating responses...")
        sorted_indices = sorted(selected_indices)

        for i, real_idx in enumerate(sorted_indices):
            response = all_responses[i]
            stat = all_stats[i]
            gt_raw = gt_answers[i]

            accept, llm_answer, gt_answer = evaluate_answer(
                args.dataset_name, response, gt_raw
            )

            if accept:
                correct += 1

            # Detailed candidate info (DTS mode)
            candidate_traces = stat.get("all_candidate_traces")
            if candidate_traces:
                detailed_cands = []
                for cand in candidate_traces:
                    cand_accept, cand_llm, cand_gt = evaluate_answer(
                        args.dataset_name, cand["text"], gt_raw
                    )
                    detailed_cands.append({
                        "llm_answer": cand_llm,
                        "accept": cand_accept,
                        "llm_response": cand["text"],
                        "tokens": cand.get("token_count"),
                        "logprob": cand.get("logprob"),
                    })

                all_detailed_candidates.append({
                    "seed": s,
                    "index": real_idx + 1,
                    "question": test_examples[real_idx][prompt_key],
                    "gt_answer": gt_answer,
                    "candidates": detailed_cands,
                })

            answers.append({
                "question": test_examples[real_idx][prompt_key],
                "gt_answer": gt_answer,
                "llm_answer": llm_answer,
                "accept": accept,
                "llm_response": response,
                "tokens": stat["generated_len"],
            })

            num_tokens += stat["generated_len"]
            branch_events += stat["branch_events"]

        n_examples = len(selected_indices) or 1
        avg_tokens = num_tokens / n_examples
        avg_branch = branch_events / n_examples
        accuracy = correct / n_examples

        print(
            f"\n  Trial seed={s}: "
            f"Acc={accuracy:.4f} ({correct}/{n_examples}), "
            f"Avg Tokens={avg_tokens:.1f}, "
            f"Avg Branches={avg_branch:.1f}, "
            f"Time={trial_time:.1f}s"
        )

        all_accuracy.append(accuracy)
        all_branch_events.append(avg_branch)
        all_time.append(trial_time)
        all_token_num.append(avg_tokens)

    # ─── Aggregate ───
    n_trials = len(all_accuracy) or 1
    final_accuracy = sum(all_accuracy) / n_trials
    final_branch = sum(all_branch_events) / n_trials
    final_time = sum(all_time) / n_trials
    final_tokens = sum(all_token_num) / n_trials

    answers.insert(0, {
        "accuracy": final_accuracy,
        "average branch events": final_branch,
        "average new tokens": final_tokens,
        "time": final_time,
    })

    # ─── Save results ───
    base_fname = f"{args.dataset_name}_{args.mode}_{args.model_name}"

    if args.mode == "dts":
        base_fname += (
            f"_entro{fmt_float(args.entropy_threshold)}"
            f"_k{args.branch_top_k}"
            f"_max_active_hyps{args.max_active_hyps}"
            f"_traces{args.num_traces}"
        )

    base_fname += (
        f"_temp{fmt_float(args.temperature)}"
        f"_trials{args.num_trials}"
        f"_seed{args.seed}"
    )
    if args.tag:
        base_fname += f"_{args.tag}"
    base_fname += ".json"

    out_dir = Path(output_config["base_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / base_fname

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  Results Summary")
    print(f"{'='*60}")
    print(f"Final Accuracy:     {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)")
    print(f"Avg Branch Events:  {final_branch:.2f}")
    print(f"Avg New Tokens:     {final_tokens:.1f}")
    print(f"Avg Wall Time:      {final_time:.1f}s")
    print(f"Saved to:           {out_path}")

    if all_detailed_candidates:
        cand_path = out_dir / base_fname.replace(".json", "_candidates.json")
        with open(cand_path, "w", encoding="utf-8") as f:
            json.dump(all_detailed_candidates, f, ensure_ascii=False, indent=2)
        print(f"Candidates saved:   {cand_path}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
