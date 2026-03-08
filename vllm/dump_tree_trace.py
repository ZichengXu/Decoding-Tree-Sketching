#!/usr/bin/env python3
"""
Dump DTS tree traces with branch convergence analysis.

For each question × seed, records every branch point with:
- Token position, entropy H, varentropy V at the split
- Per-child: token sequence after branch, final answer, correctness
- Convergence metrics: how quickly branches diverge/reconverge,
  text similarity, answer agreement, aleatoric vs epistemic classification

Usage:
  python dump_tree_trace.py \
    --model_name 1.5B --dataset_name aime24 \
    -s 0 -n 3 --num_questions 5 \
    -e 2.5 -k 3 -a 48 -t 0.6 \
    -o tree_traces.json
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import json
import time
import random
import argparse
from itertools import combinations
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

import yaml
import numpy as np
from datasets import load_dataset
from vllm import SamplingParams, TokensPrompt

from decoding_tree_sketching.tree_decoder import (
    TreeDecoder, HypNode, compute_entropy_varentropy, get_branch_tokens,
)
from decoding_tree_sketching.utils.eval_utils import (
    extract_answer_qwq, extract_answer_llm, extract_gpqa_answer,
)


# ─── Prompt formatting ──────────────────────────────────────────────

QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()


def make_answer_extractor(dataset_name):
    def extractor(text):
        if dataset_name in ["aime24", "aime25"]:
            ans = extract_answer_qwq(text)
            if not ans or ans == "[invalid]":
                ans = extract_answer_llm(text)
            return ans or ""
        elif dataset_name == "gpqa_diamond":
            return extract_gpqa_answer(text) or ""
        else:
            return text.strip()
    return extractor


def format_prompt(example, dataset_name, dataset_config, seed):
    """Format a single example into a prompt string + ground truth answer."""
    prompt_key = dataset_config.get("prompt_key", "Problem")
    answer_key = dataset_config.get("answer_key", "Answer")
    prompt_tail = dataset_config.get("prompt_tail", "")

    if dataset_name == "gpqa_diamond":
        random.seed(seed)
        options = [
            example[answer_key],
            example["Incorrect Answer 1"],
            example["Incorrect Answer 2"],
            example["Incorrect Answer 3"],
        ]
        options = [str(o).strip() if o else "" for o in options]
        correct = str(example[answer_key]).strip()
        random.shuffle(options)
        gt_answer = ["A", "B", "C", "D"][options.index(correct)]
        prompt_content = QUERY_TEMPLATE_MULTICHOICE.format(
            Question=example[prompt_key],
            A=options[0], B=options[1], C=options[2], D=options[3],
        )
    elif dataset_name == "livebench_reasoning":
        prompt_content = str(example[prompt_key][0])
        gt_answer = str(example[answer_key])
    else:
        prompt_content = example[prompt_key] + prompt_tail
        gt_answer = str(example[answer_key])

    return prompt_content, gt_answer


# ─── Convergence analysis ───────────────────────────────────────────

RECONVERGE_WINDOW = 5


def pairwise_diverge_step(seq_a: List[int], seq_b: List[int]) -> int:
    """First index where two token sequences differ. Returns min(len) if identical up to shorter."""
    for i in range(min(len(seq_a), len(seq_b))):
        if seq_a[i] != seq_b[i]:
            return i
    return min(len(seq_a), len(seq_b))


def pairwise_reconverge_step(
    seq_a: List[int], seq_b: List[int], diverge: int, window: int = RECONVERGE_WINDOW,
) -> Optional[int]:
    """After diverge point, find first position where W consecutive tokens match. None if never."""
    max_len = min(len(seq_a), len(seq_b))
    for start in range(diverge + 1, max_len - window + 1):
        if seq_a[start:start + window] == seq_b[start:start + window]:
            return start
    return None


def token_jaccard(seq_a: List[int], seq_b: List[int]) -> float:
    """Jaccard similarity of token ID sets."""
    set_a, set_b = set(seq_a), set(seq_b)
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def compute_convergence(
    branch_seqs: List[List[int]],
    branch_answers: List[Optional[str]],
) -> Dict[str, Any]:
    """Compute convergence metrics for a set of branch token sequences."""
    n = len(branch_seqs)
    if n < 2:
        return {
            "pairwise_diverge_steps": [],
            "mean_diverge_step": None,
            "pairwise_reconverge_steps": [],
            "final_text_similarity": None,
            "same_answer_ratio": None,
            "type": "single_branch",
        }

    diverge_steps = []
    reconverge_steps = []
    similarities = []

    for i, j in combinations(range(n), 2):
        d = pairwise_diverge_step(branch_seqs[i], branch_seqs[j])
        diverge_steps.append(d)
        r = pairwise_reconverge_step(branch_seqs[i], branch_seqs[j], d)
        reconverge_steps.append(r)
        similarities.append(token_jaccard(branch_seqs[i], branch_seqs[j]))

    mean_diverge = sum(diverge_steps) / len(diverge_steps) if diverge_steps else 0

    # Same answer ratio
    valid_answers = [a for a in branch_answers if a]
    if len(valid_answers) >= 2:
        same_pairs = 0
        total_pairs = 0
        for i, j in combinations(range(len(valid_answers)), 2):
            total_pairs += 1
            if valid_answers[i] == valid_answers[j]:
                same_pairs += 1
        same_answer_ratio = same_pairs / total_pairs
    elif len(valid_answers) == 1:
        same_answer_ratio = 1.0
    else:
        same_answer_ratio = None

    mean_sim = sum(similarities) / len(similarities) if similarities else 0

    # Classification
    if same_answer_ratio is not None and same_answer_ratio >= 0.8:
        branch_type = "aleatoric"
    elif mean_diverge > 10:
        branch_type = "aleatoric"
    else:
        branch_type = "epistemic"

    return {
        "pairwise_diverge_steps": diverge_steps,
        "mean_diverge_step": round(mean_diverge, 2),
        "pairwise_reconverge_steps": reconverge_steps,
        "final_text_similarity": round(mean_sim, 4),
        "same_answer_ratio": round(same_answer_ratio, 4) if same_answer_ratio is not None else None,
        "type": branch_type,
    }


# ─── Instrumented generation ────────────────────────────────────────

def generate_with_trace(
    decoder: TreeDecoder,
    prompt: str,
    *,
    entropy_threshold: float = 2.5,
    varentropy_threshold: float = 1.5,
    branch_top_k: int = 3,
    max_new_tokens: int = 32768,
    max_active_hyps: int = 48,
    temperature: float = 0.6,
    top_logprobs: int = 20,
    answer_extractor=None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run DTS generation with full branch tracking and lineage."""

    tokenizer = decoder.tokenizer
    llm = decoder.llm
    stop_token_ids = decoder.stop_token_ids

    prompt_ids = tokenizer.encode(prompt)
    prompt_len = len(prompt_ids)

    # Root hypothesis
    root = HypNode(
        tokens=list(prompt_ids),
        prompt_len=prompt_len,
        logprob=0.0,
        entropies=[],
        varentropies=[],
    )

    # Lineage tracking: id(hyp) -> metadata
    hyp_meta: Dict[int, dict] = {}
    hyp_meta[id(root)] = {"lineage": [], "branch_token_idx": 0}

    active = [root]
    finished: List[HypNode] = []
    frozen = False
    branch_events = 0

    # Branch point log
    branch_points_log: List[dict] = []

    step_params = SamplingParams(
        temperature=temperature,
        max_tokens=1,
        logprobs=top_logprobs,
        seed=seed,
    )

    # Phase 1: token-by-token with branching
    t0 = time.time()
    for step in range(max_new_tokens):
        if not active:
            break

        prompts_batch = [TokensPrompt(prompt_token_ids=h.tokens) for h in active]
        outputs = llm.generate(prompts_batch, sampling_params=step_params, use_tqdm=False)

        next_active: List[HypNode] = []
        current_total = len(active)

        for hyp, output in zip(active, outputs):
            result = output.outputs[0]
            if result.logprobs is None or len(result.logprobs) == 0:
                continue
            logprobs_dict = result.logprobs[0]
            if logprobs_dict is None:
                continue

            sampled_token = result.token_ids[0]
            token_logprob = 0.0
            if sampled_token in logprobs_dict:
                token_logprob = logprobs_dict[sampled_token].logprob

            H, V = compute_entropy_varentropy(logprobs_dict, entropy_temp=0.6)

            should_branch = (
                not frozen
                and H <= entropy_threshold
                and V > varentropy_threshold
            )

            if should_branch:
                all_children_tokens = get_branch_tokens(
                    logprobs_dict, sampled_token, branch_top_k,
                )

                bp_idx = branch_events
                branch_events += 1
                current_total += len(all_children_tokens) - 1

                if not frozen and current_total >= max_active_hyps:
                    frozen = True

                parent_meta = hyp_meta[id(hyp)]
                parent_token_idx = len(hyp.tokens) - prompt_len

                bp_record = {
                    "bp_idx": bp_idx,
                    "branch_point_token_idx": parent_token_idx,
                    "entropy_at_split": round(H, 4),
                    "varentropy_at_split": round(V, 4),
                    "children_token_ids": all_children_tokens,
                    "children_logprobs": [],
                }

                for child_idx, child_token in enumerate(all_children_tokens):
                    child_lp = 0.0
                    if child_token in logprobs_dict:
                        child_lp = logprobs_dict[child_token].logprob
                    bp_record["children_logprobs"].append(round(child_lp, 4))

                    child = HypNode(
                        tokens=hyp.tokens + [child_token],
                        prompt_len=prompt_len,
                        logprob=hyp.logprob + child_lp,
                        entropies=hyp.entropies + [H],
                        varentropies=hyp.varentropies + [V],
                    )
                    hyp_meta[id(child)] = {
                        "lineage": parent_meta["lineage"] + [(bp_idx, child_idx)],
                        "branch_token_idx": parent_token_idx,
                    }

                    if child_token in stop_token_ids:
                        child.finished = True
                        child.finish_reason = "stop"
                        finished.append(child)
                    else:
                        next_active.append(child)

                branch_points_log.append(bp_record)

            else:
                child = HypNode(
                    tokens=hyp.tokens + [sampled_token],
                    prompt_len=prompt_len,
                    logprob=hyp.logprob + token_logprob,
                    entropies=hyp.entropies + [H],
                    varentropies=hyp.varentropies + [V],
                )
                # Inherit parent's lineage
                hyp_meta[id(child)] = dict(hyp_meta[id(hyp)])

                if sampled_token in stop_token_ids:
                    child.finished = True
                    child.finish_reason = "stop"
                    finished.append(child)
                else:
                    next_active.append(child)

        if len(next_active) > max_active_hyps:
            next_active.sort(key=lambda h: h.logprob, reverse=True)
            # Preserve meta for pruned-away hyps is fine (they just get GC'd)
            next_active = next_active[:max_active_hyps]

        active = next_active
        if frozen:
            break

    phase1_time = time.time() - t0
    phase1_steps = step + 1 if (active or finished) else 0
    print(
        f"  [Phase 1] steps={phase1_steps}, active={len(active)}, "
        f"finished={len(finished)}, branches={branch_events}, "
        f"frozen={frozen}, time={phase1_time:.1f}s"
    )

    # Phase 2: batch complete remaining hypotheses
    t1 = time.time()
    if active:
        remaining_tokens = max_new_tokens - phase1_steps
        if remaining_tokens > 0:
            prompts = [TokensPrompt(prompt_token_ids=h.tokens) for h in active]
            params = SamplingParams(
                temperature=temperature,
                max_tokens=remaining_tokens,
                seed=seed,
                stop_token_ids=list(stop_token_ids),
            )
            outputs = llm.generate(prompts, params, use_tqdm=False)
            for hyp, output in zip(active, outputs):
                result = output.outputs[0]
                hyp.tokens.extend(result.token_ids)
                hyp.finished = True
                hyp.finish_reason = result.finish_reason or "stop"
                finished.append(hyp)
        else:
            for hyp in active:
                hyp.finished = True
                hyp.finish_reason = "length"
                finished.append(hyp)

    phase2_time = time.time() - t1
    print(f"  [Phase 2] completed={len(finished)}, time={phase2_time:.1f}s")

    # Extract answers for all finished hyps
    for hyp in finished:
        if answer_extractor and hyp.answer is None:
            gen_ids = hyp.tokens[prompt_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            try:
                ans = answer_extractor(gen_text)
                if ans:
                    hyp.answer = str(ans).strip()
            except Exception:
                pass

    # Post-process: build per-branch-point output with convergence
    results = []
    for bp in branch_points_log:
        bp_idx = bp["bp_idx"]
        branch_token_idx = bp["branch_point_token_idx"]

        # Group finished hyps by which child they took at this branch point
        child_groups: Dict[int, List[HypNode]] = defaultdict(list)
        for hyp in finished:
            meta = hyp_meta.get(id(hyp))
            if meta is None:
                continue
            for lin_bp_idx, lin_child_idx in meta["lineage"]:
                if lin_bp_idx == bp_idx:
                    child_groups[lin_child_idx].append(hyp)
                    break

        # For each child group, pick best hyp and extract sequence after branch
        branch_seqs = []
        branch_answers = []
        branches_out = []

        for child_idx in range(len(bp["children_token_ids"])):
            group = child_groups.get(child_idx, [])
            if not group:
                continue

            # Pick best by logprob
            best = max(group, key=lambda h: h.logprob)
            # Tokens after the branch point
            # branch_token_idx is how many tokens were generated before branch
            # so the branch child token starts at prompt_len + branch_token_idx
            after_branch_start = prompt_len + branch_token_idx
            tokens_after = best.tokens[after_branch_start:]
            token_strs = [tokenizer.decode([t]) for t in tokens_after[:50]]  # first 50 for readability

            branch_seqs.append(tokens_after)
            branch_answers.append(best.answer)

            branches_out.append({
                "child_token": tokenizer.decode([bp["children_token_ids"][child_idx]]),
                "child_token_id": bp["children_token_ids"][child_idx],
                "child_logprob": bp["children_logprobs"][child_idx],
                "tokens_after_branch": token_strs,
                "length_after_branch": len(tokens_after),
                "final_answer": best.answer,
                "correct": None,  # filled in by caller with gt_answer
            })

        convergence = compute_convergence(branch_seqs, branch_answers)

        results.append({
            "branch_point_token_idx": branch_token_idx,
            "entropy_at_split": bp["entropy_at_split"],
            "varentropy_at_split": bp["varentropy_at_split"],
            "convergence": convergence,
            "branches": branches_out,
        })

    return {
        "branch_points": results,
        "num_branch_points": len(results),
        "total_finished": len(finished),
        "phase1_steps": phase1_steps,
        "phase1_time": round(phase1_time, 1),
        "phase2_time": round(phase2_time, 1),
    }


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DTS branch convergence trace dump")
    parser.add_argument("--model_name", type=str, default="1.5B")
    parser.add_argument("--dataset_name", type=str, default="aime24")
    parser.add_argument("--config_file", type=str, default="../configs/config.yaml")
    parser.add_argument("--num_questions", type=int, default=5,
                        help="Number of questions to process (-1 for all)")
    parser.add_argument("-s", "--seed", type=int, default=0, help="Starting seed")
    parser.add_argument("-n", "--num_seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("-t", "--temperature", type=float, default=0.6)
    parser.add_argument("-e", "--entropy_threshold", type=float, default=2.5)
    parser.add_argument("-v", "--varentropy_threshold", type=float, default=1.5)
    parser.add_argument("-k", "--branch_top_k", type=int, default=3)
    parser.add_argument("-a", "--max_active_hyps", type=int, default=48)
    parser.add_argument("-m", "--max_new_tokens", type=int, default=32768)
    parser.add_argument("--top_logprobs", type=int, default=20)
    parser.add_argument("-o", "--output", type=str, default="tree_traces.json")
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = yaml.safe_load(f)

    model_config = config["models"][args.model_name]
    dataset_config = config["datasets"][args.dataset_name]
    model_name_hf = model_config["model_name"]

    # Load dataset
    ds_name = dataset_config["dataset_name"]
    ds_split = dataset_config.get("split", "train")
    if ds_name == "Idavidrein/gpqa":
        examples = list(load_dataset(ds_name, "gpqa_diamond", split=ds_split))
    else:
        examples = list(load_dataset(ds_name, split=ds_split))

    if args.num_questions > 0:
        examples = examples[:args.num_questions]

    print(f"Dataset: {args.dataset_name}, {len(examples)} questions")
    print(f"Model: {model_name_hf}")
    print(f"Seeds: {args.seed} to {args.seed + args.num_seeds - 1}")

    # Init decoder
    decoder = TreeDecoder(
        model_name=model_name_hf,
        gpu_memory_utilization=0.9,
        max_model_len=40960,
        seed=args.seed,
        enforce_eager=True,
    )

    answer_extractor = make_answer_extractor(args.dataset_name)
    all_results = []

    for seed in range(args.seed, args.seed + args.num_seeds):
        for q_idx, example in enumerate(examples):
            prompt_content, gt_answer = format_prompt(
                example, args.dataset_name, dataset_config, seed,
            )

            messages = [{"role": "user", "content": prompt_content}]
            text = decoder.tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=True,
            )

            print(f"\n{'='*50}")
            q_preview = str(example.get(dataset_config.get("prompt_key", "Problem"), ""))[:80]
            print(f"  Q{q_idx} seed={seed}: {q_preview}...")
            print(f"  GT: {gt_answer}")

            result = generate_with_trace(
                decoder, text,
                entropy_threshold=args.entropy_threshold,
                varentropy_threshold=args.varentropy_threshold,
                branch_top_k=args.branch_top_k,
                max_new_tokens=args.max_new_tokens,
                max_active_hyps=args.max_active_hyps,
                temperature=args.temperature,
                top_logprobs=args.top_logprobs,
                answer_extractor=answer_extractor,
                seed=seed,
            )

            # Fill in correctness
            for bp in result["branch_points"]:
                for br in bp["branches"]:
                    if br["final_answer"] is not None and gt_answer:
                        try:
                            if args.dataset_name in ["aime24", "aime25"]:
                                br["correct"] = (
                                    int(round(float(br["final_answer"])))
                                    == int(round(float(gt_answer)))
                                )
                            else:
                                br["correct"] = br["final_answer"].strip() == gt_answer.strip()
                        except (ValueError, TypeError, OverflowError):
                            br["correct"] = False
                    else:
                        br["correct"] = None

            # Summary stats
            n_bp = result["num_branch_points"]
            types = [bp["convergence"]["type"] for bp in result["branch_points"]]
            n_aleatoric = types.count("aleatoric")
            n_epistemic = types.count("epistemic")
            print(
                f"  Branch points: {n_bp}, "
                f"aleatoric: {n_aleatoric}, epistemic: {n_epistemic}"
            )

            prompt_key = dataset_config.get("prompt_key", "Problem")
            all_results.append({
                "question": str(example.get(prompt_key, ""))[:500],
                "gt_answer": gt_answer,
                "seed": seed,
                "question_idx": q_idx,
                **result,
            })

            # Incremental save after each question-seed
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Final save
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"Saved {len(all_results)} traces to {args.output}")

    # Global summary
    all_types = []
    all_varentropies_by_type = {"aleatoric": [], "epistemic": []}
    for entry in all_results:
        for bp in entry["branch_points"]:
            t = bp["convergence"]["type"]
            all_types.append(t)
            if t in all_varentropies_by_type:
                all_varentropies_by_type[t].append(bp["varentropy_at_split"])

    total = len(all_types)
    if total > 0:
        n_a = all_types.count("aleatoric")
        n_e = all_types.count("epistemic")
        print(f"\nGlobal: {total} branch points")
        print(f"  Aleatoric: {n_a} ({100*n_a/total:.1f}%)")
        print(f"  Epistemic: {n_e} ({100*n_e/total:.1f}%)")
        for t in ["aleatoric", "epistemic"]:
            vals = all_varentropies_by_type[t]
            if vals:
                print(f"  {t} varentropy: mean={np.mean(vals):.3f}, std={np.std(vals):.3f}")


if __name__ == "__main__":
    main()
