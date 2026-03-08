"""
Two-phase entropy-guided tree decoder for vLLM.

Implements the Decoding Tree Sketching (DTS) algorithm on top of vLLM:

Phase 1 (Branching):
    Generate one token at a time for all active hypotheses. At each step,
    compute Shannon entropy H and varentropy V from the model's logprobs.
    If H is low and V is high (confident overall but uncertain between a
    few top candidates), branch into top-k children. Once the total number
    of hypotheses reaches max_active_hyps, freeze — no further branching.

Phase 2 (Completion):
    After freeze, all remaining hypotheses are completed in a single
    batched vLLM generate() call, avoiding per-token overhead.

KV cache reuse is handled automatically by vLLM's prefix caching (APC):
each hypothesis is submitted as a full token sequence, and vLLM detects
shared prefixes across hypotheses to avoid redundant computation.
"""

import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Callable, Set, Tuple

from vllm import LLM, SamplingParams, TokensPrompt


# ─── Data Structures ────────────────────────────────────────────────


@dataclass
class HypNode:
    """A single hypothesis (path) in the decoding tree."""
    tokens: List[int]              # Full token sequence (prompt + generated)
    prompt_len: int                # Length of the prompt portion
    logprob: float                 # Cumulative log probability of generated tokens
    entropies: List[float]         # Per-step entropy history
    varentropies: List[float]      # Per-step varentropy history
    finished: bool = False
    finish_reason: Optional[str] = None
    answer: Optional[str] = None


# ─── Utility Functions ───────────────────────────────────────────────


def compute_entropy_varentropy(logprobs_dict: dict, entropy_temp: float = 0.6) -> Tuple[float, float]:
    """
    Compute Shannon entropy H and varentropy V from top-k logprobs.

    Since vLLM returns raw logprobs at T=1.0, we rescale them by
    entropy_temp to match the original DTS paper's entropy computation
    (which uses entropy_temp=0.6 for all datasets).

    Math: softmax(raw_lp / T) = softmax(logit / T) because the constant
    offset from log-normalization cancels out in softmax.

    Args:
        logprobs_dict: {token_id: LogprobInfo} from vLLM output (raw, T=1).
        entropy_temp: Temperature for entropy computation (default 0.6).

    Returns:
        (H, V) where H is Shannon entropy and V is varentropy
        (variance of information content across the distribution).
    """
    if not logprobs_dict:
        return 0.0, 0.0

    # Collect raw T=1 probabilities from vLLM logprobs
    raw_probs = []  # exact T=1 probabilities for top-k tokens
    for token_id, logprob_info in logprobs_dict.items():
        p = math.exp(logprob_info.logprob)
        if p > 0:
            raw_probs.append(p)

    if not raw_probs:
        return 0.0, 0.0

    # Remaining probability mass at T=1 (tokens outside top-k)
    remaining_t1 = max(0.0, 1.0 - sum(raw_probs))

    # Rescale to entropy_temp: p_i(T=τ) ∝ p_i(T=1)^(1/τ)
    # For the remaining bucket (sum of many small tokens), we approximate
    # it as a single token with probability remaining_t1, so its
    # unnormalized score is remaining_t1^(1/τ). This is a rough
    # approximation, but the remaining mass at T<1 is even smaller
    # than at T=1, so its impact on H and V is minimal.
    inv_temp = 1.0 / entropy_temp
    unnorm = [p ** inv_temp for p in raw_probs]
    unnorm_remaining = remaining_t1 ** inv_temp if remaining_t1 > 1e-10 else 0.0
    Z = sum(unnorm) + unnorm_remaining

    if Z <= 0:
        return 0.0, 0.0

    probs = [u / Z for u in unnorm]
    log_probs = [math.log(p) if p > 0 else -50.0 for p in probs]
    remaining = unnorm_remaining / Z

    # Shannon entropy: H = -sum(p * log(p))
    H = 0.0
    for p, lp in zip(probs, log_probs):
        H -= p * lp
    if remaining > 1e-10:
        H -= remaining * math.log(remaining)

    # Varentropy: V = sum(p * (-log(p) - H)^2)
    V = 0.0
    for p, lp in zip(probs, log_probs):
        V += p * (-lp - H) ** 2
    if remaining > 1e-10:
        V += remaining * (-math.log(remaining) - H) ** 2

    return H, V


def get_branch_tokens(
    logprobs_dict: dict,
    chosen_token: int,
    branch_top_k: int,
) -> List[int]:
    """Select top-k tokens for branching at a decision point.

    Returns the top-k most likely tokens (sorted by logprob descending,
    ties broken by token ID). The already-sampled token is guaranteed
    to be included; if it falls outside top-k, it replaces the lowest.

    Args:
        logprobs_dict: {token_id: LogprobInfo} from vLLM output.
        chosen_token: The token that was actually sampled this step.
        branch_top_k: Number of children to create at this branch point.

    Returns:
        List of token IDs to use as children.
    """
    all_tokens = []
    for token_id, logprob_info in logprobs_dict.items():
        all_tokens.append((logprob_info.logprob, token_id))
    all_tokens.sort(key=lambda x: (-x[0], x[1]))

    top_k_tokens = [tid for _, tid in all_tokens[:branch_top_k]]

    if chosen_token not in top_k_tokens:
        top_k_tokens[-1] = chosen_token

    return top_k_tokens


def majority_vote_answers(traces: List[Dict]) -> Optional[str]:
    """Pick the most common answer among traces (simple plurality vote).

    Args:
        traces: List of trace dicts, each with an optional "answer" field.

    Returns:
        The most frequent answer string, or None if no answers extracted.
    """
    with_answer = [tr for tr in traces if tr.get("answer")]
    if not with_answer:
        return None

    cnt = Counter(tr["answer"] for tr in with_answer)
    winner, _ = cnt.most_common(1)[0]
    return winner


# ─── Tree Decoder ────────────────────────────────────────────────────


class TreeDecoder:
    """
    Two-phase entropy-guided tree decoder using vLLM.

    Phase 1: Token-by-token generation with entropy-guided branching
             until the hypothesis count reaches max_active_hyps (freeze).
    Phase 2: Batch completion of all remaining hypotheses in one call.

    Args:
        model_name: HuggingFace model name or local path.
        gpu_memory_utilization: Fraction of GPU memory for vLLM KV cache.
        max_model_len: Maximum context length (prompt + generation).
        seed: Random seed for reproducibility.
        enforce_eager: Disable CUDA graphs (recommended for variable batch sizes).
    """

    def __init__(
        self,
        model_name: str,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 40960,
        seed: int = 0,
        enforce_eager: bool = True,
    ):
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            seed=seed,
            enforce_eager=enforce_eager,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.seed = seed

        # Collect all stop token IDs from the tokenizer
        self.stop_token_ids: Set[int] = set()
        if self.tokenizer.eos_token_id is not None:
            self.stop_token_ids.add(self.tokenizer.eos_token_id)

        # Model-specific stop tokens (covers DeepSeek, Qwen, Phi families)
        stop_strings = {
            "<|end|>", "<|endoftext|>", "<|im_end|>",
            "<\uff5cend\u2581of\u2581sentence\uff5c>",  # DeepSeek-R1 (Unicode)
        }
        vocab = self.tokenizer.get_vocab()
        for tok_str in stop_strings:
            if tok_str in vocab:
                self.stop_token_ids.add(vocab[tok_str])
        print(f"  [TreeDecoder] stop_token_ids: {self.stop_token_ids}")

    def generate(
        self,
        prompt: str,
        *,
        entropy_threshold: float = 2.5,
        varentropy_threshold: float = 1.5,
        branch_top_k: int = 3,
        max_new_tokens: int = 32768,
        max_active_hyps: int = 48,
        entropy_temp: float = 0.6,
        temperature: float = 0.6,
        num_traces: int = 1,
        top_logprobs: int = 20,
        answer_extractor: Optional[Callable[[str], str]] = None,
        seed: Optional[int] = None,
        early_stop_min_ratio: float = 0.4,
        early_stop_patience: int = 4,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate with two-phase entropy-guided tree decoding.

        Args:
            prompt: Input text (already formatted with chat template).
            entropy_threshold: Branch when Shannon entropy H <= this value.
                Low entropy means the model is fairly confident.
            varentropy_threshold: Branch when varentropy V > this value.
                High varentropy means uncertainty is concentrated on a few tokens.
            branch_top_k: Number of children to create at each branch point.
            max_new_tokens: Maximum tokens to generate per hypothesis.
            max_active_hyps: Freeze branching when this many hypotheses exist.
            entropy_temp: Temperature for entropy/varentropy computation.
                Raw logprobs (T=1) are rescaled to this temperature (default 0.6).
            temperature: Sampling temperature.
            num_traces: Number of completed traces to collect before stopping.
                Use 1 for Greedy mode, 8 for Stable (majority vote) mode.
            top_logprobs: Number of top logprobs to request from vLLM (max 20).
            answer_extractor: Function to extract an answer from generated text.
            seed: Random seed for this generation call.
            early_stop_min_ratio: Stop early if the top answer's share among
                finished traces reaches this ratio (default 0.4).
            early_stop_patience: Minimum finished traces before checking
                early stop condition.

        Returns:
            Dict with "traces" (list of trace dicts) and "stats" (generation stats).
            Each trace has: text, tokens, logprob, finished, length,
            mean_entropy, entropy_history, varentropy_history, answer.
        """
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_len = len(prompt_ids)

        # Start with a single root hypothesis
        active: List[HypNode] = [
            HypNode(
                tokens=list(prompt_ids),
                prompt_len=prompt_len,
                logprob=0.0,
                entropies=[],
                varentropies=[],
            )
        ]
        finished: List[HypNode] = []

        branch_events = 0
        total_branches_created = 0
        max_active_batch = 1
        frozen = False
        phase1_steps = 0

        if entropy_temp != temperature:
            print(
                f"  [NOTE] entropy_temp={entropy_temp} != temperature={temperature}. "
                f"Entropy is rescaled to T={entropy_temp}, sampling uses T={temperature}."
            )

        # Phase 1 sampling params: generate 1 token per step with logprobs
        step_params = SamplingParams(
            temperature=temperature,
            max_tokens=1,
            logprobs=top_logprobs,
            seed=seed,
        )

        # ════════════════════════════════════════════════════════════
        #  Phase 1: Token-by-token generation with branching
        # ════════════════════════════════════════════════════════════
        t_phase1 = time.time()

        step = -1
        for step in range(max_new_tokens):
            if not active or len(finished) >= num_traces:
                break

            max_active_batch = max(max_active_batch, len(active))

            # Generate 1 token for every active hypothesis
            prompts_batch = [
                TokensPrompt(prompt_token_ids=hyp.tokens)
                for hyp in active
            ]
            outputs = self.llm.generate(
                prompts_batch,
                sampling_params=step_params,
                use_tqdm=False,
            )

            # Process each output: decide whether to branch or continue
            next_active: List[HypNode] = []
            current_total = len(active)

            for hyp, output in zip(active, outputs):
                if len(finished) >= num_traces:
                    break

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

                H, V = compute_entropy_varentropy(logprobs_dict, entropy_temp=entropy_temp)

                # Branching condition: low entropy + high varentropy
                # → model is confident overall but split between a few options
                should_branch = (
                    not frozen
                    and H <= entropy_threshold
                    and V > varentropy_threshold
                )

                if should_branch:
                    all_children_tokens = get_branch_tokens(
                        logprobs_dict, sampled_token, branch_top_k,
                    )

                    branch_events += 1
                    total_branches_created += len(all_children_tokens) - 1
                    current_total += len(all_children_tokens) - 1

                    # Freeze: once we have enough hypotheses, stop branching
                    if not frozen and current_total >= max_active_hyps:
                        frozen = True

                    for child_token in all_children_tokens:
                        child_lp = 0.0
                        if child_token in logprobs_dict:
                            child_lp = logprobs_dict[child_token].logprob

                        child = HypNode(
                            tokens=hyp.tokens + [child_token],
                            prompt_len=hyp.prompt_len,
                            logprob=hyp.logprob + child_lp,
                            entropies=hyp.entropies + [H],
                            varentropies=hyp.varentropies + [V],
                        )

                        if child_token in self.stop_token_ids:
                            child.finished = True
                            child.finish_reason = "stop"
                            if answer_extractor:
                                gen_ids = child.tokens[child.prompt_len:]
                                gen_text = self.tokenizer.decode(
                                    gen_ids, skip_special_tokens=True
                                )
                                try:
                                    ans = answer_extractor(gen_text)
                                    if ans:
                                        child.answer = str(ans).strip()
                                except Exception:
                                    pass
                            finished.append(child)
                        else:
                            next_active.append(child)

                else:
                    # No branch: continue with the sampled token
                    child = HypNode(
                        tokens=hyp.tokens + [sampled_token],
                        prompt_len=hyp.prompt_len,
                        logprob=hyp.logprob + token_logprob,
                        entropies=hyp.entropies + [H],
                        varentropies=hyp.varentropies + [V],
                    )

                    if sampled_token in self.stop_token_ids:
                        child.finished = True
                        child.finish_reason = "stop"
                        if answer_extractor:
                            gen_ids = child.tokens[child.prompt_len:]
                            gen_text = self.tokenizer.decode(
                                gen_ids, skip_special_tokens=True
                            )
                            try:
                                ans = answer_extractor(gen_text)
                                if ans:
                                    child.answer = str(ans).strip()
                            except Exception:
                                pass
                        finished.append(child)
                    else:
                        next_active.append(child)

            # Prune: keep only the best hypotheses by cumulative logprob
            if len(next_active) > max_active_hyps:
                next_active.sort(key=lambda h: h.logprob, reverse=True)
                next_active = next_active[:max_active_hyps]

            active = next_active

            if len(finished) >= num_traces:
                break

            # Early stop voting: if enough traces agree on an answer, stop
            if (
                len(finished) >= early_stop_patience
                and answer_extractor is not None
            ):
                answers_so_far = [h.answer for h in finished if h.answer]
                if answers_so_far:
                    cnt = Counter(answers_so_far)
                    top_ans, top_count = cnt.most_common(1)[0]
                    ratio = top_count / len(answers_so_far)
                    if ratio >= early_stop_min_ratio:
                        print(
                            f"  [Early stop] {top_count}/{len(answers_so_far)} "
                            f"traces agree on '{top_ans}' "
                            f"(ratio={ratio:.2f} >= {early_stop_min_ratio})"
                        )
                        break

            # Progress logging
            if (step + 1) % 500 == 0:
                gen_lens = [len(h.tokens) - h.prompt_len for h in active]
                avg_len = sum(gen_lens) / len(gen_lens) if gen_lens else 0
                print(
                    f"  [Phase1 Step {step + 1}] active={len(active)}, "
                    f"finished={len(finished)}, "
                    f"branches={branch_events}, "
                    f"frozen={frozen}, "
                    f"avg_gen_len={avg_len:.0f}"
                )

            # Once frozen, switch to Phase 2 (batch completion)
            if frozen:
                break

        phase1_steps = step + 1 if (active or finished) else 0
        t_phase1_end = time.time()
        print(
            f"  [Phase 1 done] steps={phase1_steps}, "
            f"active={len(active)}, finished={len(finished)}, "
            f"branches={branch_events}, frozen={frozen}, "
            f"time={t_phase1_end - t_phase1:.1f}s"
        )

        # ════════════════════════════════════════════════════════════
        #  Phase 2: Batch completion of remaining hypotheses
        # ════════════════════════════════════════════════════════════
        phase2_batch_size = 0
        phase1_finished_count = len(finished)

        if active and len(finished) < num_traces:
            phase2_batch_size = len(active)
            remaining = max_new_tokens - phase1_steps
            if remaining <= 0:
                # Max tokens exhausted during Phase 1
                for hyp in active:
                    hyp.finished = False
                    hyp.finish_reason = "length"
                    if answer_extractor:
                        gen_ids = hyp.tokens[hyp.prompt_len:]
                        gen_text = self.tokenizer.decode(
                            gen_ids, skip_special_tokens=True
                        )
                        try:
                            ans = answer_extractor(gen_text)
                            if ans:
                                hyp.answer = str(ans).strip()
                        except Exception:
                            pass
                    finished.append(hyp)
            else:
                print(
                    f"  [Phase 2] Batch completing {len(active)} hypotheses, "
                    f"max_tokens={remaining}"
                )
                t_phase2 = time.time()

                batch_params = SamplingParams(
                    temperature=temperature,
                    max_tokens=remaining,
                    logprobs=1,
                    seed=seed,
                    stop_token_ids=list(self.stop_token_ids),
                )

                prompts_batch = [
                    TokensPrompt(prompt_token_ids=hyp.tokens)
                    for hyp in active
                ]
                outputs = self.llm.generate(
                    prompts_batch,
                    sampling_params=batch_params,
                    use_tqdm=False,
                )

                for hyp, output in zip(active, outputs):
                    result = output.outputs[0]

                    hyp.logprob += result.cumulative_logprob
                    hyp.tokens.extend(result.token_ids)
                    hyp.finished = (result.finish_reason == "stop")
                    hyp.finish_reason = result.finish_reason or "length"

                    if answer_extractor:
                        gen_ids = hyp.tokens[hyp.prompt_len:]
                        gen_text = self.tokenizer.decode(
                            gen_ids, skip_special_tokens=True
                        )
                        try:
                            ans = answer_extractor(gen_text)
                            if ans:
                                hyp.answer = str(ans).strip()
                        except Exception:
                            pass

                    finished.append(hyp)

                t_phase2_end = time.time()
                print(
                    f"  [Phase 2 done] completed={len(active)}, "
                    f"time={t_phase2_end - t_phase2:.1f}s"
                )

            active = []

        # Simulate first-to-finish ordering for Phase 2 traces:
        # In Phase 1, traces finish in natural order (first to hit EOS).
        # In Phase 2 (batch), all complete at once — sort by length to
        # approximate which would have finished first.
        if len(finished) > num_traces:
            phase2_finished = finished[phase1_finished_count:]
            phase2_finished.sort(key=lambda h: len(h.tokens))
            needed_from_phase2 = max(0, num_traces - phase1_finished_count)
            finished = (
                finished[:phase1_finished_count]
                + phase2_finished[:needed_from_phase2]
            )

        # Fallback: if nothing finished, return best active by logprob
        if finished:
            output_hyps = finished
        else:
            active.sort(key=lambda h: h.logprob, reverse=True)
            output_hyps = []
            for hyp in active[:num_traces]:
                hyp.finished = False
                hyp.finish_reason = "length"
                if answer_extractor:
                    gen_ids = hyp.tokens[hyp.prompt_len:]
                    gen_text = self.tokenizer.decode(
                        gen_ids, skip_special_tokens=True
                    )
                    try:
                        ans = answer_extractor(gen_text)
                        if ans:
                            hyp.answer = str(ans).strip()
                    except Exception:
                        pass
                output_hyps.append(hyp)

        # ── Format output ──
        traces_out = []
        for hyp in output_hyps:
            gen_ids = hyp.tokens[hyp.prompt_len:]
            gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            traces_out.append({
                "text": gen_text,
                "tokens": gen_ids,
                "logprob": hyp.logprob,
                "finished": hyp.finished,
                "length": len(gen_ids),
                "mean_entropy": (
                    sum(hyp.entropies) / len(hyp.entropies)
                    if hyp.entropies else 0.0
                ),
                "entropy_history": hyp.entropies,
                "varentropy_history": hyp.varentropies,
                "answer": hyp.answer,
            })

        return {
            "traces": traces_out,
            "stats": {
                "num_steps": phase1_steps,
                "num_finished": len([h for h in output_hyps if h.finished]),
                "branch_events": branch_events,
                "total_branches_created": total_branches_created,
                "max_active_batch": max_active_batch,
                "unfinished_left": len(active),
                "num_traces": len(output_hyps),
                "phase1_steps": phase1_steps,
                "phase2_batch_size": phase2_batch_size,
            },
        }
