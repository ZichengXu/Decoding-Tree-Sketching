from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable  
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import Counter          
import math

try:
    from transformers.cache_utils import DynamicCache
except Exception:
    DynamicCache = None  # If needed, upgrade transformers >= 4.41

@dataclass
class Hyp:
    ids: List[int]
    logprob: float
    finished: bool
    entropies: List[float]
    pkv: Optional[Any]                  
    next_logits: Optional[torch.Tensor]  
    varentropies: List[float] = field(default_factory=list)
    token_metrics: List[Dict[str, float]] = field(default_factory=list)
    branch_steps: List[int] = field(default_factory=list)
    answer: Optional[str] = None

def token_entropy_from_logits(logits: torch.Tensor, temperature: float = 1.0):
    if logits.dim() == 2:
        logits = logits[-1]
    scaled = logits / temperature
    logp = F.log_softmax(scaled, dim=-1)
    p = logp.exp()
    H = -(p * logp).sum().item()
    return H, logp, p

def token_varentropy_from_p(p: torch.Tensor, logp: torch.Tensor, H: float) -> float:
    # I(x_i) = -logp
    # (I(X) - H(X))^2 = (-logp - H)**2s
    vent = (p * (-logp - H)**2).sum().item()
    return vent

def chain_entropy_from_hyp(h: Hyp) -> float:
    if not h.entropies:
        return float("nan")
    return float(sum(h.entropies) / len(h.entropies))

def majority_vote_answers(hyps: List[Hyp]) -> Optional[str]:
    answers = [h.answer for h in hyps if h.answer is not None and h.answer != ""]
    if not answers:
        return None
    cnt = Counter(answers)
    winner, _ = cnt.most_common(1)[0]
    return winner

def get_winner_hyps(hyps: List[Hyp], voting_mode: str = "majority") -> Optional[str]:
    return majority_vote_answers(hyps)


def _tuple_to_dynamic_cache(pkv_tuple):
    if DynamicCache is None:
        raise RuntimeError("DynamicCache unavailable; upgrade transformers.")
    dc = DynamicCache()
    for l, (k, v) in enumerate(pkv_tuple):
        # Expect k,v batch shape [1, ...]
        dc.update(k, v, layer_idx=l)
    return dc

def _to_dynamic_cache(pkv):
    if DynamicCache is not None and isinstance(pkv, DynamicCache):
        return pkv
    to_legacy = getattr(pkv, "to_legacy_cache", None)
    if callable(to_legacy):
        legacy = to_legacy()
        return _tuple_to_dynamic_cache(legacy)
    if isinstance(pkv, tuple):
        return _tuple_to_dynamic_cache(pkv)
    key_cache = getattr(pkv, "key_cache", None)
    value_cache = getattr(pkv, "value_cache", None)
    if key_cache is not None and value_cache is not None:
        legacy_like = tuple((key_cache[l], value_cache[l]) for l in range(len(key_cache)))
        return _tuple_to_dynamic_cache(legacy_like)
    raise TypeError(f"Unsupported cache type for stacking/splitting: {type(pkv)}")

def _stack_cache(caches: List[Any]):
    """
    Stack a list of batch=1 caches into one batched cache (batch=B) via concatenation on dim=0,
    using DynamicCache.update to avoid list-append assumptions.
    """
    if len(caches) == 0:
        raise ValueError("No caches to stack.")
    caches = [_to_dynamic_cache(c) for c in caches]
    base = caches[0]
    if DynamicCache is None:
        raise RuntimeError("DynamicCache unavailable; upgrade transformers.")
    new = DynamicCache()
    num_layers = len(base.key_cache)
    for l in range(num_layers):
        ks = [c.key_cache[l]   for c in caches]  # each [1, ...]
        vs = [c.value_cache[l] for c in caches]
        k = torch.cat(ks, dim=0)                 # [B, ...]
        v = torch.cat(vs, dim=0)
        new.update(k, v, layer_idx=l)
    # If the cache tracks seen length, copy it over (optional)
    if hasattr(base, "seen_tokens"):
        new.seen_tokens = base.seen_tokens
    return new

def _split_cache(batched_cache: Any):
    """
    Split a batched cache (batch=B) into a list of B caches with batch=1 using .update.
    """
    bc = _to_dynamic_cache(batched_cache)
    if DynamicCache is None:
        raise RuntimeError("DynamicCache unavailable; upgrade transformers.")
    B = bc.key_cache[0].shape[0]
    outs = []
    num_layers = len(bc.key_cache)
    for i in range(B):
        c = DynamicCache()
        for l in range(num_layers):
            k_l = bc.key_cache[l][i:i+1]
            v_l = bc.value_cache[l][i:i+1]
            c.update(k_l, v_l, layer_idx=l)
        if hasattr(bc, "seen_tokens"):
            c.seen_tokens = bc.seen_tokens
        outs.append(c)
    return outs

class KVBatchEGDT:
    def __init__(self, model, tokenizer, device: Optional[str] = None, seed: Optional[int] = None):
        self.model = model
        self.tok = tokenizer
        self.device = device or (next(model.parameters()).device if hasattr(model, "parameters") else "cuda")
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        if hasattr(self.model, "generation_config") and hasattr(self.model.generation_config, "use_cache"):
            self.model.generation_config.use_cache = True
        self.model.eval()
        
        self.rng = torch.Generator(device=self.device)
        if seed is not None:
            self.rng.manual_seed(seed)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        entropy_threshold: float = 2.5,
        varentropy_threshold: float = 1.5,
        branch_top_k: int = 3,
        max_new_tokens: int = 256,
        max_active_hyps: int = 48,
        entropy_temp: float = 0.6,
        temperature: float = 1.0,
        stop_strs: Optional[List[str]] = None,
        prefer_greedy_when_confident: bool = False,
        num_traces: int = 1,
        early_stop_min_ratio: float = 0.4,     
        early_stop_patience: int = 4,           
        early_stop_voting_mode: str = "majority",    
        answer_extractor: Optional[Callable[[str], str]] = None,  
    ) -> Dict[str, Any]:
        tok, device = self.tok, self.device

        stop_token_ids_set = set()

        if tok.eos_token_id is not None:
            stop_token_ids_set.add(tok.eos_token_id)

        if hasattr(tok, 'eos_token_ids') and tok.eos_token_ids:
            if isinstance(tok.eos_token_ids, list):
                stop_token_ids_set.update(tok.eos_token_ids)
            elif isinstance(tok.eos_token_ids, int):
                 stop_token_ids_set.add(tok.eos_token_ids)

        phi_stops = self.tok.convert_tokens_to_ids(["<|end|>", "<|endoftext|>"])
        for tid in phi_stops:
            if tid is not None and tid != self.tok.unk_token_id:
                stop_token_ids_set.add(tid)
        
        print(f"[KVBatchEGDT] Using stop token IDs: {stop_token_ids_set}")

        # 0) Prompt
        prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(self.device)
        first_device = prompt_ids.device
        out0 = self.model(input_ids=prompt_ids, use_cache=True)
        pkv0 = _to_dynamic_cache(out0.past_key_values)
        logits0 = out0.logits[:, -1, :]  # [1, vocab]

        active: List[Hyp] = [Hyp(ids=[], logprob=0.0, finished=False, entropies=[], varentropies=[], token_metrics=[], branch_steps=[], pkv=pkv0, next_logits=logits0)]
        finished: List[Hyp] = []

        # Stats
        branch_events = 0
        total_branches_created = 0
        max_active_batch = 1
        frozen = False

        all_token_metrics: List[tuple[float, float]] = []
        early_stop_triggered = False
        early_stop_info: Dict[str, Any] = {}

        def decode_text(ids: List[int], skip_special_tokens=True) -> str:
            return tok.decode(ids, skip_special_tokens=skip_special_tokens)

        def is_stopped_by_str(ids: List[int]) -> bool:
            if not stop_strs:
                return False
            decoded_text = decode_text(ids, skip_special_tokens=False)
            return any(s in decoded_text for s in stop_strs)

        def should_run_early_stop() -> bool:
            return (
                early_stop_min_ratio > 0.0
                and early_stop_patience > 0
                and answer_extractor is not None
            )

        def try_update_early_stop() -> bool:

            nonlocal early_stop_triggered, early_stop_info

            if not should_run_early_stop():
                return False

            answers = [h.answer for h in finished if h.answer is not None and h.answer != ""]
            total = len(answers)
            if total < early_stop_patience:
                return False

            clusters: List[Dict[str, Any]] = []

            for ans in answers:
                placed = False
                if answer_equal_fn is not None:
                    for c in clusters:
                        try:
                            eq = answer_equal_fn(ans, c["repr"])
                        except Exception:
                            eq = (ans == c["repr"])
                        if eq:
                            c["count"] += 1
                            placed = True
                            break
                else:
                    for c in clusters:
                        if ans == c["repr"]:
                            c["count"] += 1
                            placed = True
                            break

                if not placed:
                    clusters.append({"repr": ans, "count": 1})

            if not clusters:
                return False

            winner_cluster = max(clusters, key=lambda c: c["count"])
            winner = winner_cluster["repr"]
            cnt_w = winner_cluster["count"]
            ratio = cnt_w / total if total > 0 else 0.0

            if ratio >= early_stop_min_ratio:
                early_stop_triggered = True
                early_stop_info = {
                    "winner_answer": winner,
                    "winner_ratio": ratio,
                    "num_finished_for_vote": total,
                }
                return True
            return False

        def prune(hyps: List[Hyp], current_max_hyps: int) -> List[Hyp]:
            if len(hyps) <= current_max_hyps:
                return hyps
            scored = sorted(hyps, key=lambda h: (h.logprob - length_penalty * len(h.ids)), reverse=True)
            for h in scored[current_max_hyps:]:
                h.pkv = None
                h.next_logits = None
            return scored[:current_max_hyps]

        steps = 0
        
        for _ in tqdm(range(max_new_tokens), desc="Decoding"):
            steps += 1
            if not active:
                break

            # 1) Plan children from cached logits
            child_tok_ids: List[int] = []
            child_parent_idx: List[int] = []
            child_branch_steps: List[List[int]] = []

            current_total_hyps = len(active)

            for idx, h in enumerate(active):
                if h.next_logits is None:
                    last_inp = torch.tensor([[h.ids[-1]]], device=first_device, dtype=torch.long)
                    out = self.model(input_ids=last_inp, past_key_values=h.pkv, use_cache=True)
                    h.pkv = _to_dynamic_cache(out.past_key_values)
                    h.next_logits = out.logits[:, -1, :]

                logits_i = h.next_logits[0]

                H, logp_T, p_T = token_entropy_from_logits(logits_i, temperature=entropy_temp) # Get H, logp, and p
                vent = token_varentropy_from_p(p_T, logp_T, H)
                h.entropies.append(H)
                h.varentropies.append(vent)
                all_token_metrics.append((H, vent))

                if not frozen and current_total_hyps >= max_active_hyps:
                    frozen = True

                if frozen:
                    if prefer_greedy_when_confident:
                        maxv = torch.max(logp_T)
                        eq = torch.nonzero(logp_T == maxv, as_tuple=False).squeeze(-1)
                        cand_ids = [int(eq.min().item())]
                    else:
                        probs_T = torch.softmax(logits_i / temperature, dim=-1)
                        tid = int(torch.multinomial(probs_T, num_samples=1, generator=self.rng).item())
                        cand_ids = [tid]

                else:
                    is_low_entropy_high_varentropy = (H <= entropy_threshold) and (vent > varentropy_threshold)
                    if is_low_entropy_high_varentropy:
                        vals, inds = torch.topk(logp_T, k=min(branch_top_k, logp_T.shape[-1]))
                        pairs = list(zip(vals.tolist(), inds.tolist()))
                        pairs.sort(key=lambda x: (-x[0], x[1]))  # tie-break
                        cand_ids = [tid for _, tid in pairs]
                        branch_events += 1
                        total_branches_created += (len(cand_ids) - 1)
                        current_total_hyps += (len(cand_ids) - 1)
                    else:
                        if prefer_greedy_when_confident:
                            maxv = torch.max(logp_T)
                            eq = torch.nonzero(logp_T == maxv, as_tuple=False).squeeze(-1)
                            cand_ids = [int(eq.min().item())] # greedy decoding
                        else:
                            probs_T = torch.softmax(logits_i / temperature, dim=-1)
                            tid = int(torch.multinomial(probs_T, num_samples=1, generator=self.rng).item())
                            cand_ids = [tid]
                            

                for tid in cand_ids:
                    child_tok_ids.append(tid)
                    child_parent_idx.append(idx)
                    
            if not child_tok_ids:
                break

            # 2) One batched step for all children
            B = len(child_tok_ids)
            input_ids = torch.tensor(child_tok_ids, device=first_device, dtype=torch.long).unsqueeze(1)
            batched_pkv = _stack_cache([active[idx].pkv for idx in child_parent_idx])

            out = self.model(input_ids=input_ids, past_key_values=batched_pkv, use_cache=True)
            child_logits = out.logits[:, -1, :]               # [B, vocab]
            child_pkv_list = _split_cache(out.past_key_values)
            
            del out
            
            # 3) Materialize children; free parents
            next_active: List[Hyp] = []
            for i in range(B):
                parent = active[child_parent_idx[i]]
                tid = child_tok_ids[i]

                # accumulate logprob from parent's distribution at T=1 
                logits_i = parent.next_logits[0]
                logp_base = torch.log_softmax(logits_i, dim=-1)   # <-- NO temp
                new_lp = parent.logprob + float(logp_base[tid].item())

                new_ids = parent.ids + [tid]
                done = (tid in stop_token_ids_set) or (stop_strs and is_stopped_by_str(new_ids))

                pkv_child = child_pkv_list[i]
                next_logits_child = child_logits[i:i+1, :]
                
                if done:
                    pkv_child = None
                    next_logits_child = None

                current_token_detail = {
                    "token_id": tid,
                    "logprob": float(logp_base[tid].item()),
                    "entropy": parent.entropies[-1],
                    "varentropy": parent.varentropies[-1],
                    "step": steps
                }

                child_token_metrics = parent.token_metrics.copy()
                child_token_metrics.append(current_token_detail)

                child_answer = None
                if done and answer_extractor is not None:
                    decoded = decode_text(new_ids, skip_special_tokens=True)
                    try:
                        ans = answer_extractor(decoded)
                    except Exception:
                        ans = None
                    if ans:
                        child_answer = str(ans).strip()

                child = Hyp(
                    ids=new_ids,
                    logprob=new_lp,
                    finished=done,
                    entropies=parent.entropies.copy(),
                    varentropies=parent.varentropies.copy(),
                    token_metrics=child_token_metrics,
                    pkv=pkv_child,
                    next_logits=next_logits_child,
                    answer=child_answer,   
                )
                if child.finished:
                    finished.append(child)
                    if try_update_early_stop():
                        break 
                else:
                    next_active.append(child)

            if early_stop_triggered:
                for p in active:
                    p.pkv = None
                    p.next_logits = None
                active = next_active
                break


            # Free parents
            for p in active:
                p.pkv = None
                p.next_logits = None

            # 4) Prune
            # active = prune(next_active, current_max_hyps_dynamic)
            active = next_active
            max_active_batch = max(max_active_batch, len(active))
            if not active:
                break

            if len(finished) >= num_traces:
                print(f"[KVBatchEGDT] Reached num_traces={num_traces}, "
                      f"stopping early at step {steps}.")
                break

            if "child_logits" in locals():
                del child_logits
            if "batched_pkv" in locals():
                del batched_pkv
            if "input_ids" in locals():
                del input_ids

            torch.cuda.empty_cache()

        if finished:
            all_hyps = finished
        else:
            all_hyps = sorted(active, key=lambda h: h.logprob, reverse=True)[:num_traces]

        if not all_hyps:
            return {"text": "", "tokens": [], "traces": [], "stats": {"reason": "no_candidates"}}
        
        traces = []
        for h in all_hyps:
            traces.append({
                "text": self.tok.decode(h.ids, skip_special_tokens=True),
                "tokens": h.ids,
                "logprob": h.logprob,
                "finished": h.finished,
                "length": len(h.ids),
                "mean_entropy": (sum(h.entropies) / len(h.entropies)) if h.entropies else 0.0,
                "entropy_history": h.entropies,
                "varentropy_history": h.varentropies,
                "token_metrics": h.token_metrics,
            })

        del prompt_ids, out0, logits0

        if "child_logits" in locals():
            del child_logits
        if "batched_pkv" in locals():
            del batched_pkv
        if "input_ids" in locals():
            del input_ids

        
        torch.cuda.empty_cache()
        stats = {
            "num_steps": steps,
            "num_finished": len(finished),
            "branch_events": branch_events,
            "total_branches_created": total_branches_created,
            "max_active_batch": max_active_batch,
            "unfinished_left": len(active),
            "all_token_metrics": all_token_metrics,
            "num_traces": len(traces),
        }

        return {
            "traces": traces,
            "stats": stats,
        }
