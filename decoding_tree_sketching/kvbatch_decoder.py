
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import torch
import torch.nn.functional as F
from tqdm import tqdm

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

def token_entropy_from_logits(logits: torch.Tensor, temperature: float = 1.0):
    if logits.dim() == 2:
        logits = logits[-1]
    scaled = logits / temperature
    logp = F.log_softmax(scaled, dim=-1)
    p = logp.exp()
    H = -(p * logp).sum().item()
    return H, logp

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
        entropy_threshold: float = 3.5,
        branch_top_k: int = 3,
        max_new_tokens: int = 256,
        max_active_hyps: int = 12,
        length_penalty: float = 0.0,
        temperature: float = 1.0,
        eos_token_id: Optional[int] = None,
        stop_strs: Optional[List[str]] = None,
        prefer_greedy_when_confident: bool = True,
    ) -> Dict[str, Any]:
        tok, device = self.tok, self.device
        if eos_token_id is None:
            eos_token_id = tok.eos_token_id

        # 0) Prompt
        prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        out0 = self.model(input_ids=prompt_ids, use_cache=True)
        pkv0 = _to_dynamic_cache(out0.past_key_values)
        logits0 = out0.logits[:, -1, :]  # [1, vocab]

        active: List[Hyp] = [Hyp(ids=[], logprob=0.0, finished=False, entropies=[], pkv=pkv0, next_logits=logits0)]
        finished: List[Hyp] = []

        # Stats
        branch_events = 0
        total_branches_created = 0
        max_active_batch = 1

        def decode_text(ids: List[int]) -> str:
            return tok.decode(ids, skip_special_tokens=True)

        def is_stopped_by_str(ids: List[int]) -> bool:
            if not stop_strs:
                return False
            return any(s in decode_text(ids) for s in stop_strs)

        def prune(hyps: List[Hyp]) -> List[Hyp]:
            if len(hyps) <= max_active_hyps:
                return hyps
            scored = sorted(hyps, key=lambda h: (h.logprob - length_penalty * len(h.ids)), reverse=True)
            for h in scored[max_active_hyps:]:
                h.pkv = None
                h.next_logits = None
            return scored[:max_active_hyps]

        steps = 0
        for _ in tqdm(range(max_new_tokens), desc="Decoding"):
            steps += 1

            # Early stop if a finished shortest is unbeatable
            if finished:
                Lbest = min(len(h.ids) for h in finished)
                if all(len(h.ids) >= Lbest for h in active):
                    break

            # 1) Plan children from cached logits
            child_tok_ids: List[int] = []
            child_parent_idx: List[int] = []
            for idx, h in enumerate(active):
                if h.next_logits is None:
                    last_inp = torch.tensor([[h.ids[-1]]], device=device, dtype=torch.long)
                    out = self.model(input_ids=last_inp, past_key_values=h.pkv, use_cache=True)
                    h.pkv = _to_dynamic_cache(out.past_key_values)
                    h.next_logits = out.logits[:, -1, :]

                logits_i = h.next_logits[0]

                H, logp_T = token_entropy_from_logits(logits_i, temperature=temperature)
                h.entropies.append(H)

                if H > entropy_threshold:
                    vals, inds = torch.topk(logp_T, k=min(branch_top_k, logp_T.shape[-1]))
                    pairs = list(zip(vals.tolist(), inds.tolist()))
                    pairs.sort(key=lambda x: (-x[0], x[1]))  # tie-break
                    cand_ids = [tid for _, tid in pairs]
                    branch_events += 1
                    total_branches_created += (len(cand_ids) - 1)
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
            input_ids = torch.tensor(child_tok_ids, device=device, dtype=torch.long).unsqueeze(1)  # [B,1]
            batched_pkv = _stack_cache([active[idx].pkv for idx in child_parent_idx])

            out = self.model(input_ids=input_ids, past_key_values=batched_pkv, use_cache=True)
            child_logits = out.logits[:, -1, :]               # [B, vocab]
            child_pkv_list = _split_cache(out.past_key_values)

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
                done = (eos_token_id is not None and tid == eos_token_id) or (stop_strs and is_stopped_by_str(new_ids))

                child = Hyp(
                    ids=new_ids,
                    logprob=new_lp,
                    finished=done,
                    entropies=parent.entropies.copy(),
                    pkv=child_pkv_list[i],
                    next_logits=child_logits[i:i+1, :],
                )
                if child.finished:
                    finished.append(child)
                else:
                    next_active.append(child)

            # Free parents
            for p in active:
                p.pkv = None
                p.next_logits = None

            # 4) Prune
            active = prune(next_active)
            max_active_batch = max(max_active_batch, len(active))
            if not active:
                break

        candidates = finished if finished else active
        if not candidates:
            return {"text": "", "tokens": [], "stats": {"reason": "no_candidates"}}
        best = sorted(candidates, key=lambda h: (len(h.ids), -h.logprob))[0]

        return {
            "text": self.tok.decode(best.ids, skip_special_tokens=True),
            "tokens": best.ids,
            "stats": {
                "num_steps": steps,
                "num_finished": len(finished),
                "generated_len": len(best.ids),
                "mean_entropy_best": (sum(best.entropies)/len(best.entropies)) if best.entropies else 0.0,
                "branch_events": branch_events,
                "total_branches_created": total_branches_created,
                "max_active_batch": max_active_batch,
                "unfinished_left": len(active),
            }
        }
