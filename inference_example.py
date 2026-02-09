import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, set_seed
from decoding_tree_sketching.kvbatch_decoder import KVBatchEGDT
from decoding_tree_sketching.utils.eval_utils import extract_answer_llm

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Decoding hyperparameters
DECODE_CONFIG = {
    "entropy_threshold": 2.5,
    "branch_top_k": 3,
    "max_active_hyps": 12,
    "max_new_tokens": 5000,
    "temperature": 0.6,
    "num_traces": 1,
}
tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda" if torch.cuda.is_available() else torch.device("cpu"),
    torch_dtype="auto",
    trust_remote_code=True
)
streamer = TextStreamer(tokenizer)

examples = [
  "Six points $A, B, C, D, E,$ and $F$ lie in a straight line in that order. Suppose that $G$ is a point not on the line and that $AC=26, BD=22, CE=31, DF=33, AF=73, CG=40,$ and $DG=30.$ Find the area of $\\triangle BGE.$",
]
groundtruths = [
    "468",
]
reasoning_tail = r" Please reason step by step, and put your final answer within \boxed{}."
seed = 1

set_seed(seed)
ques_idx = 0
example = examples[ques_idx]
groundtruth = groundtruths[ques_idx]
full_prompt = example + reasoning_tail
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": full_prompt}],
    tokenize=False,
    add_generation_prompt=True
)

# Standard output
inputs = tokenizer(text, return_tensors="pt").to(model.device)
out = model.generate(
    **inputs,
    max_new_tokens=DECODE_CONFIG["max_new_tokens"],
    do_sample=True,
    temperature=DECODE_CONFIG["temperature"],
    streamer=streamer,
)

num_new_tokens = out[0].shape[0] - inputs["input_ids"].shape[1]
stat = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
ans = extract_answer_llm(stat)
print(f"Groundtruth = {groundtruth}, Regular decoding output = {ans}")

kvegdt = KVBatchEGDT(model, tokenizer, seed=seed)
dts_out = kvegdt.generate(
        text,
        entropy_threshold=DECODE_CONFIG["entropy_threshold"],
        branch_top_k=DECODE_CONFIG["branch_top_k"],
        max_active_hyps=DECODE_CONFIG["max_active_hyps"],
        max_new_tokens=DECODE_CONFIG["max_new_tokens"],
        temperature=DECODE_CONFIG["temperature"],
        num_traces=DECODE_CONFIG["num_traces"],
    )

print(f"\n*** GENERATION STATS ***\n{dts_out['stats']}")
dts_ans = extract_answer_llm(dts_out['text'])
print(f"Groundtruth = {groundtruth}, DTS output = {dts_ans}")      
