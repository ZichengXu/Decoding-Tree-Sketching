import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from decoding_tree_sketching.kvbatch_decoder import KVBatchEGDT

examples = [
    "Find the sum of all integer bases $b>9$ for which $17_b$ is a divisor of $97_b.$",
]
reasoning_tail = r" Please reason step by step, and put your final answer within \boxed{}."

# Replace with your model checkpoint if needed
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# TODO: Replace with your Hugging Face access token
access_token = "YOUR_HF_ACCESS_TOKEN"
# TODO: Set a directory for model/tokenizer cache
cache_dir = "./YOUR_CACHE_DIR"
seed = 0

# Decoding hyperparameters
DECODE_CONFIG = {
    "entropy_threshold": 2.5,
    "branch_top_k": 3,
    "max_active_hyps": 12,
    "max_new_tokens": 4096,
    "temperature": 0.6,
}

def main():
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=access_token,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda",
        torch_dtype="auto",
        token=access_token,
        cache_dir=cache_dir,
        trust_remote_code=True
    )

    kvegdt = KVBatchEGDT(model, tokenizer, seed=seed)

    for example in examples:
        full_prompt = example + reasoning_tail
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": full_prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
        out = kvegdt.generate(
            text,
            entropy_threshold=DECODE_CONFIG["entropy_threshold"],
            branch_top_k=DECODE_CONFIG["branch_top_k"],
            max_active_hyps=DECODE_CONFIG["max_active_hyps"],
            max_new_tokens=DECODE_CONFIG["max_new_tokens"],
            temperature=DECODE_CONFIG["temperature"],
        )
        
        print(f"*** MODEL OUTPUT ***\n{out['text']}")
        # Print generation statistics such as steps, branch events, and sequence length
        print(f"\n*** GENERATION STATS ***\n{out['stats']}")

if __name__ == "__main__":
    main()       
