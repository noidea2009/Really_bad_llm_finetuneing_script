# src/dataset.py
from typing import List, Dict, Optional
import json
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

def load_json_dataset(path: str) -> List[Dict]:
    """
    Loads a JSON array file into a list of dicts.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # allow {"data": [...]} or single-dict
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        else:
            return [data]
    return data


def _build_prompt(example: Dict) -> Dict:
    # Case 1: instruction / input / output (like Alpaca)
    if "instruction" in example:
        instr = example.get("instruction", "").strip()
        inp = example.get("input", "").strip()
        out = example.get("output", "").strip() or example.get("output_text", "").strip()
        if inp:
            prompt = f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instr}\n\n### Response:\n"
        return {"prompt": prompt, "response": out}

    # Case 2: prompt/response
    if "prompt" in example and "response" in example:
        # Modified to include a boundary header so the masking is effective
        prompt = example["prompt"].strip()
        if "### Response:" not in prompt:
            prompt = f"{prompt}\n\n### Response:\n"
        return {"prompt": prompt, "response": example["response"].strip()}

    # Case 3: text containing both prompt/response separated by delimiter
    if "text" in example:
        text = example["text"].strip()
        for sep in ["\n\n### Response:\n", "\n\n### Response:", "\n\nResponse:", "\n\n---\n\n"]:
            if sep in text:
                parts = text.split(sep, 1)
                return {"prompt": parts[0].strip() + sep, "response": parts[1].strip()}
        return {"prompt": "", "response": text}

    # fallback
    all_values = " ".join(str(v) for v in example.values())
    return {"prompt": "", "response": all_values.strip()}


def prepare_dataset(
        json_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 1024,
        add_eos_token: bool = True,
):
    raw = load_json_dataset(json_path)
    pairs = [_build_prompt(x) for x in raw]

    texts = []
    for p in pairs:
        if isinstance(p, dict):
            prompt = str(p.get("prompt", ""))
            response = str(p.get("response", ""))
        else:
            prompt = ""
            response = str(p)

        full = prompt + response
        if add_eos_token and tokenizer.eos_token:
            if not full.endswith(tokenizer.eos_token):
                full = full + tokenizer.eos_token

        texts.append({"prompt": prompt, "response": response, "text": full})

    ds = Dataset.from_list(texts)

    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True,
        )

        all_labels = []
        for i in range(len(tokenized["input_ids"])):
            input_ids = list(tokenized["input_ids"][i])
            prompt_text = examples["prompt"][i]

            prompt_tokenized = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True
            )
            prompt_len = len(prompt_tokenized["input_ids"])

            # Handle tokenizers that append EOS to every string
            if prompt_len > 0 and prompt_tokenized["input_ids"][-1] == tokenizer.eos_token_id:
                prompt_len -= 1

            mask_limit = min(prompt_len, len(input_ids))
            labels = [-100] * mask_limit + input_ids[mask_limit:]
            all_labels.append(labels)

        tokenized["labels"] = all_labels
        return tokenized

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)

    # --- TESTING COMPONENT START ---
    print("\n" + "=" * 50)
    print("DATASET MASKING VERIFICATION")
    print("=" * 50)
    # Check the first sample
    test_idx = 0
    test_ids = tokenized[test_idx]["input_ids"]
    test_labels = tokenized[test_idx]["labels"]

    # Reconstruct what the model sees (Labels with -100 hidden)
    visual_labels = []
    for i, (t_id, l_id) in enumerate(zip(test_ids, test_labels)):
        if l_id == -100:
            visual_labels.append("[MASK]")
        else:
            visual_labels.append(tokenizer.decode([t_id]))

    print(f"Sample 0 (First 50 tokens):\n{''.join(visual_labels[:50])}...")
    print("\nVerification: You should see [MASK] for instructions and headers,")
    print("and actual text for the response. If so, your code is ECHO-PROOF.")
    print("=" * 50 + "\n")
    # --- TESTING COMPONENT END ---

    tokenized.set_format(type="torch")
    return tokenized