from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Define your paths
base_path = "$PATH TO BASE MODEL$
adapter_path = "PATH TO LORA ADAPTERS"
save_path = "PATH TO MERGED MODEL"

# 1. Load the tokenizer from the base path (CRITICAL)
tokenizer = AutoTokenizer.from_pretrained(base_path)

# 2. Load the base model in FP16 (Required for merging QLoRA/4-bit adapters)
# Use device_map="cpu" to avoid OOM if you aren't using the GPU for inference right now
base_model = AutoModelForCausalLM.from_pretrained(
    base_path,
    dtype=torch.float16,
    device_map="cpu"
)

# 3. Load the adapter
adapter_model = PeftModel.from_pretrained(base_model, adapter_path)

# 4. Merge
print("Merging layers... this may take a minute.")
merged_model = adapter_model.merge_and_unload()

# 5. Save everything to the same folder (The "Standalone" part)
merged_model.save_pretrained(save_path, safe_serialization=True)
tokenizer.save_pretrained(save_path)

print(f"Success! stand-alone model saved to: {save_path}")
