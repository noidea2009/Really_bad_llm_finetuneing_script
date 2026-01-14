from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Paths
merged_model_path = "/home/noidea/PycharmProjects/PythonProject/models/granite_4.0-test01_merged"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(merged_model_path, use_fast=False)

# Load model
model = AutoModelForCausalLM.from_pretrained(merged_model_path)

# Move model to device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()  # set to evaluation mode
input_text = input()
inputs = tokenizer(input_text, return_tensors="pt").to(device)

outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    do_sample=True,
    top_p=0.9,
    temperature=0.7,
    repetition_penalty=1.1,
    eos_token_id=tokenizer.eos_token_id
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
