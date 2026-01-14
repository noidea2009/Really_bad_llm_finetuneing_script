# Really_bad_llm_finetuneing_script
basically what I made in like 3 weeks, its really not that good and is pretty slow, with a good dataset and model it could do fine tho 

# IN YOUR PROJECT MANAGER JUST ADD A MODELS FOLDER AND PUT THEM IN THERE MICROSOFT IS MONEY HUNGRY AND I CANT UPLOAD MORE THAN 25MB 
yeah gl

# Configuration & Command-Line Arguments

This script uses command-line arguments to configure model fine-tuning, LoRA settings, and training behavior. Below is a summary of the available options and their defaults.

Basic Paths

--model_dir
Path to the pretrained base model.
Default: ../models/granite4-nano

--data_path
Path to the training dataset (JSON format).
Default: ../data/dataset.json

--output_dir
Directory where the fine-tuned LoRA adapter will be saved.
Default: ../models/finetuned-lora

Training Settings

--batch_size
Number of samples per training step.
Default: 4

--micro_batch_size
Batch size per forward pass when using gradient accumulation.
Default: 4

--epochs
Number of training epochs.
Default: 3

--lr
Learning rate for the optimizer.
Default: 2e-5

--weight_decay
Weight decay (L2 regularization).
Default: 0.0

--gradient_accumulation_steps
Number of steps to accumulate gradients before updating weights.
Default: 1

--seed
Random seed for reproducibility.
Default: 42

Model & Tokenization

--max_length
Maximum token length for input sequences.
Default: 1024

--device_map
Device placement strategy (auto, cpu, cuda, etc.).
Default: auto

--use_4bit
Enable 4-bit quantization using bitsandbytes to reduce GPU memory usage.
Default: Disabled

LoRA Configuration

--lora_r
Rank of the LoRA adapter matrices.
Default: 8

--lora_alpha
Scaling factor applied to LoRA updates.
Default: 16

--lora_dropout
Dropout rate for LoRA layers during training.
Default: 0.05

Example Usage
accelerate launch train.py \
  --model_dir ../models/granite4-nano \
  --data_path ../data/dataset.json \
  --output_dir ../models/finetuned-lora \
  --epochs 5 \
  --use_4bit
