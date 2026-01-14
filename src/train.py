"""
main training program, run using accelerate command
"""
#libraries for hte program to work and save model
import os
import argparse
import logging
from pathlib import Path
from tqdm.auto import tqdm
from torch.optim import AdamW
#torch libraries for training
import torch
from torch.utils.data import DataLoader

from transformers import (
AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForSeq2Seq, AdamWeightDecay,
get_linear_schedule_with_warmup,DataCollatorWithPadding, DataCollatorForLanguageModeling
)

from accelerate import Accelerator
from peft import (prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType)


from dataset import prepare_dataset
from utils import set_seed, save_peft_model
#basic logging using the logging library
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#this function takes arguments it does stuff but not much tbh
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, default="../models/granite4-nano", help="Path to pretrained model")
    p.add_argument("--data_path", type=str, default="../data/dataset.json", help="Path to JSON dataset")
    p.add_argument("--output_dir", type=str, default="../models/finetuned-lora", help="Where to save adapter")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--micro_batch_size", type=int, default=4, help="used if doing grad accumulation")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization via bitsandbytes")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--device_map", type=str, default="auto", help="device_map passed to from_pretrained (auto|cpu| etc)")
    return p.parse_args()

"""
so basicly ts kinda dum and just add it below the tokenizer later in the main function
this is changed by a little bit probably wont work tho 
"""
def collate_fn(tokenizer):

   collator = DataCollatorForSeq2Seq(
    tokenizer,
    padding=True,
    pad_to_multiple_of=8,
    return_tensors="pt",
    label_pad_token_id=-100
   )
   return collator

def main():
    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision= 'fp16')
    device = accelerator.device

    #tokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False, trust_remote_code=True)
    # I dont think I need ts
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit= True,
            bnb_4bit_compute_dtype= torch.float16,
            bnb_4bit_use_double_quant= True,
            bnb_4bit_quant_type="nf4"
        )

    if accelerator.is_main_process:
        logger.info(f"Loading model from: {args.model_dir}")
        print("Accelerator device:", accelerator.device)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        quantization_config= bnb_config,
        device_map= None,#Qlora
        dtype = torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    #saves ram usage
    model.gradient_checkpointing_enable()
    #for qlora
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    #dataset
    data_collator = collate_fn(tokenizer)
    tokenized = prepare_dataset(args.data_path, tokenizer, max_length=args.max_length, add_eos_token=True)

    train_loader = DataLoader(
        tokenized,
        batch_size= args.batch_size,
        shuffle= True,
        collate_fn=data_collator,
        num_workers=min(8, os.cpu_count())
    )
    #LoRA
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        # Targeting specific linear layers is usually safer/better than guessing.
        # Common targets for Llama/Mistral/Granite architectures:
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    #so this does nothing other than telling you how much of the weights are being trained
    if accelerator.is_main_process:
        model.print_trainable_parameters()

    #Optimizer here I use AdamWeight Decay
    optimizer = AdamW(model.parameters(), lr= args.lr, weight_decay=args.weight_decay)

    num_update_steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    max_train_steps = args.epochs * num_update_steps_per_epoch

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * max_train_steps),
        num_training_steps=max_train_steps
    )

    #model prepare with accelerator
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    #training loop
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        # Only show progress bar on main process
        if accelerator.is_main_process:
            progress_bar = tqdm(total=num_update_steps_per_epoch, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for step, batch in enumerate(train_loader):
            # Accelerator handles device placement automatically via prepare()
            # Context manager handles gradient accumulation automatically
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss

                # 1. Track the loss
                # We use .detach() so we don't accidentally keep the whole grad graph
                total_loss += loss.detach().float()

                accelerator.backward(loss)

                # Clip gradients for stability
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Logging
            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process:
                    progress_bar.update(1)
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        if accelerator.is_main_process:
            progress_bar.close()

        # Save checkpoint per epoch
        if accelerator.is_main_process:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch + 1}")
            save_peft_model(model, checkpoint_dir)  # Use your utils function or model.save_pretrained

    # Final Save
    if accelerator.is_main_process:
        logger.info(f"Saving final model to {args.output_dir}")
        save_peft_model(model, args.output_dir)


if __name__ == "__main__":
    main()