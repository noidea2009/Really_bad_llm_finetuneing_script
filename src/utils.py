import random
import os
import torch
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_peft_model(peft_model, save_directory: str):
    os.makedirs(save_directory, exist_ok=True)
    peft_model.save_pretrained(save_directory)

