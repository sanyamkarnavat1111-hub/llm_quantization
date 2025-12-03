'''
Install commands :- 

# The 4 bit quantization needs strictly pytorch version 2.6 or greater and also this install is specifically for CUDA 12 

 uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

 uv pip install --upgrade transformers accelerate bitsandbytes
'''

import os

os.environ['HF_HOME'] = "/mnt/d/VLLMs/HF_CACHE"


from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_id = "microsoft/DialoGPT-medium"  # Or larger like "mistralai/Mistral-7B-v0.1"

# 4-bit config with optimizations
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit
    bnb_4bit_quant_type="nf4",  # NF4 (best for LLMs) or "fp4"
    bnb_4bit_compute_dtype=torch.float16,  # Compute in FP16
    bnb_4bit_use_double_quant=True,  # Nested quant for ~0.4 bits extra savings
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

model.save_pretrained("./4_bit_quantized_model")
tokenizer.save_pretrained("./4_bit_quantized_model")


# Reload (stays quantized)
model = AutoModelForCausalLM.from_pretrained("./4_bit_quantized_model", device_map="auto")

# Test
inputs = tokenizer("What is quantization?", return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))