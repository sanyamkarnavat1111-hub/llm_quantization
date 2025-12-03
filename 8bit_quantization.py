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

# Model ID from Hugging Face Hub (use a small one first for testing)
model_id = "gpt2"  # Or "meta-llama/Llama-2-7b-hf" (requires HF token for gated models)

# Optional: Configure quantization (defaults work fine)
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Enable 8-bit
    llm_int8_threshold=6.0,  # Skip quant if abs(mean) > threshold (default)
    llm_int8_has_fp16_weight=False,  # Use FP16 weights if True
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",  # Auto-distribute across GPUs (or "cuda:0")
    trust_remote_code=True,  # If model requires custom code
    torch_dtype=torch.float16,  # Use FP16 for activations
)


model.save_pretrained("./8_bit_quantized_model")
tokenizer.save_pretrained("./8_bit_quantized_model")

# Reload (stays quantized)
model = AutoModelForCausalLM.from_pretrained("./8_bit_quantized_model", device_map="auto")

# Test inference
inputs = tokenizer("What is model quantization in large language models ?", return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print(model.get_memory_footprint())