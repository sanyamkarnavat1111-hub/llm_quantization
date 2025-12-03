
'''
# What is Nested Quantization?
    - Nested quantization (also called double quantization or quantization of quantization parameters) is an advanced post-training quantization technique used to further compress the weights of large language models (LLMs) while minimizing accuracy loss. It's particularly popular in libraries like BitsAndBytes for 4-bit quantization schemes (e.g., NF4 or FP4).

# Key Concepts:

- Standard 4-bit Quantization: Weights are quantized to 4 bits per parameter, but this requires storing additional "quantization parameters" (like scales and zero-points) in higher precision (e.g., FP16). These extras can add 0.5-1 bit per parameter overhead, reducing overall savings.
- Nested Quantization: Instead of storing those scales/zero-points in full precision, they are themselves quantized—typically to 8 bits. This "nests" a second quantization layer, saving an additional ~0.4 bits per parameter (total effective bits: ~3.6 instead of 4).

# Benefits:
    Memory Savings: Up to 10-15% more efficient than plain 4-bit, crucial for fitting massive models (e.g., 70B params) on consumer GPUs (8-24 GB VRAM).
    Speed: Minimal overhead; computations remain fast on CUDA.
    Accuracy: Negligible degradation (often <0.1 perplexity increase) due to careful scaling.

# When to Use: Ideal for inference on resource-constrained setups. It's enabled by default in BitsAndBytes' 4-bit config but can be toggled.


# Limitations: Only for supported dtypes (NF4/FP4); not for 8-bit. Requires CUDA 11+ (your 12.6 setup is fine).

# Additionally we can 
'''
import os

os.environ['HF_HOME'] = "/mnt/d/VLLMs/HF_CACHE"


from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from dotenv import load_dotenv


load_dotenv() # The HF_TOKEN is set directly in .env file

# Model ID (use a small one for testing; requires HF token for gated like Llama)
model_id = "meta-llama/Llama-3.1-8B-Instruct"  # Or "meta-llama/Llama-2-7b-hf"

# BitsAndBytesConfig for 4-bit with NESTED (double) quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit base
    bnb_4bit_quant_type="nf4",  # NF4: NormalFloat4 (recommended for LLMs)
    bnb_4bit_compute_dtype=torch.float16,  # FP16 for computations
    bnb_4bit_use_double_quant=True,  # KEY: Enable nested (double) quantization
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",  # Auto to GPU
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

# Print memory footprint (in bytes; divide by 1e9 for GB)
footprint = model.get_memory_footprint()
print(f"Quantized Footprint: {footprint} bytes (~{footprint / 1e9:.2f} GB)")



model.save_pretrained("./double_quantized_model")
tokenizer.save_pretrained("./double_quantized_model")



model = AutoModelForCausalLM.from_pretrained("./double_quantized_model", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("./double_quantized_model" , device_map="auto")



prompt = "Explain nested quantization in one sentence."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True
    )
print(tokenizer.decode(outputs[0], skip_special_tokens=True))