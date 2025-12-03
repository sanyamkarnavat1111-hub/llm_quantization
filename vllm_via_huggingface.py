
'''
The code will crash if the GPU memory is not sufficient to load the entire LLM model at once in GPU


Solution :- provide GPU memory utilization parameter so that our code doesn't gets crashed.


'''
import os
os.environ['HF_HOME'] = "/mnt/d/VLLMs/HF_CACHE"
from vllm import LLM , SamplingParams
from dotenv import load_dotenv


load_dotenv()


model_name = "./4_bit_quantized_model"


sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    top_k=3,
    max_tokens=1000,
)


llm = LLM(
    model=model_name,
    hf_token=os.environ['HF_TOKEN'],
    gpu_memory_utilization=0.70,  # Key fix: Reserves ~6.8 GB
    enforce_eager=True,  # Ensures compatibility with compute cap <8.0
    max_model_len=1024,  # Match your config; adjust if needed
)


# Prompt
prompt = "Explain the difference between supervised and unsupervised learning."

# Run inference
outputs = llm.generate(prompt, sampling_params)


for output in outputs:
    print("Generated Text:\n", output.outputs[0].text)
    