import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None 
_tokenizer = None 

def load_model_and_tokenizer(prompt: str,):
    # Lazy load the model for a first use
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        print(f"Loading model: {MODEL_NAME} on {_device}")
        # Move model after to CPU or GPU 
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        _model.to(_device)
    return _model, _tokenizer

def generate_answer(prompt:str, max_new_tokens: int = 256):
    model, tokenizer = load_model_and_tokenizer(prompt)

    #Tokenize the prompt to PyTorch Tensors
    inputs = tokenizer(prompt, return_tensors="pt").to(_device)

    start = time.time()
     
    output_ids = model.generate(
        **inputs,
        max_new_tokens = max_new_tokens,
        do_sample = False, # Greedy decoding
    )
    end = time.time()

    # Count only generated tokens not including the prompt
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][prompt_len:]
    num_tokens = generated_ids.shape[0]

    # Token IDs to Python String
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    elapsed = max(end - start, 1e-6) # Avoids division by 0
    tps = num_tokens / elapsed 

    return text, num_tokens, elapsed, tps


