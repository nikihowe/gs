from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch


# Load the model and tokenizer
checkpoint = 'HuggingFaceTB/SmolLM2-135M-Instruct'

device = 'cuda'   # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

messages = [{'role': 'user', 'content': 'What is gravity?'}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False)
print(input_text)
inputs = tokenizer.encode(input_text, return_tensors='pt').to(device)
outputs = model.generate(
    inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True
)
print(tokenizer.decode(outputs[0]))

# Prepare the dataset
dataset = load_dataset("Unified-Language-Model-Alignment/Anthropic_HH_Golden")

# Print the first example
print(dataset['train'][0])

# Write a DPO training loop
