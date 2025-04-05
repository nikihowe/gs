from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import matplotlib.pyplot as plt

from training_utils import get_minibatch
from dataset_utils import dpoify_dataset


EPOCHS = 1
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
MAX_INPUT = (
    1024  # We empirically verified that all datapoints are <=1023 tokens
)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load the model and tokenizer
checkpoint = 'HuggingFaceTB/SmolLM2-135M-Instruct'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# Prepare the dataset
# TODO: stop truncating the dataset
# TODO: cut examples that are too long
dataset = load_dataset('Unified-Language-Model-Alignment/Anthropic_HH_Golden')
dpoified_train_dataset = dpoify_dataset(dataset['train'])

print("first datapoint", dpoified_train_dataset[0])
print("second datapoint", dpoified_train_dataset[1])

small_train = dpoified_train_dataset.select(range(100))

tokenized_train_prompt = small_train.map(
    lambda x: tokenizer(
        x['prompt'],
        truncation=True,
        padding='max_length',
        max_length=MAX_INPUT,
    ),
    batched=True,
)
tokenized_train_prompt = tokenized_train_prompt.remove_columns(['chosen', 'rejected'])
tokenized_train_chosen = small_train.map(
    lambda x: tokenizer(
        x['chosen'],
        truncation=True,
        padding='max_length',
        max_length=MAX_INPUT,
    ),
    batched=True,
)
tokenized_train_chosen = tokenized_train_chosen.remove_columns(['prompt', 'rejected'])

tokenized_train_rejected = small_train.map(
    lambda x: tokenizer(
        x['rejected'],
        truncation=True,
        padding='max_length',
        max_length=MAX_INPUT,
    ),
    batched=True,
)
tokenized_train_rejected = tokenized_train_rejected.remove_columns(['prompt', 'chosen'])

# Print the first example
print('prompt', tokenized_train_prompt[0])
print('good', tokenized_train_chosen[0])
print('bad', tokenized_train_rejected[0])
assert False

# DPO training loop
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

### COPILOT AUTOCOMPLETED STARTING HERE
for epoch in range(EPOCHS):
    for _, minibatch in enumerate(
        get_minibatch(
            tokenized_train_chosen, tokenized_train_rejected, BATCH_SIZE
        )
    ):
        # Move the batch to the device
        input_ids = minibatch['input_ids'].to(device)
        attention_mask = minibatch['attention_mask'].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f'Epoch {epoch}, Batch {i}, Loss: {loss.item()}')
### COPILOT AUTOCOMPLETED ENDING HERE


# outputs = model.generate(
#     inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True
# )
