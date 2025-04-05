from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import matplotlib.pyplot as plt


EPOCHS = 1
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
MAX_INPUT = 1024  # We empirically verified that all datapoints are <=1023 tokens


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the model and tokenizer
checkpoint = 'HuggingFaceTB/SmolLM2-135M-Instruct'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# Prepare the dataset
# TODO: stop truncating the dataset
dataset = load_dataset('Unified-Language-Model-Alignment/Anthropic_HH_Golden')

# TODO: remove datapoints that are too long
# The majority of datapoints are <512 tokens, so to speed things up
# we're just going to remove ones that are longer than that
# tokenized_train_chosen = dataset.map(
#     lambda x: tokenizer(x['chosen']),
#     batched=True,
# )
# tokenized_train_rejected = dataset.map(
#     lambda x: tokenizer(x['rejected']),
#     batched=True,
# )

# print(f"before filter, there are {len(tokenized_train_chosen)} chosen examples")
# print(f"before filter, there are {len(tokenized_train_rejected)} rejected examples")

# dataset = tokenized_train_chosen.filter(lambda x: len(x['chosen']['input_ids']) < MAX_INPUT)
# dataset = dataset.filter(lambda x: len(x['rejected']) < MAX_INPUT)

# print("after filter, there are", len(dataset['train']), "examples")
# print("of which", len(dataset['train']['chosen']), "were chosen")
# print("and", len(dataset['train']['rejected']), "were rejected")a

small_train = dataset['train'].select(range(100))
small_test = dataset['test'].select(range(100))

tokenized_train_chosen = small_train.map(
    lambda x: tokenizer(x['chosen'], truncation=True, padding='max_length', max_length=MAX_INPUT),
    batched=True,
)
tokenized_train_chosen = tokenized_train_chosen.remove_columns(['rejected'])

tokenized_train_rejected = small_train.map(
    lambda x: tokenizer(x['rejected'], truncation=True, padding='max_length', max_length=MAX_INPUT),
    batched=True,
)
tokenized_train_rejected = tokenized_train_rejected.remove_columns(['chosen'])

# Print the first example
eg_datapoint = tokenized_train_chosen[0]
print("good", eg_datapoint)

# Plot the distribution of input lengths
# train_lengths = [len(x['input_ids']) for x in tokenized_train_chosen]
# test_lengths = [len(x['input_ids']) for x in tokenized_train_rejected]

# print("max train length", max(train_lengths))
# print("max test length", max(test_lengths))

# plt.hist(train_lengths, bins=50, alpha=0.5, label='Train')
# plt.hist(test_lengths, bins=50, alpha=0.5, label='Test')
# plt.xlabel('Input Length')
# plt.ylabel('Frequency')
# plt.title('Distribution of Input Lengths')
# plt.legend()
# plt.savefig('input_lengths.png')
# What's the longest input?

# Check the rejected examples
eg_datapoint = tokenized_train_rejected[0]
print("bad", eg_datapoint)

# DPO training loop
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

### COPILOT AUTOCOMPLETED STARTING HERE
for epoch in range(EPOCHS):
    for i, batch in enumerate(tokenized_train_chosen):
        # Move the batch to the device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")
### COPILOT AUTOCOMPLETED ENDING HERE


# outputs = model.generate(
#     inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True
# )