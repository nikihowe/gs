from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from training_utils import get_minibatch
from dataset_utils import dpoify_dataset


"""
DPO. Some inspiration taken from
https://github.com/0xallam/Direct-Preference-Optimization/blob/main/src/train.py
"""


EPOCHS = 10
MINIBATCH_SIZE = 4  # TODO: consider raising or adding accumulation
LEARNING_RATE = 1e-5
TIME_SIZE = (
    1024  # We empirically verified that all datapoints are <=1023 tokens
)
BETA = 0.5  # DPO beta parameter


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load the model and tokenizer
checkpoint = 'HuggingFaceTB/SmolLM2-135M-Instruct'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
print("tokenizer has vocab size", tokenizer.vocab_size)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# Make and freeze a reference model
ref_model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False

# Prepare the dataset
# TODO: stop truncating the dataset
# TODO: cut examples that are too long

# Try loading the datasets from disk
try:
    tokenized_train_prompt = load_from_disk('tokenized_train_prompt')
    tokenized_train_chosen = load_from_disk('tokenized_train_chosen')
    tokenized_train_rejected = load_from_disk('tokenized_train_rejected')
    print('Loaded datasets from disk')
except:
    print(
        'Unable to load datasets from disk, so getting originals and tokenizing them'
    )
    dataset = load_dataset(
        'Unified-Language-Model-Alignment/Anthropic_HH_Golden'
    )
    dpoified_train_dataset = dpoify_dataset(dataset['train'])

    print('first datapoint', dpoified_train_dataset[0])
    print('second datapoint', dpoified_train_dataset[1])

    small_train = dpoified_train_dataset.select(range(100))

    tokenized_train_prompt = small_train.map(
        lambda x: tokenizer(
            x['prompt'],
            truncation=True,
            padding='max_length',
            max_length=TIME_SIZE,
        ),
        batched=True,
    )
    tokenized_train_prompt = tokenized_train_prompt.remove_columns(
        ['chosen', 'rejected']
    )
    tokenized_train_chosen = small_train.map(
        lambda x: tokenizer(
            x['chosen'],
            truncation=True,
            padding='max_length',
            max_length=TIME_SIZE,
        ),
        batched=True,
    )
    tokenized_train_chosen = tokenized_train_chosen.remove_columns(
        ['prompt', 'rejected']
    )

    tokenized_train_rejected = small_train.map(
        lambda x: tokenizer(
            x['rejected'],
            truncation=True,
            padding='max_length',
            max_length=TIME_SIZE,
        ),
        batched=True,
    )
    tokenized_train_rejected = tokenized_train_rejected.remove_columns(
        ['prompt', 'chosen']
    )

    # Save the tokenized datasets so we don't have to re-tokenize them every time
    tokenized_train_prompt.save_to_disk('tokenized_train_prompt')
    tokenized_train_chosen.save_to_disk('tokenized_train_chosen')
    tokenized_train_rejected.save_to_disk('tokenized_train_rejected')

# Print the first example
# print('prompt', tokenized_train_prompt[0])
# print('good', tokenized_train_chosen[0])
# print('bad', tokenized_train_rejected[0])

def extract_mean_logprob(outputs_B2TV: torch.Tensor, input_ids_BT: torch.Tensor) -> torch.Tensor:
    assert outputs_B2TV.logits.shape == (MINIBATCH_SIZE, 2 * TIME_SIZE, tokenizer.vocab_size)
    generated_probs_BTV = outputs_B2TV.logits[:, TIME_SIZE:, :]
    assert generated_probs_BTV.shape == (MINIBATCH_SIZE, TIME_SIZE, tokenizer.vocab_size)
    generated_logprobs_BTV = torch.log_softmax(generated_probs_BTV, dim=-1)
    # index into the chosen generated logprobs with the chosen_input_ids
    generated_logprobs_BT = generated_logprobs_BTV.gather(
        2, input_ids_BT.unsqueeze(-1)
    ).squeeze(-1)
    assert generated_logprobs_BT.shape == (MINIBATCH_SIZE, TIME_SIZE)
    # mean over time dimension
    generated_logprobs_B = generated_logprobs_BT.mean(dim=-1)
    assert generated_logprobs_B.shape == (MINIBATCH_SIZE,)
    return generated_logprobs_B

def get_dpo_loss(
    chosen_mean_logprob: torch.Tensor,
    ref_chosen_mean_logprob: torch.Tensor,
    rejected_mean_logprob: torch.Tensor,
    ref_rejected_mean_logprob: torch.Tensor,
) -> torch.Tensor:
    inner = BETA * (
        chosen_mean_logprob - ref_chosen_mean_logprob
        - rejected_mean_logprob + ref_rejected_mean_logprob
    )
    assert inner.shape == (MINIBATCH_SIZE,)

    outer = -F.logsigmoid(inner).mean()
    assert outer.shape == ()
    return outer
    

# DPO training loop
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

### COPILOT AUTOCOMPLETED STARTING HERE
for epoch in range(EPOCHS):
    minibatch = get_minibatch(
        tokenized_train_prompt, 
        tokenized_train_chosen, 
        tokenized_train_rejected, 
        MINIBATCH_SIZE,
    )
    minibatch = {k: v.to(device) for k, v in minibatch.items()}
    # B: minibatch size
    # T: time size
    # V: vocab size
    for _, v in minibatch.items():
        assert v.shape == (MINIBATCH_SIZE, TIME_SIZE)

    prompt_and_chosen_ids_B2T = torch.cat(
        [minibatch['prompt_input_ids'], minibatch['chosen_input_ids']],
        dim=-1,
    )
    prompt_and_chosen_mask_B2T = torch.cat(
        [minibatch['prompt_attention_mask'], minibatch['chosen_attention_mask']],
        dim=-1,
    )
    prompt_and_rejected_ids_B2T = torch.cat(
        [minibatch['prompt_input_ids'], minibatch['rejected_input_ids']],
        dim=-1,
    )
    prompt_and_rejected_mask_B2T = torch.cat(
        [minibatch['prompt_attention_mask'], minibatch['rejected_attention_mask']],
        dim=-1,
    )

    # Get the mean logprobs for the chosen and rejected completions, for the policy model and reference model
    chosen_outputs_B2TV = model(prompt_and_chosen_ids_B2T, attention_mask=prompt_and_chosen_mask_B2T, return_dict=True)
    chosen_generated_logprobs_BT = extract_mean_logprob(chosen_outputs_B2TV, minibatch['chosen_input_ids'])

    ref_chosen_outputs_B2TV = ref_model(prompt_and_chosen_ids_B2T, attention_mask=prompt_and_chosen_mask_B2T, return_dict=True)
    ref_chosen_generated_logprobs_BT = extract_mean_logprob(ref_chosen_outputs_B2TV, minibatch['chosen_input_ids'])

    rejected_outputs_B2TV = model(prompt_and_rejected_ids_B2T, attention_mask=prompt_and_rejected_mask_B2T, return_dict=True)
    rejected_generated_logprobs_BT = extract_mean_logprob(rejected_outputs_B2TV, minibatch['rejected_input_ids'])

    ref_rejected_outputs_B2TV = ref_model(prompt_and_rejected_ids_B2T, attention_mask=prompt_and_rejected_mask_B2T, return_dict=True)
    ref_rejected_generated_logprobs_BT = extract_mean_logprob(ref_rejected_outputs_B2TV, minibatch['rejected_input_ids'])

    # Now we can compute the loss
    loss = get_dpo_loss(
        chosen_generated_logprobs_BT,
        ref_chosen_generated_logprobs_BT,
        rejected_generated_logprobs_BT,
        ref_rejected_generated_logprobs_BT,
    )

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if _ % 10 == 0:
        print(f'Epoch {epoch}, Batch {_}, Loss: {loss.item()}')



### COPILOT AUTOCOMPLETED ENDING HERE


# outputs = model.generate(
#     inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True
# )
