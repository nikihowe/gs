from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from training_utils import get_minibatch
from dataset_utils import dpoify_dataset, get_the_datasets
from constants import TIME_SIZE


"""
DPO. Some inspiration taken from
https://github.com/0xallam/Direct-Preference-Optimization/blob/main/src/train.py
"""


EPOCHS = 8
MINIBATCH_SIZE = 2
BATCH_SIZE = 16
BATCHES_PER_EPOCH = 8
ACCUMULATION_STEPS = BATCH_SIZE // MINIBATCH_SIZE
LEARNING_RATE = 5e-6
BETA = 0.5  # DPO beta parameter


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load the model and tokenizer
# checkpoint = 'HuggingFaceTB/SmolLM2-135M-Instruct'
# checkpoint = 'HuggingFaceTB/SmolLM2-360M-Instruct'
checkpoint = 'HuggingFaceTB/SmolLM2-1.7B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
print('tokenizer has vocab size', tokenizer.vocab_size)
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

train_datasets = get_the_datasets(tokenizer)
(
    tokenized_train_prompt,
    tokenized_train_chosen,
    tokenized_train_rejected,
) = train_datasets

test_datasets = get_the_datasets(tokenizer, test=True)
(
    tokenized_test_prompt,
    tokenized_test_chosen,
    tokenized_test_rejected,
) = test_datasets

# Print the first example
# print('prompt', tokenized_train_prompt[0])
# print('good', tokenized_train_chosen[0])
# print('bad', tokenized_train_rejected[0])


def get_loss(
    minibatch: dict[str, torch.Tensor],
    model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
) -> torch.Tensor:

    prompt_and_chosen_ids_B2T = torch.cat(
        [minibatch['prompt_input_ids'], minibatch['chosen_input_ids']],
        dim=-1,
    )
    prompt_and_chosen_mask_B2T = torch.cat(
        [
            minibatch['prompt_attention_mask'],
            minibatch['chosen_attention_mask'],
        ],
        dim=-1,
    )
    prompt_and_rejected_ids_B2T = torch.cat(
        [minibatch['prompt_input_ids'], minibatch['rejected_input_ids']],
        dim=-1,
    )
    prompt_and_rejected_mask_B2T = torch.cat(
        [
            minibatch['prompt_attention_mask'],
            minibatch['rejected_attention_mask'],
        ],
        dim=-1,
    )

    # Get the mean logprobs for the chosen and rejected completions, for the policy model and reference model
    chosen_outputs_B2TV = model(
        prompt_and_chosen_ids_B2T,
        attention_mask=prompt_and_chosen_mask_B2T,
        return_dict=True,
    )
    chosen_generated_logprobs_BT = extract_mean_logprob(
        chosen_outputs_B2TV, minibatch['chosen_input_ids']
    )

    ref_chosen_outputs_B2TV = ref_model(
        prompt_and_chosen_ids_B2T,
        attention_mask=prompt_and_chosen_mask_B2T,
        return_dict=True,
    )
    ref_chosen_generated_logprobs_BT = extract_mean_logprob(
        ref_chosen_outputs_B2TV, minibatch['chosen_input_ids']
    )

    rejected_outputs_B2TV = model(
        prompt_and_rejected_ids_B2T,
        attention_mask=prompt_and_rejected_mask_B2T,
        return_dict=True,
    )
    rejected_generated_logprobs_BT = extract_mean_logprob(
        rejected_outputs_B2TV, minibatch['rejected_input_ids']
    )

    ref_rejected_outputs_B2TV = ref_model(
        prompt_and_rejected_ids_B2T,
        attention_mask=prompt_and_rejected_mask_B2T,
        return_dict=True,
    )
    ref_rejected_generated_logprobs_BT = extract_mean_logprob(
        ref_rejected_outputs_B2TV, minibatch['rejected_input_ids']
    )

    # Now we can compute the loss
    loss = dpo_loss_function(
        chosen_generated_logprobs_BT,
        ref_chosen_generated_logprobs_BT,
        rejected_generated_logprobs_BT,
        ref_rejected_generated_logprobs_BT,
    )
    return loss


def extract_mean_logprob(
    outputs_B2TV: torch.Tensor, input_ids_BT: torch.Tensor
) -> torch.Tensor:
    assert outputs_B2TV.logits.shape == (
        MINIBATCH_SIZE,
        2 * TIME_SIZE,
        tokenizer.vocab_size,
    )
    generated_probs_BTV = outputs_B2TV.logits[:, TIME_SIZE:, :]
    assert generated_probs_BTV.shape == (
        MINIBATCH_SIZE,
        TIME_SIZE,
        tokenizer.vocab_size,
    )
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


def dpo_loss_function(
    chosen_mean_logprob: torch.Tensor,
    ref_chosen_mean_logprob: torch.Tensor,
    rejected_mean_logprob: torch.Tensor,
    ref_rejected_mean_logprob: torch.Tensor,
) -> torch.Tensor:
    inner = BETA * (
        chosen_mean_logprob
        - ref_chosen_mean_logprob
        - rejected_mean_logprob
        + ref_rejected_mean_logprob
    )
    assert inner.shape == (MINIBATCH_SIZE,)

    outer = -F.logsigmoid(inner).mean()
    assert outer.shape == ()
    return outer


# DPO training loop
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
optimizer.zero_grad()

### COPILOT AUTOCOMPLETED STARTING HERE
acc_steps = 0
done = False
for epoch in range(EPOCHS):
    if done:
        break
    for mb in range(BATCHES_PER_EPOCH * ACCUMULATION_STEPS):
        if done:
            break
        minibatch = get_minibatch(
            tokenized_train_prompt,
            tokenized_train_chosen,
            tokenized_train_rejected,
            MINIBATCH_SIZE,
            device=device,
        )

        # B: minibatch size
        # T: time size
        # V: vocab size
        loss = get_loss(minibatch, model, ref_model) / ACCUMULATION_STEPS
        acc_steps += 1

        # Backward pass
        loss.backward()

        if acc_steps == ACCUMULATION_STEPS:
            optimizer.step()
            optimizer.zero_grad()
            acc_steps = 0

            # Also get the test loss
            test_minibatch = get_minibatch(
                tokenized_test_prompt,
                tokenized_test_chosen,
                tokenized_test_rejected,
                MINIBATCH_SIZE,
                device=device,
            )

            test_loss = get_loss(test_minibatch, model, ref_model)

            print(
                f'Epoch {epoch}, Minibatch {mb}, Train Loss: {loss.item()}, Test Loss: {test_loss.item()}'
            )

            if loss.item() < 0.005 and test_loss.item() < 0.005:
                print('Test loss is low enough, stopping training')
                done = True

# Now, check that the reference model has not changed
ref_ref_model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
for param, ref_param in zip(
    ref_model.parameters(), ref_ref_model.parameters()
):
    assert torch.allclose(param, ref_param)

print('done!')


# Save the trained model
model.save_pretrained('big_dpo_model')

### COPILOT AUTOCOMPLETED ENDING HERE
