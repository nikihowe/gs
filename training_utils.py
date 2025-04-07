from datasets import Dataset
import torch
from constants import TIME_SIZE


### GENERATED AND THEN MODIFIED CODE
def get_minibatch(
    prompt: Dataset,
    chosen: Dataset,
    rejected: Dataset,
    minibatch_size: int,
    device: str,
) -> dict[str, torch.Tensor]:
    """
    Get a minibatch of data from the chosen and rejected datasets.

    Args:
        prompt (Dataset): The dataset containing the prompts.
        chosen (Dataset): The dataset containing the chosen examples.
        rejected (Dataset): The dataset containing the rejected examples.
        minibatch_size (int): The size of the minibatch.

    Returns:
        dict: A dictionary containing the input IDs and
            attention masks for the chosen and rejected examples.
    """
    assert len(prompt) == len(chosen) == len(rejected)

    # Randomly select indices for the minibatch
    indices = torch.randint(0, len(prompt), (minibatch_size,))

    prompt_batch = prompt.select(indices.tolist())
    chosen_batch = chosen.select(indices.tolist())
    rejected_batch = rejected.select(indices.tolist())

    minibatch = {
        'prompt_input_ids': torch.tensor(prompt_batch['input_ids']).to(device),
        'prompt_attention_mask': torch.tensor(
            prompt_batch['attention_mask']
        ).to(device),
        'chosen_input_ids': torch.tensor(chosen_batch['input_ids']).to(device),
        'chosen_attention_mask': torch.tensor(
            chosen_batch['attention_mask']
        ).to(device),
        'rejected_input_ids': torch.tensor(rejected_batch['input_ids']).to(
            device
        ),
        'rejected_attention_mask': torch.tensor(
            rejected_batch['attention_mask']
        ).to(device),
    }
    for v in minibatch.values():
        assert v.shape == (minibatch_size, TIME_SIZE)

    return minibatch


### END GENERATED CODE
