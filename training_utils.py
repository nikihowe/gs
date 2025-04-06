from datasets import Dataset
import torch


### GENERATED AND THEN MODIFIED CODE
def get_minibatch(
    prompt: Dataset, chosen: Dataset, rejected: Dataset, batch_size: int
) -> dict[str, torch.Tensor]:
    """
    Get a minibatch of data from the chosen and rejected datasets.

    Args:
        prompt (Dataset): The dataset containing the prompts.
        chosen (Dataset): The dataset containing the chosen examples.
        rejected (Dataset): The dataset containing the rejected examples.
        batch_size (int): The size of the minibatch.

    Returns:
        dict: A dictionary containing the input IDs and
            attention masks for the chosen and rejected examples.
    """
    assert len(prompt) == len(chosen) == len(rejected)

    # Randomly select indices for the minibatch
    indices = torch.randint(0, len(prompt), (batch_size,))

    prompt_batch = prompt.select(indices.tolist())
    chosen_batch = chosen.select(indices.tolist())
    rejected_batch = rejected.select(indices.tolist())

    return {
        'prompt_input_ids': torch.tensor(prompt_batch['input_ids']),
        'prompt_attention_mask': torch.tensor(prompt_batch['attention_mask']),
        'chosen_input_ids': torch.tensor(chosen_batch['input_ids']),
        'chosen_attention_mask': torch.tensor(chosen_batch['attention_mask']),
        'rejected_input_ids': torch.tensor(rejected_batch['input_ids']),
        'rejected_attention_mask': torch.tensor(rejected_batch['attention_mask']),
    }


### END GENERATED CODE
