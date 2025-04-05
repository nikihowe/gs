from datasets import Dataset
import torch


### GENERATED AND THEN MODIFIED CODE
def get_minibatch(chosen: Dataset, rejected: Dataset, batch_size: int):
    """
    Get a minibatch of data from the chosen and rejected datasets.

    Args:
        chosen (Dataset): The dataset containing the chosen examples.
        rejected (Dataset): The dataset containing the rejected examples.
        batch_size (int): The size of the minibatch.

    Returns:
        dict: A dictionary containing the input IDs and attention masks for the chosen and rejected examples.
    """
    assert len(chosen) == len(rejected), "Chosen and rejected datasets must have the same length."

    # Randomly select indices for the minibatch
    indices = torch.randint(0, len(chosen), (batch_size,))
    
    # Get the chosen examples
    chosen_batch = chosen.select(indices.tolist())
    
    # Get the rejected examples
    rejected_batch = rejected.select(indices.tolist())
    
    return {
        'chosen_input_ids': chosen_batch['input_ids'],
        'chosen_attention_mask': chosen_batch['attention_mask'],
        'rejected_input_ids': rejected_batch['input_ids'],
        'rejected_attention_mask': rejected_batch['attention_mask']
    }
### END GENERATED CODE