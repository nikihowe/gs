# dataset_utils.py (or dataset_utils2.py)

import os
from datasets import Dataset, load_dataset, load_from_disk
from transformers import PreTrainedTokenizer

# Helper function to extract prompt and response from HH format
def extract_prompt_and_response(example: str) -> tuple[str, str]:
    """Extracts prompt and response assuming HH format."""
    match_string = '\n\nAssistant: '
    assistant_start_idx = example.rfind(match_string)
    if assistant_start_idx == -1 or assistant_start_idx == 0:
        print(f"Warning: Could not find '{match_string}' properly in example: {example[:100]}...")
        if assistant_start_idx == -1: return example, "" # Treat as prompt? Or handle differently?
        return match_string, example[len(match_string):] # No prompt, just response
    return (
        example[: assistant_start_idx + len(match_string)],
        example[assistant_start_idx + len(match_string) :],
    )

# Helper function to convert HH dataset to DPO format (text columns)
def dpoify_dataset(dataset: Dataset) -> Dataset:
    """Converts raw HH dataset format to DPO format with text columns."""
    new_dataset = {'prompt': [], 'chosen': [], 'rejected': []}
    num_skipped = 0
    num_errors = 0
    for example in dataset:
        try:
            raw_chosen = example.get('chosen')
            raw_rejected = example.get('rejected')
            if not raw_chosen or not raw_rejected:
                # print("Warning: Missing 'chosen' or 'rejected' key in raw data.")
                num_errors += 1
                continue

            prompt, chosen_response = extract_prompt_and_response(raw_chosen)
            prompt2, rejected_response = extract_prompt_and_response(raw_rejected)

            # Use stricter check or normalize whitespace if needed
            if prompt.strip() != prompt2.strip():
                # print(f"Skipping due to prompt mismatch:\nP1: {prompt[:50]}...\nP2: {prompt2[:50]}...")
                num_skipped += 1
                continue

            new_dataset['prompt'].append(prompt)
            new_dataset['chosen'].append(chosen_response) # Store only the response part
            new_dataset['rejected'].append(rejected_response) # Store only the response part
        except Exception as e:
            # print(f"Error processing example during dpoify: {e}")
            num_errors += 1
            continue

    print(f'DPOify: Num skipped due to prompt mismatch: {num_skipped}')
    if num_errors > 0:
        print(f'DPOify: Num skipped due to missing keys or errors: {num_errors}')
    if not new_dataset['prompt']:
         print("Warning: DPOify resulted in an empty dataset!")
    return Dataset.from_dict(new_dataset)

# Helper function to filter DPO-formatted dataset by response token length
def filter_dpo_dataset_by_response_length(
    dataset: Dataset, tokenizer: PreTrainedTokenizer, max_response_length: int
) -> Dataset:
    """
    Filters a DPO-formatted dataset based on tokenized length of chosen/rejected RESPONSES.
    Expects dataset with 'chosen', 'rejected' text columns.
    """
    if not dataset or not all(col in dataset.column_names for col in ['chosen', 'rejected']):
        print("Warning: Dataset missing 'chosen' or 'rejected' columns for filtering. Skipping length filtering.")
        return dataset
    if max_response_length <= 0:
        print("Warning: max_response_length is non-positive. Skipping length filtering.")
        return dataset

    initial_count = len(dataset)
    if initial_count == 0: return dataset # Skip if empty

    print(f"Filtering dataset with {initial_count} examples by max response length {max_response_length}...")

    # Define function to check length
    def is_response_length_ok(example):
        len_chosen = len(tokenizer.encode(example['chosen'], add_special_tokens=False))
        len_rejected = len(tokenizer.encode(example['rejected'], add_special_tokens=False))
        return len_chosen <= max_response_length and len_rejected <= max_response_length

    # Use filter directly
    try:
        filtered_dataset = dataset.filter(is_response_length_ok, num_proc=os.cpu_count()) # Use multiple processes
    except Exception as e:
        print(f"Warning: Error during filtering process: {e}. Returning unfiltered dataset.")
        return dataset

    final_count = len(filtered_dataset)
    num_removed = initial_count - final_count
    if num_removed > 0:
        print(
            f'Filtered {num_removed}/{initial_count} examples where chosen/rejected response exceeded max token length {max_response_length}'
        )
    else:
        print("No examples removed by length filtering.")

    return filtered_dataset


# --- Main Function ---
def get_the_datasets(
    tokenizer: PreTrainedTokenizer, # Tokenizer needed for filtering step
    max_length: int, # Max length used for filtering logic buffer
    test: bool = False,
    force_reprocess: bool = False, # Add option to force reprocessing
    data_dir: str = "." # Base directory for saving/loading data
) -> Dataset:
    """
    Loads/processes HH dataset into DPO format with 'prompt', 'chosen', 'rejected' text columns.
    Handles caching of the processed text dataset. Tokenization happens later in collate_fn.
    """
    dset_type = 'test' if test else 'train'
    # Define path for the processed TEXT dataset relative to data_dir
    processed_text_save_path = os.path.join(data_dir, f'dpo_{dset_type}_text_dataset')

    if not force_reprocess:
        try:
            if os.path.exists(processed_text_save_path):
                 # Try loading the dataset with text columns
                dpo_text_dataset = load_from_disk(processed_text_save_path)
                print(f'Loaded processed {dset_type} TEXT dataset from disk: {processed_text_save_path}')
                # Verify required columns
                if not all(col in dpo_text_dataset.column_names for col in ['prompt', 'chosen', 'rejected']):
                    print("Warning: Loaded dataset missing required columns ('prompt', 'chosen', 'rejected'). Reprocessing.")
                    raise ValueError("Loaded dataset format incorrect.") # Trigger reprocessing
                print(f'Final {dset_type} dataset has {len(dpo_text_dataset)} examples')
                return dpo_text_dataset
            else:
                 print(f'Processed {dset_type} text dataset not found at {processed_text_save_path}. Processing originals...')
        except Exception as e:
             print(f"Warning: Error loading dataset from disk ({e}). Reprocessing.")

    # --- Processing starts if loading failed or force_reprocess=True ---
    print("Processing original dataset...")
    # --- Load Raw Data ---
    try:
        # Consider adding cache_dir argument if needed
        dataset = load_dataset('Unified-Language-Model-Alignment/Anthropic_HH_Golden')
        raw_dset = dataset[dset_type]
        print(f"Original {dset_type} dataset has {len(raw_dset)} examples")
    except Exception as e:
         print(f"ERROR: Failed to load raw dataset 'Anthropic_HH_Golden': {e}")
         # Depending on use case, maybe return empty Dataset or raise error
         raise ValueError(f"Could not load raw data for {dset_type}") from e

    # --- DPO-ify ---
    print(f"Creating DPO format for {dset_type} set...")
    dpo_text_dataset = dpoify_dataset(raw_dset)
    if len(dpo_text_dataset) == 0:
        print(f"Warning: DPOification resulted in empty {dset_type} dataset. Check source data and extraction logic.")
        # Return the empty dataset, subsequent steps should handle it gracefully
        return dpo_text_dataset

    # --- Filtering based on DPO format (Now Active) ---
    # Calculate a max length for *responses only* based on the overall max_length
    # Reserve some tokens for the prompt (e.g., 10 or a percentage)
    prompt_buffer = 10
    filter_max_response_len = max_length - prompt_buffer
    print(f"Calculated max response length for filtering: {filter_max_response_len} (max_length={max_length}, buffer={prompt_buffer})")

    if filter_max_response_len <= 0:
        print("Warning: max_length is too small for filtering buffer (<= {prompt_buffer}). Skipping response length filtering.")
    else:
        dpo_text_dataset = filter_dpo_dataset_by_response_length(
            dpo_text_dataset,
            tokenizer,
            filter_max_response_len,
        )
        print(f"Dataset size after response length filtering: {len(dpo_text_dataset)}")

    if len(dpo_text_dataset) == 0:
         print(f"Warning: Dataset became empty after filtering for {dset_type}. Check data and filtering criteria.")
         # Return empty dataset

    # --- Save the processed TEXT dataset ---
    print(f"Saving processed {dset_type} TEXT dataset to disk: {processed_text_save_path}...")
    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(processed_text_save_path), exist_ok=True)
        dpo_text_dataset.save_to_disk(processed_text_save_path)
        print("Dataset saved successfully.")
    except Exception as e:
        print(f"Error saving processed dataset: {e}")


    print(f'Final {dset_type} dataset has {len(dpo_text_dataset)} examples')
    # Return the dataset that has 'prompt', 'chosen', 'rejected' text columns
    return dpo_text_dataset