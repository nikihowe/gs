import os

from transformers import PreTrainedTokenizer

from datasets import Dataset, load_dataset, load_from_disk

# TODO: add better checks that we were able to extract the prompt and response properly
# Helper function to extract prompt and response from HH format
# Note that "Assistant: " is used to separate the prompt and response,
# and is kept as part of the prompt.
def extract_prompt_and_response(example: str) -> tuple[str, str]:
    """Extracts prompt and response assuming HH format."""
    match_string = '\n\nAssistant: '
    assistant_start_idx = example.rfind(match_string)
    if assistant_start_idx == -1 or assistant_start_idx == 0:
        if assistant_start_idx == -1:
            return example, ''  # TODO: Handle this case better
        return (
            match_string,
            example[len(match_string) :],
        )  # No prompt, just response
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
                num_errors += 1
                continue

            prompt, chosen_response = extract_prompt_and_response(raw_chosen)
            prompt2, rejected_response = extract_prompt_and_response(
                raw_rejected
            )

            if prompt != prompt2:
                num_skipped += 1
                continue

            new_dataset['prompt'].append(prompt)
            new_dataset['chosen'].append(chosen_response)
            new_dataset['rejected'].append(rejected_response)
        except Exception as e:
            num_errors += 1
            continue

    print(f'DPOify: Num skipped due to prompt mismatch: {num_skipped}')
    if num_errors > 0:
        print(
            f'DPOify: Num skipped due to missing keys or errors: {num_errors}'
        )
    if not new_dataset['prompt']:
        print('Warning: DPOify resulted in an empty dataset!')
    return Dataset.from_dict(new_dataset)


# Helper function to filter DPO-formatted dataset by response token length
# Since most of the datapoints are <512 tokens, we do this to speed
# things up.
# TODO: decide if this is actually necessary given our new padding scheme.
# TODO: simplify this function
def filter_dpo_dataset_by_response_length(
    dataset: Dataset, tokenizer: PreTrainedTokenizer, max_response_length: int
) -> Dataset:
    """
    Filters a DPO-formatted dataset based on tokenized length of chosen/rejected RESPONSES.
    Ensures response lengths are > 0 and <= max_response_length.
    Expects dataset with 'chosen', 'rejected' text columns.
    """
    if not dataset or not all(
        col in dataset.column_names for col in ['chosen', 'rejected']
    ):
        print(
            "Warning: Dataset missing 'chosen' or 'rejected' columns for filtering. Skipping length filtering."
        )
        return dataset
    # Allow max_response_length <= 0 to disable MAX length filtering,
    # but we still want to filter ZERO length responses.

    initial_count = len(dataset)
    if initial_count == 0:
        return dataset  # Skip if empty

    print(
        f'Filtering dataset with {initial_count} examples by response length (must be > 0'
        + (f' and <= {max_response_length}' if max_response_length > 0 else '')
        + ')...'
    )

    # Define function to check length
    def is_response_length_ok(example):
        # Get text safely, default to empty string if missing or None
        chosen_text = example.get('chosen', '')
        rejected_text = example.get('rejected', '')

        # Check for potentially empty strings before tokenization (optional but good practice)
        if not chosen_text or not rejected_text:
            return False

        # Tokenize (without adding special tokens for length check)
        chosen_tokens = tokenizer.encode(chosen_text, add_special_tokens=False)
        rejected_tokens = tokenizer.encode(
            rejected_text, add_special_tokens=False
        )

        len_chosen = len(chosen_tokens)
        len_rejected = len(rejected_tokens)

        # Ensure length is strictly positive AND respects max_length (if specified)
        is_len_ok = len_chosen > 0 and len_rejected > 0
        if max_response_length > 0:
            is_len_ok = is_len_ok and (
                len_chosen <= max_response_length
                and len_rejected <= max_response_length
            )

        return is_len_ok

    # Use filter directly
    try:
        # Determine num_proc safely
        num_proc = min(
            os.cpu_count(), 8
        )   # Limit cpu usage somewhat if many cores
        print(f'Using {num_proc} processes for filtering...')
        filtered_dataset = dataset.filter(
            is_response_length_ok, num_proc=num_proc
        )
    except Exception as e:
        print(
            f'Warning: Error during filtering process: {e}. Trying with num_proc=1.'
        )
        # Fallback to single-process filtering
        try:
            print('Retrying filtering with num_proc=1...')
            filtered_dataset = dataset.filter(
                is_response_length_ok, num_proc=1
            )
        except Exception as e2:
            print(
                f'Single-process filtering also failed: {e2}. Returning unfiltered.'
            )
            return dataset   # Return unfiltered on error

    final_count = len(filtered_dataset)
    num_removed = initial_count - final_count
    if num_removed > 0:
        print(
            f'Filtered {num_removed}/{initial_count} examples due to response length constraints (length was 0 or > {max_response_length} [if specified])'
        )
    else:
        print('No examples removed by length filtering.')

    return filtered_dataset


def get_the_datasets(
    tokenizer: PreTrainedTokenizer,  # Tokenizer needed for filtering step
    max_length: int,  # Max length used for filtering logic buffer
    test: bool = False,
    force_reprocess: bool = False,  # Add option to force reprocessing
    processed_cache_base: str = '/tmp',
) -> Dataset:
    """
    Loads/processes HH dataset into DPO format with 'prompt', 'chosen', 'rejected' text columns.
    Handles caching of the processed text dataset in `processed_cache_base`.
    Tokenization happens later in collate_fn.
    """
    dset_type = 'test' if test else 'train'

    # Define path for the processed TEXT dataset within the specified cache base
    processed_text_save_path = os.path.join(
        processed_cache_base,
        f'dpo_{dset_type}_text_dataset_ml{max_length}',  # Added max_length to name
    )
    print(f'Using processed dataset cache path: {processed_text_save_path}')

    # Try to load from disk if possible
    if not force_reprocess:
        try:
            if os.path.exists(processed_text_save_path):
                dpo_text_dataset = load_from_disk(processed_text_save_path)
                print(
                    f'Loaded processed {dset_type} TEXT dataset from disk: {processed_text_save_path}'
                )
                # Verify required columns
                if not all(
                    col in dpo_text_dataset.column_names
                    for col in ['prompt', 'chosen', 'rejected']
                ):
                    print(
                        "Warning: Loaded dataset missing required columns ('prompt', 'chosen', 'rejected'). Reprocessing."
                    )
                    raise ValueError('Loaded dataset format incorrect.')
                print(
                    f'Final {dset_type} dataset has {len(dpo_text_dataset)} examples'
                )
                return dpo_text_dataset
            else:
                print(
                    f'Processed {dset_type} text dataset not found at {processed_text_save_path}. Processing originals...'
                )
        except Exception as e:
            print(
                f'Warning: Error loading dataset from disk ({e}). Reprocessing.'
            )

    # --- Processing starts if loading failed or force_reprocess=True ---
    print('Processing original dataset...')
    # --- Load Raw Data ---
    try:
        dataset = load_dataset(
            'Unified-Language-Model-Alignment/Anthropic_HH_Golden',
        )
        raw_dset = dataset[dset_type]
        print(f'Original {dset_type} dataset has {len(raw_dset)} examples')
    except Exception as e:
        print(f"ERROR: Failed to load raw dataset 'Anthropic_HH_Golden': {e}")
        raise ValueError(f'Could not load raw data for {dset_type}') from e

    # --- DPO-ify ---
    print(f'Creating DPO format for {dset_type} set...')
    dpo_text_dataset = dpoify_dataset(raw_dset)
    if len(dpo_text_dataset) == 0:
        print(
            f'Warning: DPOification resulted in empty {dset_type} dataset. Check source data and extraction logic.'
        )
        return dpo_text_dataset   # Return empty

    # --- Filtering based on DPO format (Now Active) ---
    # We want the prompt and the response to be below the max length,
    # So adding a buffer here to make sure that when we put them 
    # together, nothing weird happens. It should not be necessary.
    # TODO: check if this is necessary.
    prompt_buffer = 10
    filter_max_response_len = max_length - prompt_buffer
    print(
        f'Calculated max response length for filtering: {filter_max_response_len} (max_length={max_length}, buffer={prompt_buffer})'
    )

    if filter_max_response_len <= 0:
        print(
            f'Warning: max_length ({max_length}) is too small for filtering buffer (<= {prompt_buffer}). Skipping response length filtering.'
        )
    else:
        dpo_text_dataset = filter_dpo_dataset_by_response_length(
            dpo_text_dataset,
            tokenizer,
            filter_max_response_len,
        )
        print(
            f'Dataset size after response length filtering: {len(dpo_text_dataset)}'
        )

    if len(dpo_text_dataset) == 0:
        print(
            f'Warning: Dataset became empty after filtering for {dset_type}. Check data and filtering criteria.'
        )
        # Return empty dataset - Don't try to save an empty one

    # --- Save the processed TEXT dataset ---
    # Only save if the dataset is not empty
    if len(dpo_text_dataset) > 0:
        print(
            f'Saving processed {dset_type} TEXT dataset to disk: {processed_text_save_path}...'
        )
        try:
            # Ensure parent directory exists (e.g., /tmp/ might exist but maybe not /tmp/some_subdir/)
            os.makedirs(
                os.path.dirname(processed_text_save_path), exist_ok=True
            )
            # Save using the new path
            dpo_text_dataset.save_to_disk(processed_text_save_path)
            print('Dataset saved successfully.')
        except Exception as e:
            print(
                f'Error saving processed dataset to {processed_text_save_path}: {e}'
            )
            raise e
    elif not force_reprocess and os.path.exists(processed_text_save_path):
        try:
            print(
                f'Attempting to remove potentially invalid empty cache: {processed_text_save_path}'
            )
            import shutil

            shutil.rmtree(processed_text_save_path)
        except Exception as e_rm:
            print(
                f'Warning: Failed to remove empty cache directory {processed_text_save_path}: {e_rm}'
            )

    print(f'Final {dset_type} dataset has {len(dpo_text_dataset)} examples')
    # Return the dataset that has 'prompt', 'chosen', 'rejected' text columns
    return dpo_text_dataset
