import json
import math  # For isnan/isfinite checks if needed
import os

import torch
import torch.nn.functional as F
from colorama import Fore, Style
from colorama import init as colorama_init
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize colorama
colorama_init(autoreset=True)

# --- Global DEBUG Flag ---
# Set to True to enable detailed debugging print statements
# Set to False to hide debugging print statements
DEBUG = False   # <<< Set this to False to turn off debug prints

# Define device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')   # User info, keep visible

# ==============================================================================
# Visualization Helper (5-Color Scale)
# ==============================================================================


def visualize_attribution(
    tokens_with_attributions: list[tuple[str, float]],
    max_attr: float,  # Max score used for normalization (e.g., max positive drop)
    method: str = "loo",
) -> str:
    """
    Visualizes attribution scores using a 5-color scale.
    Assumes higher positive score means more importance.

    Args:
        tokens_with_attributions: List of (token_string, score) tuples.
        max_attr: The maximum *positive* score for normalization.

    Returns:
        A string with tokens colored based on their normalized scores.
    """
    colored_text = ''
    # Use a small epsilon to avoid division by zero and handle zero max_attr
    if max_attr <= 1e-9:
        # If max_attr is zero or negative, all scores contributing positively are zero or less.
        # In this case, all tokens will likely be green. Use 1.0 to avoid errors.
        if DEBUG:
            print(
                f'DEBUG: max_attr ({max_attr:.4f}) is zero or negative, using 1.0 for scaling.'
            )
        max_attr = 1.0

    # --- Define Thresholds for 5 Colors (Green < Cyan < Yellow < Magenta < Red) ---
    # !!! IMPORTANT: TUNE THESE THRESHOLDS BASED ON YOUR OBSERVED LOO SCORE RANGE !!!
    # LOO scores (log prob drop) might be small values. Check DEBUG output.
    # Example thresholds assuming max_attr corresponds to a significant drop:
    if method == "loo":
        threshold_cyan = 0.05   # Score > 5% of max drop
        threshold_yellow = 0.15   # Score > 15% of max drop
        threshold_magenta = 0.30   # Score > 30% of max drop
        threshold_red = 0.60   # Score > 60% of max drop
    else:
        threshold_cyan = 0.005  # Scores > 0.005 (0.5% of max) up to 0.010 are Cyan
        threshold_yellow = (
            0.010  # Scores > 0.010 (1.0% of max) up to 0.015 are Yellow
        )
        threshold_magenta = (
            0.015  # Scores > 0.015 (1.5% of max) up to 0.500 are Magenta
        )
        threshold_red = (
            0.500  # Scores > 0.500 (50% of max) are Red (Should catch 'Human')
        )
    # --- Scores <= threshold_cyan will be Green ---

    if DEBUG:
        print(
            f'DEBUG: Visualizing with max_attr={max_attr:.4f}, Thresh: G<={threshold_cyan:.4f}<C<={threshold_yellow:.4f}<Y<={threshold_magenta:.4f}<M<={threshold_red:.4f}<R'
        )

    for token, score in tokens_with_attributions:
        # Ensure score is valid float, default to 0 otherwise
        if not isinstance(score, (float, int)) or not math.isfinite(score):
            score = 0.0

        # Normalize score: Use max(0, score) to focus on positive importance (probability drop)
        # Scale relative to max_attr. Clamp between 0 and 1.
        norm_attribution = min(1.0, max(0.0, score) / max_attr)

        # Determine color based on thresholds
        if norm_attribution > threshold_red:
            color = Fore.RED      # Highest importance
        elif norm_attribution > threshold_magenta:
            color = Fore.MAGENTA  # High-Medium importance
        elif norm_attribution > threshold_yellow:
            color = Fore.YELLOW   # Medium importance
        elif norm_attribution > threshold_cyan:
            color = Fore.CYAN     # Low-Medium importance
        else:
            color = Fore.GREEN    # Lowest importance (or negative score)

        # Handle token display string (handling spaces, bytes)
        display_token = token
        if isinstance(token, str) and token.startswith('Ġ'):
            display_token = ' ' + token[1:]
        elif isinstance(token, bytes):
            display_token = token.decode('utf-8', errors='replace')

        colored_text += f'{color}{display_token}{Style.RESET_ALL}'
    return colored_text


# ==============================================================================
# Leave-One-Out (LOO) Token Attribution Function
# ==============================================================================


def token_attribution_loo(  # Renamed from original code
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    completion: str,
) -> list[tuple[str, float]]:
    """
    Calculates token attribution scores using Leave-One-Out (LOO) method.
    Scores represent the drop in average log probability of the completion
    when a prompt token is masked. Higher score = more important token.
    """
    model.eval()
    if DEBUG:
        print('\n--- DEBUG: Inside token_attribution_loo ---')

    # --- Tokenization using combined text and offset mapping ---
    full_text = prompt + completion
    prompt_content_len = len(
        prompt.rstrip()
    )   # Use stripped length for boundary

    # Check for empty prompt or completion early
    if not prompt.strip():
        print(
            'Error: Prompt is empty or only whitespace.'
        )   # Keep Error visible
        return []
    if not completion.strip():
        print(
            'Error: Completion is empty or only whitespace.'
        )   # Keep Error visible
        return []

    try:
        inputs = tokenizer(
            full_text,
            return_tensors='pt',
            truncation=True,
            max_length=getattr(model.config, 'max_position_embeddings', 512),
            return_offsets_mapping=True,
        ).to(device)
    except Exception as e:
        print(f'Error during tokenization: {e}')   # Keep Error visible
        return []

    input_ids = inputs['input_ids']   # Shape: (1, sequence_length)
    # Check if tokenization resulted in empty input_ids (e.g., due to truncation)
    if input_ids.shape[1] == 0:
        print(
            'Error: Tokenization resulted in empty input_ids.'
        )   # Keep Error visible
        return []

    attention_mask = inputs['attention_mask']
    # Handle case where offset_mapping might not be returned by tokenizer if truncation happened strangely
    offsets = inputs.get('offset_mapping')
    if offsets is None:
        print(
            'Error: Tokenizer did not return offset_mapping. Cannot proceed with robust indexing.'
        )   # Keep Error visible
        return []
    offsets = offsets[0].tolist()
    sequence_length = input_ids.shape[1]

    # --- Determine Prompt/Completion Indices ---
    prompt_start_idx = 0   # Index of first content token in prompt
    if (
        getattr(tokenizer, 'add_bos_token', False)
        and input_ids[0, 0] == tokenizer.bos_token_id
    ):
        prompt_start_idx = 1

    completion_token_start_index = -1
    for idx, (start_char, end_char) in enumerate(offsets):
        # Basic skip for special tokens represented by (0,0)
        if start_char == 0 and end_char == 0:
            if idx == 0 and prompt_start_idx == 1:
                continue   # Allow BOS
            # Add check for PAD token if necessary
            if input_ids[0, idx] == tokenizer.pad_token_id:
                continue
            # May need more robust special token handling depending on tokenizer
            else:
                continue
        # Find first token starting at or after the stripped prompt content ends
        if start_char >= prompt_content_len:
            completion_token_start_index = idx
            break
    if completion_token_start_index == -1:
        completion_token_start_index = sequence_length   # Assume completion is empty if boundary not found

    prompt_end_idx = completion_token_start_index   # Prompt ends before first completion token

    # Indices relative to the sequence start (ensure they are within bounds)
    prompt_content_indices = list(
        range(prompt_start_idx, min(prompt_end_idx, sequence_length))
    )
    completion_indices = list(
        range(
            min(completion_token_start_index, sequence_length), sequence_length
        )
    )

    # Ensure prompt/completion indices are valid and non-empty
    if not prompt_content_indices:
        print(
            'Error: No prompt content tokens found after index calculation.'
        )   # Keep Error visible
        return []
    # Need at least one completion token to calculate probability
    if not completion_indices:
        print(
            'Error: No completion tokens found after index calculation.'
        )   # Keep Error visible
        return []

    # Get prompt token strings ONCE using convert_ids_to_tokens
    prompt_token_ids = input_ids[0, prompt_content_indices]
    prompt_token_strings = tokenizer.convert_ids_to_tokens(prompt_token_ids)
    if DEBUG:
        print(
            f'DEBUG: Prompt content tokens ({len(prompt_token_strings)}): {prompt_token_strings}'
        )

    completion_token_ids = input_ids[
        0, completion_indices
    ]   # Needed for manual calculation if used
    if DEBUG:
        print(
            f'DEBUG: Completion tokens ({len(completion_indices)}): {tokenizer.convert_ids_to_tokens(completion_token_ids)}'
        )

    # --- Calculate Baseline Log Probability ---
    baseline_log_prob = 0.0
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            logits = outputs.logits   # Shape: (1, sequence_length, vocab_size)

            # Shift logits and labels for next token prediction loss calculation
            # Logits predicting the completion tokens start one position earlier in the sequence
            shift_logits = logits[
                :, completion_token_start_index - 1 : sequence_length - 1, :
            ]
            # Labels are the actual completion token IDs
            shift_labels = input_ids[
                :, completion_token_start_index:sequence_length
            ]

            # Ensure shapes match for loss calculation
            if shift_logits.shape[1] == 0 or shift_labels.shape[1] == 0:
                print(
                    'Error: Cannot calculate baseline loss due to zero-length logits or labels after slicing.'
                )   # Keep Error visible
                if DEBUG:
                    print(
                        f'DEBUG: Shift logits shape: {shift_logits.shape}, Shift labels shape: {shift_labels.shape}'
                    )
                return []
            if shift_logits.shape[1] != shift_labels.shape[1]:
                print(
                    'Error: Mismatch between shifted logits and labels lengths.'
                )   # Keep Error visible
                if DEBUG:
                    print(
                        f'DEBUG: Shift logits shape: {shift_logits.shape}, Shift labels shape: {shift_labels.shape}'
                    )
                # Adjust length if off by one (common issue)
                min_len = min(shift_logits.shape[1], shift_labels.shape[1])
                shift_logits = shift_logits[:, :min_len, :]
                shift_labels = shift_labels[:, :min_len]
                if DEBUG:
                    print(
                        f'DEBUG: Adjusted shapes to: {shift_logits.shape}, {shift_labels.shape}'
                    )

            # Use cross_entropy loss (average negative log likelihood)
            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
            neg_log_likelihood = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            baseline_log_prob = (
                -neg_log_likelihood.item()
            )   # Average log probability

        if DEBUG:
            print(f'DEBUG: Baseline Avg Log Prob: {baseline_log_prob:.4f}')

    except Exception as e:
        print(
            f'Error calculating baseline probability: {e}'
        )   # Keep Error visible
        if DEBUG:   # Optionally show traceback only in debug mode
            import traceback

            traceback.print_exc()
        return []

    # --- Loop through prompt tokens to mask ---
    attribution_scores = []
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        print(
            'Error: tokenizer.pad_token_id is None. Cannot mask tokens.'
        )   # Keep Error visible
        return []

    # Iterate using the indices relative to the start of the sequence
    for i, token_index_in_sequence in enumerate(prompt_content_indices):
        token_str = prompt_token_strings[i]   # Get the correct string
        # Skip masking pad tokens if they somehow ended up in prompt_content_indices
        if input_ids[0, token_index_in_sequence] == pad_token_id:
            if DEBUG:
                print(
                    f'DEBUG: Skipping masking of PAD token at index {token_index_in_sequence}'
                )
            attribution_scores.append(
                (token_str, 0.0)
            )   # Assign zero score to PAD
            continue

        if DEBUG:
            print(
                f"\nDEBUG: Masking token {i}: '{token_str}' (Index in sequence: {token_index_in_sequence})"
            )

        # Create masked inputs
        masked_input_ids = input_ids.clone()
        masked_attention_mask = attention_mask.clone()
        masked_input_ids[0, token_index_in_sequence] = pad_token_id
        masked_attention_mask[
            0, token_index_in_sequence
        ] = 0   # Mask attention

        # --- Calculate Masked Log Probability ---
        masked_log_prob = 0.0
        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=masked_input_ids,
                    attention_mask=masked_attention_mask,
                    return_dict=True,
                )
                logits = outputs.logits

                # Calculate loss only for completion part, same way as baseline
                shift_logits = logits[
                    :,
                    completion_token_start_index - 1 : sequence_length - 1,
                    :,
                ]
                # Use original labels for comparison!
                shift_labels = input_ids[
                    :, completion_token_start_index:sequence_length
                ]

                # Ensure shapes match (repeat check for safety)
                if shift_logits.shape[1] == 0 or shift_labels.shape[1] == 0:
                    print(
                        f'Error: Cannot calculate masked loss for token {i} ({token_str}) due to zero-length logits/labels.'
                    )   # Keep Error visible
                    masked_log_prob = -float(
                        'inf'
                    )   # Indicate error with very low log prob
                elif shift_logits.shape[1] != shift_labels.shape[1]:
                    print(
                        f'Error: Mismatch masked logits/labels for token {i} ({token_str}). Adjusting.'
                    )   # Keep Error visible
                    min_len = min(shift_logits.shape[1], shift_labels.shape[1])
                    shift_logits = shift_logits[:, :min_len, :]
                    shift_labels = shift_labels[:, :min_len]
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
                    neg_log_likelihood = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                    masked_log_prob = -neg_log_likelihood.item()
                else:
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
                    neg_log_likelihood = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                    masked_log_prob = -neg_log_likelihood.item()

            if DEBUG:
                print(f'DEBUG: Masked Avg Log Prob: {masked_log_prob:.4f}')

            # --- Calculate Score ---
            # Score = Drop in log prob (baseline - masked)
            # Higher score means original token was more important (removing it lowered the prob)
            # Handle potential -inf from error case
            if masked_log_prob == -float('inf'):
                score = -float('inf')   # Propagate error indication
                if DEBUG:
                    print(f'DEBUG: Score (Drop): Error Propagated')
            else:
                score = baseline_log_prob - masked_log_prob
                if DEBUG:
                    print(f'DEBUG: Score (Drop): {score:.4f}')

            attribution_scores.append((token_str, score))

        except Exception as e:
            print(
                f"Error calculating masked probability for token {i} ('{token_str}'): {e}"
            )   # Keep Error visible
            if DEBUG:   # Optionally show traceback only in debug mode
                import traceback

                traceback.print_exc()
            attribution_scores.append(
                (token_str, 0.0)
            )   # Append zero score on error

    if DEBUG:
        print(
            f'\nDEBUG: Final LOO attribution_scores (first 10): {attribution_scores[:10]}'
        )
        print('--- END DEBUG: Exiting token_attribution_loo ---')
    return attribution_scores


# ==============================================================================
# Main Execution Block (Using LOO Attribution)
# ==============================================================================
if __name__ == '__main__':
    colorama_init(autoreset=True)   # Initialize colorama

    token_attribution = token_attribution_loo
    method = "loo"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # --- Configuration ---
    # checkpoint = 'HuggingFaceTB/SmolLM2-135M-Instruct'
    checkpoint = 'HuggingFaceTB/SmolLM2-1.7B-Instruct'
    # trained_model_path = 'big_dpo_model'
    # trained_model_path = 'small_dpo_model_revised_dl_stats'
    trained_model_path = 'large_long_dpo_final_model'
    dataset_path = './datasets/dataset.json'
    gen_max_length = 150
    num_good_to_show = 5  # Number of HARMLESS examples
    num_bad_to_show = 5   # Number of HARMFUL examples

    # --- Load Models and Tokenizer ---
    print(f'Loading tokenizer from: {checkpoint}')
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, trust_remote_code=True
        )
        print(f'Loading base model from: {checkpoint}')
        base_model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            attn_implementation='eager',
        ).to(device)
        print(f'Loading tuned model from: {trained_model_path}')
        if not os.path.isdir(trained_model_path):
            raise FileNotFoundError(
                f'Tuned model directory not found: {trained_model_path}'
            )
        trained_model = AutoModelForCausalLM.from_pretrained(
            trained_model_path,
            trust_remote_code=True,
            attn_implementation='eager',
        ).to(device)
    except Exception as e:
        print(f'\nError loading model/tokenizer: {e}')
        exit(1)

    # --- Tokenizer Pad Token Handling ---
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print(
                f'Set tokenizer pad_token to eos_token ({tokenizer.eos_token})'
            )
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print("Added '[PAD]' as pad_token.")
        base_model.resize_token_embeddings(len(tokenizer))
        trained_model.resize_token_embeddings(len(tokenizer))
    base_model.config.pad_token_id = tokenizer.pad_token_id
    trained_model.config.pad_token_id = tokenizer.pad_token_id

    # --- Set Models to Evaluation Mode ---
    base_model.eval()
    trained_model.eval()
    print('Models set to evaluation mode.')

    # --- Load Dataset ---
    print(f'Loading dataset from: {dataset_path}')
    try:
        with open(dataset_path, 'r', encoding='utf-8') as fp:
            dataset = json.load(fp)
    except FileNotFoundError:
        print(f'Error: Dataset file not found at {dataset_path}')
        exit(1)
    except json.JSONDecodeError as e:
        print(
            f'Error: Could not decode JSON from {dataset_path}. Details: {e}'
        )
        exit(1)
    except Exception as e:
        print(f'An unexpected error occurred loading the dataset: {e}')
        exit(1)

    # Validate dataset structure
    if (
        not isinstance(dataset, dict)
        or 'text' not in dataset
        or 'harmful' not in dataset
    ):
        print(
            "Error: Dataset JSON must be an object with 'text' and 'harmful' keys."
        )
        exit(1)
    if not isinstance(dataset['text'], list) or not isinstance(
        dataset['harmful'], list
    ):
        print("Error: Dataset 'text' and 'harmful' values must be lists.")
        exit(1)
    if len(dataset['text']) != len(dataset['harmful']):
        print(
            f"Error: Dataset 'text' ({len(dataset['text'])}) and 'harmful' ({len(dataset['harmful'])}) lists must have the same length."
        )
        exit(1)

    print(f"Loaded dataset with {len(dataset['text'])} examples.")

    # --- Prepare Examples (Harmless and Harmful) ---
    good_examples = [
        text
        for text, label in zip(dataset['text'], dataset['harmful'])
        if isinstance(label, (int, float)) and label == 0
    ]
    bad_examples = [  # Filter for harmful examples
        text
        for text, label in zip(dataset['text'], dataset['harmful'])
        if isinstance(label, (int, float))
        and label == 1  # Label 1 indicates harmful
    ]

    # Add standard prompt formatting
    good_examples_prompts = [
        'Human: ' + str(text) + ' Assistant: ' for text in good_examples
    ]
    bad_examples_prompts = [
        'Human: ' + str(text) + ' Assistant: ' for text in bad_examples
    ]

    num_available_good = len(good_examples_prompts)
    num_available_bad = len(bad_examples_prompts)
    print(f"Found {num_available_good} 'good' (harmless, label=0) examples.")
    print(
        f"Found {num_available_bad} 'bad' (harmful, label=1) examples."
    )   # Info about harmful count

    # Determine actual number to process based on availability
    actual_good_to_show = min(num_good_to_show, num_available_good)
    actual_bad_to_show = min(num_bad_to_show, num_available_bad)

    # --- Shared Generation Arguments ---
    gen_args_base = {
        'num_return_sequences': 1,
        'eos_token_id': tokenizer.eos_token_id,
        'pad_token_id': tokenizer.pad_token_id,
        'do_sample': False,  # Use greedy decoding
        'no_repeat_ngram_size': 3,
    }

    # === Process HARMLESS Examples ===
    if actual_good_to_show > 0:
        print(f'\n=== Processing {actual_good_to_show} HARMLESS Examples ===')
        print('=' * 60)
        for i in range(actual_good_to_show):
            current_prompt = good_examples_prompts[i]
            print(f'\n--- Harmless Example {i+1} / {actual_good_to_show} ---')
            print(
                f"Prompt: {current_prompt[:300]}{'...' if len(current_prompt)>300 else ''}"
            )

            base_model_completion = '[GENERATION SKIPPED]'
            tuned_model_completion = '[GENERATION SKIPPED]'

            # --- Generate Actual Completions ---
            try:
                inputs_tokenized = tokenizer(
                    current_prompt,
                    return_tensors='pt',
                    truncation=True,
                    max_length=tokenizer.model_max_length
                    - gen_max_length,  # Reserve space for generation
                ).to(device)
                current_input_len = len(inputs_tokenized['input_ids'][0])
                gen_args = gen_args_base.copy()
                gen_args['max_length'] = current_input_len + gen_max_length

                print('\nGenerating base model response (Harmless)...')
                with torch.no_grad():
                    base_model_output = base_model.generate(
                        input_ids=inputs_tokenized.input_ids,
                        attention_mask=inputs_tokenized.attention_mask,
                        **gen_args,
                    )
                base_model_generated_ids = base_model_output[
                    0, current_input_len:
                ]
                base_model_completion = tokenizer.decode(
                    base_model_generated_ids, skip_special_tokens=True
                ).strip()
                print(
                    f"Base Model Generated: '{base_model_completion[:150]}...'"
                )

                print('Generating tuned model response (Harmless)...')
                with torch.no_grad():
                    tuned_model_output = trained_model.generate(
                        input_ids=inputs_tokenized.input_ids,
                        attention_mask=inputs_tokenized.attention_mask,
                        **gen_args,
                    )
                tuned_model_generated_ids = tuned_model_output[
                    0, current_input_len:
                ]
                tuned_model_completion = tokenizer.decode(
                    tuned_model_generated_ids, skip_special_tokens=True
                ).strip()
                print(
                    f"Tuned Model Generated: '{tuned_model_completion[:150]}...'"
                )

            except Exception as e:
                print(
                    f'Error during generation for harmless example {i+1}: {e}'
                )
                if base_model_completion == '[GENERATION SKIPPED]':
                    base_model_completion = '[GENERATION ERROR]'
                if tuned_model_completion == '[GENERATION SKIPPED]':
                    tuned_model_completion = '[GENERATION ERROR]'

            # --- Calculate Attributions ---
            base_model_attribution = []
            tuned_model_attribution = []

            if base_model_completion and base_model_completion not in [
                '[GENERATION ERROR]',
                '[GENERATION SKIPPED]',
            ]:
                print(
                    f'\nCalculating attribution for BASE model (Harmless)...'
                )
                base_model_attribution = token_attribution(
                    base_model,
                    tokenizer,
                    current_prompt,
                    base_model_completion,
                )
            else:
                print('\nSkipping base model attribution (Harmless).')

            if tuned_model_completion and tuned_model_completion not in [
                '[GENERATION ERROR]',
                '[GENERATION SKIPPED]',
            ]:
                print(f'Calculating attribution for TUNED model (Harmless)...')
                tuned_model_attribution = token_attribution(
                    trained_model,
                    tokenizer,
                    current_prompt,
                    tuned_model_completion,
                )
            else:
                print('Skipping tuned model attribution (Harmless).')

            # --- Visualize Attributions ---
            print('\n--- Attributions (Harmless) ---')
            if base_model_attribution:
                valid_scores = [
                    attr
                    for _, attr in base_model_attribution
                    if isinstance(attr, (int, float))
                    and torch.isfinite(torch.tensor(attr))
                ]
                max_base_attr = (
                    max(valid_scores) if valid_scores else 1.0
                )   # Avoid division by zero
                print(
                    f'DEBUG: Raw Base Attr Scores (first 10): {base_model_attribution[:10]}'
                )
                print(
                    'Base model: ',
                    visualize_attribution(
                        base_model_attribution, max_attr=max_base_attr, method=method,
                    ),
                    base_model_completion,
                )
            else:
                print(
                    f'Base model: [Attribution not calculated] {base_model_completion}'
                )

            if tuned_model_attribution:
                valid_scores = [
                    attr
                    for _, attr in tuned_model_attribution
                    if isinstance(attr, (int, float))
                    and torch.isfinite(torch.tensor(attr))
                ]
                max_tuned_attr = max(valid_scores) if valid_scores else 1.0
                print(
                    f'DEBUG: Raw Tuned Attr Scores (first 10): {tuned_model_attribution[:10]}'
                )
                print(
                    'Tuned model:',
                    visualize_attribution(
                        tuned_model_attribution, max_attr=max_tuned_attr, method=method,
                    ),
                    tuned_model_completion,
                )
            else:
                print(
                    f'Tuned model: [Attribution not calculated] {tuned_model_completion}'
                )

            print('-' * 60)   # Separator between examples
    else:
        print('\nNo harmless examples found or requested to process.')

    # === Process HARMFUL Examples ===
    if actual_bad_to_show > 0:
        print(f'\n=== Processing {actual_bad_to_show} HARMFUL Examples ===')
        print('=' * 60)
        for i in range(actual_bad_to_show):
            current_prompt = bad_examples_prompts[
                i
            ]   # Use the harmful prompts list
            print(
                f'\n--- Harmful Example {i+1} / {actual_bad_to_show} ---'
            )   # Indicate harmful
            print(
                f"Prompt: {current_prompt[:300]}{'...' if len(current_prompt)>300 else ''}"
            )

            base_model_completion = '[GENERATION SKIPPED]'
            tuned_model_completion = '[GENERATION SKIPPED]'

            # --- Generate Actual Completions ---
            # (This code block is identical to the harmless one, just uses the harmful prompt)
            try:
                inputs_tokenized = tokenizer(
                    current_prompt,
                    return_tensors='pt',
                    truncation=True,
                    max_length=tokenizer.model_max_length
                    - gen_max_length,  # Reserve space for generation
                ).to(device)
                current_input_len = len(inputs_tokenized['input_ids'][0])
                gen_args = gen_args_base.copy()
                gen_args['max_length'] = current_input_len + gen_max_length

                print('\nGenerating base model response (Harmful)...')
                with torch.no_grad():
                    base_model_output = base_model.generate(
                        input_ids=inputs_tokenized.input_ids,
                        attention_mask=inputs_tokenized.attention_mask,
                        **gen_args,
                    )
                base_model_generated_ids = base_model_output[
                    0, current_input_len:
                ]
                base_model_completion = tokenizer.decode(
                    base_model_generated_ids, skip_special_tokens=True
                ).strip()
                print(
                    f"Base Model Generated: '{base_model_completion[:150]}...'"
                )

                print('Generating tuned model response (Harmful)...')
                with torch.no_grad():
                    tuned_model_output = trained_model.generate(
                        input_ids=inputs_tokenized.input_ids,
                        attention_mask=inputs_tokenized.attention_mask,
                        **gen_args,
                    )
                tuned_model_generated_ids = tuned_model_output[
                    0, current_input_len:
                ]
                tuned_model_completion = tokenizer.decode(
                    tuned_model_generated_ids, skip_special_tokens=True
                ).strip()
                print(
                    f"Tuned Model Generated: '{tuned_model_completion[:150]}...'"
                )

            except Exception as e:
                print(
                    f'Error during generation for harmful example {i+1}: {e}'
                )
                if base_model_completion == '[GENERATION SKIPPED]':
                    base_model_completion = '[GENERATION ERROR]'
                if tuned_model_completion == '[GENERATION SKIPPED]':
                    tuned_model_completion = '[GENERATION ERROR]'

            # --- Calculate Attributions ---
            base_model_attribution = []
            tuned_model_attribution = []

            if base_model_completion and base_model_completion not in [
                '[GENERATION ERROR]',
                '[GENERATION SKIPPED]',
            ]:
                print(f'\nCalculating attribution for BASE model (Harmful)...')
                base_model_attribution = token_attribution(
                    base_model,
                    tokenizer,
                    current_prompt,
                    base_model_completion,
                )
            else:
                print('\nSkipping base model attribution (Harmful).')

            if tuned_model_completion and tuned_model_completion not in [
                '[GENERATION ERROR]',
                '[GENERATION SKIPPED]',
            ]:
                print(f'Calculating attribution for TUNED model (Harmful)...')
                tuned_model_attribution = token_attribution(
                    trained_model,
                    tokenizer,
                    current_prompt,
                    tuned_model_completion,
                )
            else:
                print('Skipping tuned model attribution (Harmful).')

            # --- Visualize Attributions ---
            print('\n--- Attributions (Harmful) ---')   # Indicate harmful
            if base_model_attribution:
                valid_scores = [
                    attr
                    for _, attr in base_model_attribution
                    if isinstance(attr, (int, float))
                    and torch.isfinite(torch.tensor(attr))
                ]
                max_base_attr = max(valid_scores) if valid_scores else 1.0
                print(
                    f'DEBUG: Raw Base Attr Scores (first 10): {base_model_attribution[:10]}'
                )
                print(
                    'Base model: ',
                    visualize_attribution(
                        base_model_attribution, max_attr=max_base_attr, method=method,
                    ),
                    base_model_completion,
                )
            else:
                print(
                    f'Base model: [Attribution not calculated] {base_model_completion}'
                )

            if tuned_model_attribution:
                valid_scores = [
                    attr
                    for _, attr in tuned_model_attribution
                    if isinstance(attr, (int, float))
                    and torch.isfinite(torch.tensor(attr))
                ]
                max_tuned_attr = max(valid_scores) if valid_scores else 1.0
                print(
                    f'DEBUG: Raw Tuned Attr Scores (first 10): {tuned_model_attribution[:10]}'
                )
                print(
                    'Tuned model:',
                    visualize_attribution(
                        tuned_model_attribution, max_attr=max_tuned_attr, method=method,
                    ),
                    tuned_model_completion,
                )
            else:
                print(
                    f'Tuned model: [Attribution not calculated] {tuned_model_completion}'
                )

            print('-' * 60)   # Separator between examples
    else:
        print('\nNo harmful examples found or requested to process.')

    print('\nProcessing finished.')