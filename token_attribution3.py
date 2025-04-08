import torch
from colorama import Fore, Style
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch.nn.functional as F
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')


def token_attribution(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    completion: str,
) -> list[tuple[str, float]]:
    model.eval()

    # --- 1. Tokenization and Index Identification ---
    full_text = prompt + completion
    # --- FIX: Use rstrip() for boundary calculation ---
    prompt_content_len = len(
        prompt.rstrip()
    )   # Length excluding trailing whitespace

    print(f'\n--- DEBUG: Inside token_attribution ---')
    print(f"DEBUG: Prompt (first 80 chars): '{prompt[:80]}...'")
    print(f"DEBUG: Completion: '{completion}'")
    print(f'DEBUG: Original Prompt Char Length: {len(prompt)}')
    print(
        f'DEBUG: Stripped Prompt Boundary Length (for check): {prompt_content_len}'
    )

    try:
        inputs = tokenizer(
            full_text,
            return_tensors='pt',
            truncation=True,
            max_length=getattr(model.config, 'max_position_embeddings', 512),
            return_offsets_mapping=True,
        ).to(device)
    except Exception as e:
        print(f'Error during tokenization: {e}')
        return []

    input_ids = inputs['input_ids'][0]
    offsets = inputs['offset_mapping'][0].tolist()
    sequence_length = len(input_ids)

    print(f'DEBUG: Sequence Length (tokens): {sequence_length}')
    # (Optional: Keep offset debug prints if needed)
    # print(f"DEBUG: Offsets (first 10): {offsets[:10]}")
    # print(f"DEBUG: Offsets (last 10): {offsets[-10:]}")

    prompt_start_idx = 0
    if (
        getattr(tokenizer, 'add_bos_token', False)
        and input_ids[0] == tokenizer.bos_token_id
    ):
        prompt_start_idx = 1
        # (Optional: Keep BOS debug print)
        # print(f"DEBUG: BOS token detected, prompt_start_idx = 1")

    completion_token_start_index = -1
    for idx, (start_char, end_char) in enumerate(offsets):
        if start_char == 0 and end_char == 0:
            if idx == 0 and prompt_start_idx == 1:
                continue
            else:
                continue

        # --- FIX: Compare against stripped length ---
        if start_char >= prompt_content_len:
            completion_token_start_index = idx
            print(
                f'DEBUG: Found completion start at index {idx} using boundary {prompt_content_len}, offset ({start_char}, {end_char})'
            )
            break

    if completion_token_start_index == -1:
        completion_token_start_index = sequence_length
        print(
            f'DEBUG: Completion start not found via offset, setting to end: {sequence_length}'
        )

    prompt_end_idx = completion_token_start_index
    print(
        f'DEBUG: Final prompt indices range: [{prompt_start_idx}:{prompt_end_idx}]'
    )
    print(
        f'DEBUG: Final completion indices range: [{completion_token_start_index}:{sequence_length}]'
    )

    prompt_indices = range(prompt_start_idx, prompt_end_idx)
    completion_indices = range(completion_token_start_index, sequence_length)

    # --- Decode for Debugging (Keep this block) ---
    if sequence_length > 0:
        actual_prompt_token_ids = input_ids[prompt_indices]
        actual_prompt_tokens_decoded = tokenizer.convert_ids_to_tokens(
            actual_prompt_token_ids
        )
        print(
            f'DEBUG: Tokens assigned to PROMPT: {actual_prompt_tokens_decoded}'
        )

        if completion_indices:
            actual_completion_token_ids = input_ids[completion_indices]
            actual_completion_tokens_decoded = tokenizer.convert_ids_to_tokens(
                actual_completion_token_ids
            )
            print(
                f'DEBUG: Tokens assigned to COMPLETION: {actual_completion_tokens_decoded}'
            )
        else:
            print('DEBUG: Tokens assigned to COMPLETION: [] (Empty Range)')
    else:
        print('DEBUG: Sequence length is 0, cannot decode tokens.')

    # --- Sanity Checks (Keep these, but ensure prompt_tokens list is defined if needed) ---
    if not (
        0
        <= prompt_start_idx
        <= prompt_end_idx
        <= completion_token_start_index
        <= sequence_length
    ):
        print(f'Error: Invalid index calculation. Skipping.')
        print(f'--- END DEBUG (Error): Exiting token_attribution ---\n')
        return []
    # Check if prompt range is valid *before* trying to decode (actual_prompt_tokens_decoded handles empty case now)
    if prompt_start_idx >= prompt_end_idx:
        print('Warning: Prompt has zero tokens.')
        print(f'--- END DEBUG (Warning): Exiting token_attribution ---\n')
        return []
    if not completion_indices:
        print(
            'Warning: Completion has zero tokens. Returning zero attribution.'
        )
        prompt_tokens = (
            actual_prompt_tokens_decoded
            if 'actual_prompt_tokens_decoded' in locals()
            else []
        )
        if not prompt_tokens:
            return []
        print(f'--- END DEBUG (Warning): Exiting token_attribution ---\n')
        return [(token, 0.0) for token in prompt_tokens]

    # --- 2. Forward Pass, Attention Aggregation, Score Calculation ---
    attribution_scores_list = []   # Initialize return list
    try:
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            attentions = outputs.attentions

        if attentions is None:
            print('Error: Model did not return attention scores.')
            print(f'--- END DEBUG (Error): Exiting token_attribution ---\n')
            return []

        # --- 3. Attention Aggregation ---
        last_layer_attention = attentions[-1].to(device)
        avg_head_attention = last_layer_attention.mean(dim=1).squeeze(0)

        # --- 4. Calculate Attribution Scores ---
        # (Keep the inner try-except for indexing errors here)
        try:
            attribution_matrix = avg_head_attention[completion_indices, :][
                :, prompt_indices
            ]
            prompt_token_scores = attribution_matrix.sum(
                dim=0
            )   # Assign scores here
        except IndexError as e:
            print(f'Error indexing attention matrix: {e}')
            print(f'--- END DEBUG (Error): Exiting token_attribution ---\n')
            return []
        except Exception as e:
            print(f'Error during attribution calculation: {e}')
            print(f'--- END DEBUG (Error): Exiting token_attribution ---\n')
            return []

        # --- 5. Format Output [INSIDE TRY BLOCK] ---
        prompt_token_ids_final = input_ids[prompt_indices]
        prompt_tokens_final = tokenizer.convert_ids_to_tokens(
            prompt_token_ids_final
        )
        print(
            f'DEBUG: Final prompt tokens list being returned: {prompt_tokens_final}'
        )

        # --- FIX: NameError resolved as prompt_token_scores is now defined ---
        if len(prompt_tokens_final) != len(prompt_token_scores):
            print(
                f'Error: Mismatch final token count ({len(prompt_tokens_final)}) vs score count ({len(prompt_token_scores)}).'
            )
            min_len = min(len(prompt_tokens_final), len(prompt_token_scores))
            attribution_scores_list = list(
                zip(
                    prompt_tokens_final[:min_len],
                    prompt_token_scores.cpu().tolist()[:min_len],
                )
            )
        else:
            attribution_scores_list = list(
                zip(prompt_tokens_final, prompt_token_scores.cpu().tolist())
            )

    except Exception as e:   # Catch errors from forward pass itself or others in the block
        print(f'Error during model forward pass or subsequent processing: {e}')
        print(f'--- END DEBUG (Error): Exiting token_attribution ---\n')
        return (
            []
        )   # Return empty list if anything in the main try block failed

    print(f'--- END DEBUG: Exiting token_attribution ---\n')
    return attribution_scores_list


# --- Rest of your code (visualize_attribution, __main__) below ---


# --------------------------------------------------------------------------
# The rest of your code (visualize_attribution, create_attribution_display,
# and the __main__ block) should remain the same.
# Make sure the __main__ block calls this updated token_attribution function.
# --------------------------------------------------------------------------

import torch   # Ensure torch is imported if used for isfinite check
from colorama import Fore, Style, init as colorama_init

# Initialize colorama if not done globally
# colorama_init(autoreset=True)


def visualize_attribution(
    tokens_with_attributions: list[tuple[str, float]],
    max_attr: float,
) -> str:
    """
    Visualizes attribution scores using a 5-color scale.

    Args:
        tokens_with_attributions: List of (token_string, score) tuples.
        max_attr: The maximum attribution score for normalization.

    Returns:
        A string with tokens colored based on their normalized scores.
    """
    colored_text = ''
    # Use a small epsilon to avoid division by zero and handle very small max_attr
    if max_attr <= 1e-9:
        print(
            f'DEBUG: max_attr ({max_attr:.4f}) is very small or non-positive, using 1.0 for scaling.'
        )
        max_attr = 1.0

    # --- Define Thresholds for 5 Colors (Green < Cyan < Yellow < Magenta < Red) ---
    # IMPORTANT: Adjust these thresholds based on your raw score distribution!
    # These examples are tuned low based on previous debug output (max~50, others <1.0).
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
    # Scores <= 0.005 are Green

    # --- Alternatively, use evenly spaced thresholds if scores are distributed differently ---
    # threshold_cyan = 0.20
    # threshold_yellow = 0.40
    # threshold_magenta = 0.60
    # threshold_red = 0.80
    # ---

    # Debug print to show thresholds being used
    print(
        f'DEBUG: Visualizing with max_attr={max_attr:.4f}, Thresh: G<={threshold_cyan:.4f}<C<={threshold_yellow:.4f}<Y<={threshold_magenta:.4f}<M<={threshold_red:.4f}<R'
    )

    for token, attribution in tokens_with_attributions:
        # Ensure score is valid before processing
        if not isinstance(attribution, (float, int)) or not torch.isfinite(
            torch.tensor(attribution)
        ):
            attribution = (
                0.0  # Default to 0 for NaN, infinity, or non-numeric types
            )

        # Normalize score relative to max_attr (clamped between 0 and 1)
        norm_attribution = max(0.0, attribution) / max_attr
        norm_attribution = min(
            norm_attribution, 1.0
        )   # Clamp top end just in case

        # Determine color using the defined thresholds
        if norm_attribution > threshold_red:
            color = Fore.RED      # Highest importance
        elif norm_attribution > threshold_magenta:
            color = Fore.MAGENTA  # High-Medium importance
        elif norm_attribution > threshold_yellow:
            color = Fore.YELLOW   # Medium importance
        elif norm_attribution > threshold_cyan:
            color = Fore.CYAN     # Low-Medium importance
        else:
            color = Fore.GREEN    # Lowest importance

        # Handle token display string (handling spaces, bytes)
        display_token = token
        if isinstance(token, str) and token.startswith('Ġ'):
            display_token = ' ' + token[1:]
        elif isinstance(token, bytes):
            display_token = token.decode('utf-8', errors='replace')

        # Append colored token to the result string
        colored_text += f'{color}{display_token}{Style.RESET_ALL}'

    return colored_text


# ==============================================================================
# Main Execution Block (Modified)
# ==============================================================================
if __name__ == '__main__':
    colorama_init(autoreset=True)   # Initialize colorama

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
                        base_model_attribution, max_attr=max_base_attr
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
                        tuned_model_attribution, max_attr=max_tuned_attr
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
                        base_model_attribution, max_attr=max_base_attr
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
                        tuned_model_attribution, max_attr=max_tuned_attr
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
