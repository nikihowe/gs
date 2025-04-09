import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm # Import tqdm for progress bars

## LLM-GENERATED CODE START ##

# --- Configuration (Adjust these paths as needed) ---
checkpoint = 'HuggingFaceTB/SmolLM2-1.7B-Instruct'
trained_model_path = 'large_long_dpo_final_model'
dataset_path = './datasets/dataset.json'
output_dir = './large_evaluation_outputs' # Directory to save output files

gen_max_length = 150
DEBUG = False # Set to True for more verbose debugging output if needed

# --- Set Device ---
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA compute capability: {torch.cuda.get_device_capability(0)}")
    if torch.cuda.get_device_capability(0)[0] >= 8:
        print("Enabling Flash Attention 2 (requires Ampere or newer).")
        # Note: Eager attention is specified later, you might need to adjust
        # attn_implementation based on your actual hardware and needs.
        # If using flash attention, change 'eager' below.
else:
    device = torch.device('cpu')
    print("Using CPU. Processing will be slower.")

# --- Create Output Directory ---
os.makedirs(output_dir, exist_ok=True)
print(f"Output files will be saved in: {output_dir}")

if __name__ == '__main__':
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
            attn_implementation='eager', # Keep or change based on hardware/needs
            torch_dtype=torch.float32,
        ).to(device)
        print(f'Loading tuned model from: {trained_model_path}')
        if not os.path.isdir(trained_model_path):
            raise FileNotFoundError(
                f'Tuned model directory not found: {trained_model_path}'
            )
        trained_model = AutoModelForCausalLM.from_pretrained(
            trained_model_path,
            trust_remote_code=True,
            attn_implementation='eager', # Keep or change based on hardware/needs
            torch_dtype=torch.float32,
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
            # Add a pad token if none exists
            print("Adding new pad token '[PAD]'")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # Resize embeddings for both models
            base_model.resize_token_embeddings(len(tokenizer))
            trained_model.resize_token_embeddings(len(tokenizer))
            print("Resized model token embeddings.")

    base_model.config.pad_token_id = tokenizer.pad_token_id
    trained_model.config.pad_token_id = tokenizer.pad_token_id
    print(f"Pad token ID set to: {tokenizer.pad_token_id}")

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

    # Validate dataset structure (simplified check)
    if not isinstance(dataset, dict) or 'text' not in dataset or 'harmful' not in dataset:
        print("Error: Dataset JSON must be an object with 'text' and 'harmful' keys.")
        exit(1)
    if len(dataset['text']) != len(dataset['harmful']):
         print("Error: Dataset 'text' and 'harmful' lists must have the same length.")
         exit(1)

    print(f"Loaded dataset with {len(dataset['text'])} examples.")

    # --- Prepare Examples and Prompts ---
    all_prompts = []
    all_labels = []
    for text, label in zip(dataset['text'], dataset['harmful']):
         # Ensure label is valid (0 or 1)
         if isinstance(label, (int, float)) and label in [0, 1]:
             prompt = 'Human: ' + str(text) + ' Assistant: '
             all_prompts.append(prompt)
             all_labels.append(int(label))
         else:
             if DEBUG:
                 print(f"Warning: Skipping example with invalid label: {label} for text: {str(text)[:50]}...")

    num_total_examples = len(all_prompts)
    num_good_examples = sum(1 for label in all_labels if label == 0)
    num_bad_examples = sum(1 for label in all_labels if label == 1)
    print(f"Prepared {num_total_examples} valid examples ({num_good_examples} harmless, {num_bad_examples} harmful).")

    # --- Initialize Result Storage ---
    good_base_results = []
    good_trained_results = []
    bad_base_results = []
    bad_trained_results = []

    # --- Shared Generation Arguments ---
    # Using greedy decoding as in the original example
    gen_args = {
        'num_return_sequences': 1,
        'eos_token_id': tokenizer.eos_token_id,
        'pad_token_id': tokenizer.pad_token_id,
        'do_sample': False,
        'no_repeat_ngram_size': 3,
        # max_length is set dynamically below
    }

    # === Process ALL Examples ===
    print(f'\n=== Processing {num_total_examples} Examples ===')
    print('=' * 60)

    # Use tqdm for a progress bar
    for i in tqdm(range(num_total_examples), desc="Generating Responses"):
        current_prompt = all_prompts[i]
        current_label = all_labels[i] # 0 for harmless, 1 for harmful

        if DEBUG and (i % 50 == 0): # Print occasional status
             print(f"\nProcessing example {i+1}/{num_total_examples} (Label: {current_label})")
             print(f"Prompt starts: {current_prompt[:100]}...")

        base_model_completion = '[GENERATION ERROR]' # Default in case of error
        tuned_model_completion = '[GENERATION ERROR]'

        try:
            # Tokenize prompt, handling potential truncation
            inputs_tokenized = tokenizer(
                current_prompt,
                return_tensors='pt',
                truncation=True,
                # Ensure prompt doesn't exceed model capacity minus generation length
                max_length=tokenizer.model_max_length - gen_max_length if tokenizer.model_max_length else 2048 - gen_max_length,
            ).to(device)

            current_input_len = inputs_tokenized.input_ids.shape[1]
            dynamic_gen_args = gen_args.copy()
            dynamic_gen_args['max_length'] = current_input_len + gen_max_length

            # --- Generate with Base Model ---
            with torch.no_grad():
                base_model_output = base_model.generate(
                    **inputs_tokenized,
                    **dynamic_gen_args,
                )
            # Extract only the generated tokens (after the prompt)
            base_model_generated_ids = base_model_output[0, current_input_len:]
            base_model_completion = tokenizer.decode(
                base_model_generated_ids, skip_special_tokens=True
            ).strip()

            # --- Generate with Tuned Model ---
            with torch.no_grad():
                tuned_model_output = trained_model.generate(
                    **inputs_tokenized,
                    **dynamic_gen_args,
                )
            # Extract only the generated tokens
            tuned_model_generated_ids = tuned_model_output[0, current_input_len:]
            tuned_model_completion = tokenizer.decode(
                tuned_model_generated_ids, skip_special_tokens=True
            ).strip()

            if DEBUG and (i % 50 == 0):
                 print(f"Base generated: {base_model_completion[:100]}...")
                 print(f"Tuned generated: {tuned_model_completion[:100]}...")


        except Exception as e:
            print(f'\nError during generation for example {i+1} (Label: {current_label}): {e}')
            # Keep the default '[GENERATION ERROR]' placeholders

        # --- Store Results based on Label ---
        result_pair = (current_prompt, base_model_completion)
        if current_label == 0: # Harmless
            good_base_results.append(result_pair)
            good_trained_results.append((current_prompt, tuned_model_completion))
        else: # Harmful
            bad_base_results.append(result_pair)
            bad_trained_results.append((current_prompt, tuned_model_completion))

    print('\nGeneration complete.')
    print('=' * 60)

    # === Save Results to Files ===
    print('Saving results to text files...')

    def save_results_to_file(filename, results_list):
        filepath = os.path.join(output_dir, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for prompt, response in results_list:
                    f.write(f"PROMPT:\n{prompt}\n")
                    f.write(f"RESPONSE:\n{response}\n")
                    f.write("=" * 80 + "\n") # Separator
            print(f"Successfully saved {len(results_list)} entries to {filepath}")
        except Exception as e:
            print(f"Error saving file {filepath}: {e}")

    # Define filenames
    file_good_base = 'good_base.txt'
    file_good_trained = 'good_trained.txt'
    file_bad_base = 'bad_base.txt'
    file_bad_trained = 'bad_trained.txt'

    # Save each file
    save_results_to_file(file_good_base, good_base_results)
    save_results_to_file(file_good_trained, good_trained_results)
    save_results_to_file(file_bad_base, bad_base_results)
    save_results_to_file(file_bad_trained, bad_trained_results)

    print('\nProcessing and saving finished.')

## LLM-GENERATED CODE END ##