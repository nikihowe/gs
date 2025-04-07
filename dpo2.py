import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
# import matplotlib.pyplot as plt # Removed as not currently used
from tqdm import tqdm
import math
from torch.utils.data import DataLoader
from functools import partial
from typing import Dict, List, Union

# Assuming these utils exist and function correctly
# get_the_datasets should return HF Dataset objects with 'prompt', 'chosen', 'rejected' text columns
from dataset_utils2 import get_the_datasets # Removed dpoify_dataset, filter_too_long as they are internal to get_the_datasets

# --- Hyperparameters ---
EPOCHS = 1  # Start with 1 epoch, see if we need more
MINIBATCH_SIZE = 4   # Per device batch size (DataLoader batch size)
BATCH_SIZE = 32      # Effective batch size after gradient accumulation
ACCUMULATION_STEPS = BATCH_SIZE // MINIBATCH_SIZE
# BATCHES_PER_EPOCH = 8 # <<<--- REMOVED: Determined by DataLoader
LEARNING_RATE = 2e-5
BETA = 0.1
EVAL_STEPS = 100     # Evaluate validation loss every N optimizer steps
MAX_EVAL_SAMPLES = 300
MAX_LENGTH = 512     # Define a max sequence length for padding/truncation
NUM_WORKERS = 4      # Number of workers for DataLoader (adjust based on system)
LOG_INTERVAL = 10     # <<<--- MODIFIED: Log accumulated train loss every N optimizer steps

# --- Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load the model and tokenizer
# checkpoint = 'HuggingFaceTB/SmolLM2-135M-Instruct'
# checkpoint = 'HuggingFaceTB/SmolLM2-360M-Instruct'
checkpoint = 'HuggingFaceTB/SmolLM2-1.7B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token # Common practice if no pad token
    print(f"Tokenizer has no pad token, setting it to EOS token: {tokenizer.pad_token}")
print(f'Tokenizer vocab size: {tokenizer.vocab_size}')
print(f'Using pad token ID: {tokenizer.pad_token_id}')
print(f'Padding side: {tokenizer.padding_side}')

model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# Make and freeze a reference model
ref_model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False
print("Reference model parameters frozen.")

# --- Collate Function ---
# Moved directly into the script
def dpo_collate_fn(batch: List[Dict[str, str]], tokenizer: PreTrainedTokenizer, max_length: int) -> Dict[str, torch.Tensor]:
    """
    Collate function to prepare DPO batches.
    Tokenizes prompts, chosen, and rejected responses separately,
    pads them dynamically (longest) within the batch.
    """
    # Ensure required keys are present
    if not batch or not all(k in batch[0] for k in ['prompt', 'chosen', 'rejected']):
        raise ValueError("Batch items must contain 'prompt', 'chosen', and 'rejected' keys.")

    prompts = [item['prompt'] for item in batch]
    chosen_responses = [item['chosen'] for item in batch]
    rejected_responses = [item['rejected'] for item in batch]

    # Tokenize each part separately, padding to longest in batch
    # Important: Ensure tokenizer handles padding side correctly (usually right for Causal LMs)
    prompt_tok = tokenizer(prompts, max_length=max_length, truncation=True, padding='longest', return_tensors='pt')
    chosen_tok = tokenizer(chosen_responses, max_length=max_length, truncation=True, padding='longest', return_tensors='pt')
    rejected_tok = tokenizer(rejected_responses, max_length=max_length, truncation=True, padding='longest', return_tensors='pt')

    return {
        'prompt_input_ids': prompt_tok.input_ids,
        'prompt_attention_mask': prompt_tok.attention_mask,
        'chosen_input_ids': chosen_tok.input_ids,
        'chosen_attention_mask': chosen_tok.attention_mask,
        'rejected_input_ids': rejected_tok.input_ids,
        'rejected_attention_mask': rejected_tok.attention_mask,
    }


# --- Dataset Preparation ---
print("Loading datasets...")
# Assuming get_the_datasets returns Hugging Face Dataset objects
# with 'prompt', 'chosen', 'rejected' text columns.
# It should handle loading from disk / processing / filtering internally.
hf_train_dataset = get_the_datasets(tokenizer, max_length=MAX_LENGTH)
hf_test_dataset = get_the_datasets(tokenizer, max_length=MAX_LENGTH, test=True)
print("Datasets loaded.")

# --- Create DataLoaders ---
collate_wrapper = partial(dpo_collate_fn, tokenizer=tokenizer, max_length=MAX_LENGTH)

train_dataloader = DataLoader(
    hf_train_dataset, # Use HF dataset directly
    batch_size=MINIBATCH_SIZE,
    shuffle=True,
    collate_fn=collate_wrapper,
    num_workers=NUM_WORKERS,
    pin_memory=True # Usually good practice with GPUs
)
val_dataloader = DataLoader(
    hf_test_dataset, # Use HF dataset directly
    batch_size=MINIBATCH_SIZE, # Use same per-device size for validation
    shuffle=False,
    collate_fn=collate_wrapper,
    num_workers=NUM_WORKERS,
    pin_memory=True
)
print(f"Train DataLoader: {len(train_dataloader)} batches of size {MINIBATCH_SIZE}")
print(f"Validation DataLoader: {len(val_dataloader)} batches of size {MINIBATCH_SIZE}")


# --- Loss Calculation Functions ---
# (Assuming extract_log_probs and dpo_loss_function are defined as in previous correct versions)
# Make sure extract_log_probs uses prompt_attention_mask to find prompt length correctly
def extract_log_probs(
    logits: torch.Tensor,      # Shape: (batch_size, sequence_length, vocab_size)
    labels: torch.Tensor,      # Shape: (batch_size, sequence_length)
    prompt_mask: torch.Tensor, # Shape: (batch_size, sequence_length) - mask TRUE for prompt tokens
    completion_mask: torch.Tensor # Shape: (batch_size, sequence_length) - mask TRUE for completion tokens
) -> torch.Tensor:
    """
    Calculates the average log probability of the completion tokens.
    Assumes labels includes prompt + completion.
    Assumes prompt_mask and completion_mask are for the *original* sequence length.
    """
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    batch_size, seq_len_shifted, vocab_size = shifted_logits.shape

    log_probs_all = F.log_softmax(shifted_logits, dim=-1)
    gathered_log_probs = torch.gather(log_probs_all, 2, shifted_labels.unsqueeze(-1)).squeeze(-1)

    # Completion mask for the *shifted* sequence (predicting token k using index k-1)
    completion_mask_orig_shifted = completion_mask[:, 1:] # Align completion mask with shifted seq
    prompt_mask_shifted = prompt_mask[:,:-1] # Align prompt mask with shifted seq

    # Valid completion indices in shifted seq: not part of prompt AND part of original completion
    valid_completion_indices_shifted = completion_mask_orig_shifted & (~prompt_mask_shifted)

    # Mask should be true only for valid completion tokens in the shifted sequence
    masked_log_probs = gathered_log_probs * valid_completion_indices_shifted # Element-wise mul

    sum_log_probs_B = masked_log_probs.sum(dim=-1)
    num_completion_tokens_B = valid_completion_indices_shifted.sum(dim=-1).float()

    mean_log_probs_B = torch.where(
        num_completion_tokens_B > 0,
        sum_log_probs_B / num_completion_tokens_B,
        torch.zeros_like(sum_log_probs_B)
    )

    if torch.isnan(mean_log_probs_B).any() or torch.isinf(mean_log_probs_B).any():
        print("Warning: NaN or Inf detected in mean_log_probs_B!")
        # Consider adding more debug info here if needed

    assert mean_log_probs_B.shape == (batch_size,) , f"Expected shape ({batch_size},) but got {mean_log_probs_B.shape}"
    return mean_log_probs_B


def dpo_loss_function(
    policy_chosen_logprobs: torch.Tensor, # Shape: (B,)
    policy_rejected_logprobs: torch.Tensor, # Shape: (B,)
    ref_chosen_logprobs: torch.Tensor, # Shape: (B,)
    ref_rejected_logprobs: torch.Tensor, # Shape: (B,)
    beta: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculates the DPO loss."""
    pi_logratios = policy_chosen_logprobs - policy_rejected_logprobs
    ref_logratios = ref_chosen_logprobs - ref_rejected_logprobs

    logits = pi_logratios - ref_logratios
    # The loss is the negative log-likelihood of the policy accurately classifying the chosen answer as better
    # Uses the Bradley-Terry model probability P(chosen > rejected) = sigmoid(beta * (log_pi(chosen)/ref(chosen) - log_pi(rejected)/ref(rejected)))
    loss = -F.logsigmoid(beta * logits).mean() # Average loss over the batch

    # Calculate rewards for logging purposes (detached from the graph)
    chosen_rewards = beta * (policy_chosen_logprobs - ref_chosen_logprobs).detach()
    rejected_rewards = beta * (policy_rejected_logprobs - ref_rejected_logprobs).detach()

    return loss, chosen_rewards, rejected_rewards


def get_batch_loss(
    batch: dict[str, torch.Tensor], # Batch directly from DataLoader/collate_fn
    model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
    beta: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the DPO loss and associated rewards for a given batch."""

    # --- 1. Concatenate prompt and completions ---
    # Assumes inputs are already padded correctly by collate_fn
    # Handle potential variations in max length due to dynamic padding
    max_prompt_len = batch['prompt_input_ids'].size(1)
    max_chosen_len = batch['chosen_input_ids'].size(1)
    max_rejected_len = batch['rejected_input_ids'].size(1)

    concat_chosen_ids = torch.cat(
        [batch['prompt_input_ids'], batch['chosen_input_ids']], dim=-1
    )
    concat_chosen_mask = torch.cat(
        [batch['prompt_attention_mask'], batch['chosen_attention_mask']], dim=-1
    )
    concat_rejected_ids = torch.cat(
        [batch['prompt_input_ids'], batch['rejected_input_ids']], dim=-1
    )
    concat_rejected_mask = torch.cat(
        [batch['prompt_attention_mask'], batch['rejected_attention_mask']], dim=-1
    )

    # --- Create masks needed for log prob extraction ---
    # These masks identify which tokens belong to the prompt and completion *within the concatenated sequence*
    prompt_only_mask_chosen = torch.cat(
        [batch['prompt_attention_mask'], torch.zeros_like(batch['chosen_attention_mask'])], dim=-1
    )
    prompt_only_mask_rejected = torch.cat(
        [batch['prompt_attention_mask'], torch.zeros_like(batch['rejected_attention_mask'])], dim=-1
    )
    completion_only_mask_chosen = torch.cat(
        [torch.zeros_like(batch['prompt_attention_mask']), batch['chosen_attention_mask']], dim=-1
    )
    completion_only_mask_rejected = torch.cat(
        [torch.zeros_like(batch['prompt_attention_mask']), batch['rejected_attention_mask']], dim=-1
    )

    # --- 2. Get Model Outputs ---
    # Policy model (requires gradients)
    policy_chosen_outputs = model(
        concat_chosen_ids, attention_mask=concat_chosen_mask, return_dict=True
    )
    policy_rejected_outputs = model(
        concat_rejected_ids, attention_mask=concat_rejected_mask, return_dict=True
    )
    # Reference model (no gradients needed)
    with torch.no_grad():
        ref_chosen_outputs = ref_model(
            concat_chosen_ids, attention_mask=concat_chosen_mask, return_dict=True
        )
        ref_rejected_outputs = ref_model(
            concat_rejected_ids, attention_mask=concat_rejected_mask, return_dict=True
        )

    # --- 3. Extract Log Probabilities for Completions ---
    # Uses the model logits, the concatenated IDs (as labels), and the specific masks
    policy_chosen_logprobs = extract_log_probs(
        policy_chosen_outputs.logits, concat_chosen_ids, prompt_only_mask_chosen, completion_only_mask_chosen
    )
    policy_rejected_logprobs = extract_log_probs(
        policy_rejected_outputs.logits, concat_rejected_ids, prompt_only_mask_rejected, completion_only_mask_rejected
    )
    # Use the same masks for the reference model outputs
    ref_chosen_logprobs = extract_log_probs(
        ref_chosen_outputs.logits, concat_chosen_ids, prompt_only_mask_chosen, completion_only_mask_chosen
    )
    ref_rejected_logprobs = extract_log_probs(
        ref_rejected_outputs.logits, concat_rejected_ids, prompt_only_mask_rejected, completion_only_mask_rejected
    )

    # --- 4. Compute DPO Loss ---
    loss, chosen_rewards, rejected_rewards = dpo_loss_function(
        policy_chosen_logprobs,
        policy_rejected_logprobs,
        ref_chosen_logprobs,
        ref_rejected_logprobs,
        beta=beta
    )

    return loss, chosen_rewards, rejected_rewards


# --- DPO Training Loop ---
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
optimizer.zero_grad() # Zero gradients initially

global_step = 0 # Tracks optimizer steps
total_steps = math.ceil(len(train_dataloader) / ACCUMULATION_STEPS) * EPOCHS
print(f"Total optimizer steps: {total_steps}")
print(f"Logging training stats every {LOG_INTERVAL} optimizer steps.")
print(f"Running validation every {EVAL_STEPS} optimizer steps.")

print("Starting training...")
for epoch in range(EPOCHS):
    print(f"--- Epoch {epoch+1}/{EPOCHS} ---")
    model.train() # Ensure model is in training mode

    # Calculate steps per epoch based on dataloader length
    num_batches_per_epoch = len(train_dataloader)
    num_optimizer_steps_per_epoch = math.ceil(num_batches_per_epoch / ACCUMULATION_STEPS)
    progress_bar = tqdm(total=num_optimizer_steps_per_epoch, desc=f"Epoch {epoch+1} Training", leave=True)

    microbatch_step = 0 # Tracks microbatches processed within an accumulation cycle

    # Accumulators for logging interval (reset every log_interval steps)
    accumulated_loss = 0.0
    accumulated_chosen_rewards = 0.0
    accumulated_rejected_rewards = 0.0
    accumulated_accurate_samples = 0 # Count samples where chosen > rejected reward
    accumulated_samples = 0

    for batch in train_dataloader:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        current_microbatch_size = batch['prompt_input_ids'].size(0) # Actual size for this microbatch

        # --- Forward Pass ---
        # No need for torch.no_grad() here as we need gradients for the policy model
        loss, chosen_rewards, rejected_rewards = get_batch_loss(
            batch, model, ref_model, BETA
        )

        # Check for NaN loss BEFORE backward pass
        if torch.isnan(loss):
            print(f"WARNING: NaN loss detected at global step {global_step}, epoch {epoch+1}. Skipping batch.")
            # Crucially, clear any potentially corrupted gradients from previous microbatches
            # if an optimizer step hasn't happened yet in this accumulation cycle.
            # If `microbatch_step % ACCUMULATION_STEPS != 0`, gradients might exist.
            # Safest is to zero grad here IF you skip the optimizer step,
            # but since we scale loss *before* backward, NaNs shouldn't propagate widely
            # unless the forward pass itself yields NaNs frequently.
            # Just continuing should be okay, as the optimizer step won't happen for this batch.
            continue # Skip backward and optimizer step for this microbatch

        # Scale loss for gradient accumulation
        # Loss is averaged over the microbatch in get_batch_loss,
        # so we scale it down here before backprop.
        scaled_loss = loss / ACCUMULATION_STEPS

        # Accumulate metrics for logging (use the *unscaled* loss for interpretable average)
        # Weight by the number of samples in the microbatch
        accumulated_loss += loss.item() * current_microbatch_size
        accumulated_chosen_rewards += chosen_rewards.sum().item() # Sum rewards over the microbatch
        accumulated_rejected_rewards += rejected_rewards.sum().item()
        # Count how many samples in the microbatch had chosen reward > rejected reward
        accumulated_accurate_samples += (chosen_rewards > rejected_rewards).sum().item()
        accumulated_samples += current_microbatch_size

        # --- Backward Pass ---
        scaled_loss.backward()
        microbatch_step += 1 # Increment after processing a microbatch

        # --- Optimizer Step ---
        # Check if we have processed enough microbatches for one optimizer step
        if microbatch_step % ACCUMULATION_STEPS == 0:
            # Optional: Gradient clipping (prevents exploding gradients)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()    # Update model weights
            optimizer.zero_grad() # Clear gradients for the next accumulation cycle
            global_step += 1    # Increment the global optimizer step counter
            progress_bar.update(1) # Update the training progress bar for the optimizer step

            # --- Log Training Statistics --- (triggered by global_step and log_interval)
            if global_step > 0 and global_step % LOG_INTERVAL == 0:
                if accumulated_samples > 0:
                    avg_loss = accumulated_loss / accumulated_samples
                    avg_chosen_reward = accumulated_chosen_rewards / accumulated_samples
                    avg_rejected_reward = accumulated_rejected_rewards / accumulated_samples
                    reward_acc = accumulated_accurate_samples / accumulated_samples

                    print(f"\n[TRAIN] Step: {global_step}, Avg Loss: {avg_loss:.4f}, "
                          f"Avg Chosen Reward: {avg_chosen_reward:.3f}, "
                          f"Avg Rejected Reward: {avg_rejected_reward:.3f}, "
                          f"Reward Acc: {reward_acc:.3f}")

                    # Update progress bar postfix (optional)
                    progress_bar.set_postfix({
                        "Loss": f"{avg_loss:.4f}",
                        "Acc": f"{reward_acc:.2f}"
                    })
                else:
                     print(f"\n[TRAIN] Step: {global_step}, No samples processed in the last {LOG_INTERVAL} steps (check accumulation/batching).")


                # Clear accumulators for the next logging interval
                accumulated_loss = 0.0
                accumulated_chosen_rewards = 0.0
                accumulated_rejected_rewards = 0.0
                accumulated_accurate_samples = 0
                accumulated_samples = 0

            # --- Validation Step --- (triggered by global_step and eval_steps)
            # <<<--- MODIFIED VALIDATION SECTION (SUBSET EVAL) --->>>
            if global_step > 0 and global_step % EVAL_STEPS == 0:
                print(f"\n--- Running Validation on approx. {MAX_EVAL_SAMPLES} samples at Step {global_step} ---")
                model.eval() # Switch to evaluation mode

                total_eval_loss = 0.0
                total_eval_chosen_rewards = 0.0
                total_eval_rejected_rewards = 0.0
                total_eval_accurate_samples = 0
                total_eval_samples = 0 # Tracks samples processed *in this validation run*

                # Note: val_progress_bar will show total batches for the *full* dataset,
                # but we will break early. leave=False makes it disappear afterwards.
                val_progress_bar = tqdm(val_dataloader, desc=f"Validation (subset ~{MAX_EVAL_SAMPLES})", leave=False)
                with torch.no_grad(): # Ensure no gradients are computed during validation
                    for val_batch in val_progress_bar:
                        # Check if we've processed enough samples already
                        if total_eval_samples >= MAX_EVAL_SAMPLES:
                            break # Stop evaluating early

                        # Move batch to device
                        val_batch = {k: v.to(device) for k, v in val_batch.items()}
                        val_microbatch_size = val_batch['prompt_input_ids'].size(0)

                        # Get loss and rewards for the validation batch
                        eval_loss, eval_chosen_rewards, eval_rejected_rewards = get_batch_loss(
                            val_batch, model, ref_model, BETA
                        )

                        if not torch.isnan(eval_loss): # Skip NaN eval losses
                            # Accumulate metrics only if loss is valid
                            total_eval_loss += eval_loss.item() * val_microbatch_size
                            total_eval_chosen_rewards += eval_chosen_rewards.sum().item()
                            total_eval_rejected_rewards += eval_rejected_rewards.sum().item()
                            total_eval_accurate_samples += (eval_chosen_rewards > eval_rejected_rewards).sum().item()
                            total_eval_samples += val_microbatch_size # Increment count *after* successful processing
                        else:
                            print("Warning: NaN detected in validation loss for a batch.")
                            # Optionally decide if you want to count these samples or not.
                            # Current logic doesn't count them in total_eval_samples if loss is NaN.

                # --- Calculation and Reporting (runs after the loop, possibly broken early) ---
                if total_eval_samples > 0:
                    avg_eval_loss = total_eval_loss / total_eval_samples
                    avg_eval_chosen_reward = total_eval_chosen_rewards / total_eval_samples
                    avg_eval_rejected_reward = total_eval_rejected_rewards / total_eval_samples
                    avg_eval_reward_acc = total_eval_accurate_samples / total_eval_samples

                    print(f"--- Validation Complete (on {total_eval_samples} samples) ---")
                    print(f"[VAL] Step: {global_step}, Avg Loss: {avg_eval_loss:.4f}, "
                          f"Avg Chosen Reward: {avg_eval_chosen_reward:.3f}, "
                          f"Avg Rejected Reward: {avg_eval_rejected_reward:.3f}, "
                          f"Reward Acc: {avg_eval_reward_acc:.3f}")
                    # Add early stopping logic here based on these subset metrics if desired
                else:
                    # This case now means either the val_dataloader was empty,
                    # MAX_EVAL_SAMPLES was 0, or all processed batches resulted in NaN loss.
                    print(f"--- Validation Warning: No valid samples processed (processed {total_eval_samples} samples) ---")

                model.train() # Set back to training mode before continuing training loop
            # <<<--- END OF MODIFIED VALIDATION SECTION --->>>

    # --- End of Epoch ---
    progress_bar.close()
    print(f"--- Epoch {epoch+1} Complete ---")
    # Optional: Run validation at the end of each epoch regardless of EVAL_STEPS
    # (You would duplicate or call a validation function here)


# --- Final Check & Save ---
print("Training finished.")

# Check that the reference model has not changed (sanity check)
print("Verifying reference model parameters...")
try:
    # Load fresh reference model for comparison
    ref_ref_model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    match = True
    # Comparing parameters requires them to be on the same device implicitly
    for name, param in ref_model.named_parameters():
        # Ensure the parameter exists in the freshly loaded model
        try:
           ref_param = ref_ref_model.get_parameter(name)
        except AttributeError:
           print(f"Parameter {name} not found in the freshly loaded reference model.")
           match = False
           break

        if not torch.allclose(param.cpu(), ref_param.cpu()): # Compare on CPU to avoid potential minor GPU differences
            print(f"Mismatch found in parameter: {name}")
            match = False
            break
    if match:
        print("Reference model parameters verified unchanged.")
    else:
        print("WARNING: Reference model parameters seem to have changed!")
    del ref_ref_model # Free memory
except Exception as e:
    print(f"Could not verify reference model parameters: {e}")


# Save the trained model
output_dir = 'big_dpo_model_revised_dl_stats' # Changed name slightly
print(f"Saving model to {output_dir}...")
try:
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print('Model saved!')
except Exception as e:
    print(f"Error saving model: {e}")

print('done!')