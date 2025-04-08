import math
import os  # <<< Added for path operations
from functools import partial
from typing import Dict, List, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizer)

# Assuming these utils exist and function correctly
from dataset_utils2 import get_the_datasets

# TODO: add in seeds to make the code determinstic. This is particularly important
# when restarting from failure, as we want to be sure to shuffle the training dataset
# in the same way each time.
# TODO: add WandB logging

## LLM-GENERATED CODE START ##


torch.set_default_dtype(torch.float32)
# torch.autograd.set_detect_anomaly(True) # Enable for debugging NaNs, disable for performance/checkpointing runs

# --- Hyperparameters ---
EPOCHS = 1  # Adjust as needed for longer runs
MINIBATCH_SIZE = 2   # Per device batch size (DataLoader batch size)
BATCH_SIZE = 32       # Effective batch size after gradient accumulation
ACCUMULATION_STEPS = BATCH_SIZE // MINIBATCH_SIZE
LEARNING_RATE = 1e-6
BETA = 0.1
EVAL_STEPS = 100       # Evaluate validation loss every N optimizer steps
SAVE_INTERVAL = 20    # <Save checkpoint every N optimizer steps
MAX_EVAL_SAMPLES = 300
MAX_LENGTH = 512       # Define a max sequence length for padding/truncation
NUM_WORKERS = 4  # Number of workers for DataLoader (adjust based on system)
LOG_INTERVAL = 10      # Log accumulated train loss every N optimizer steps (adjusted for less noise)
CHECKPOINT_DIR = './large_dpo_checkpoints'  # Directory to save checkpoints
FINAL_MODEL_DIR = (
    './large_long_dpo_final_model'  # Directory for the final trained model
)
TRAIN_DATASET_SIZE = 40000

# --- Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Create checkpoint directory if it doesn't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f'Checkpoints will be saved in: {CHECKPOINT_DIR}')

# Load the model and tokenizer
# checkpoint = 'gpt2'
# checkpoint = 'HuggingFaceTB/SmolLM2-135M-Instruct'
# checkpoint = 'HuggingFaceTB/SmolLM2-360M-Instruct'
checkpoint = 'HuggingFaceTB/SmolLM2-1.7B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(
        f'Tokenizer has no pad token, setting it to EOS token: {tokenizer.pad_token}'
    )
print(f'Tokenizer vocab size: {tokenizer.vocab_size}')
print(f'Using pad token ID: {tokenizer.pad_token_id}')
print(f'Padding side: {tokenizer.padding_side}')

model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# Make and freeze a reference model
ref_model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False
print('Reference model parameters frozen.')

# --- Optimizer Definition ---
# Define optimizer here so we can load its state if resuming
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# --- Checkpoint Loading ---
start_epoch = 0
global_step = 0   # Initialize global_step here
latest_checkpoint_path = os.path.join(CHECKPOINT_DIR, 'latest_checkpoint.pth')

if os.path.exists(latest_checkpoint_path):
    print(f'Loading checkpoint from {latest_checkpoint_path}')
    try:
        checkpoint_data = torch.load(
            latest_checkpoint_path, map_location=device
        )
        model.load_state_dict(checkpoint_data['model_state_dict'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        start_epoch = checkpoint_data['epoch']
        global_step = checkpoint_data['global_step']
        # Ensure optimizer tensors are on the correct device (sometimes needed after loading)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print(
            f'Resumed training from Epoch {start_epoch}, Global Step {global_step}'
        )
    except Exception as e:
        print(f'Error loading checkpoint: {e}. Starting from scratch.')
        start_epoch = 0
        global_step = 0
else:
    print('No checkpoint found. Starting training from scratch.')


# --- Collate Function ---
def dpo_collate_fn(
    batch: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    if not batch or not all(
        k in batch[0] for k in ['prompt', 'chosen', 'rejected']
    ):
        raise ValueError(
            "Batch items must contain 'prompt', 'chosen', and 'rejected' keys."
        )

    prompts = [item['prompt'] for item in batch]
    chosen_responses = [item['chosen'] for item in batch]
    rejected_responses = [item['rejected'] for item in batch]

    prompt_tok = tokenizer(
        prompts,
        max_length=max_length,
        truncation=True,
        padding='longest',
        return_tensors='pt',
    )
    chosen_tok = tokenizer(
        chosen_responses,
        max_length=max_length,
        truncation=True,
        padding='longest',
        return_tensors='pt',
    )
    rejected_tok = tokenizer(
        rejected_responses,
        max_length=max_length,
        truncation=True,
        padding='longest',
        return_tensors='pt',
    )

    return {
        'prompt_input_ids': prompt_tok.input_ids,
        'prompt_attention_mask': prompt_tok.attention_mask,
        'chosen_input_ids': chosen_tok.input_ids,
        'chosen_attention_mask': chosen_tok.attention_mask,
        'rejected_input_ids': rejected_tok.input_ids,
        'rejected_attention_mask': rejected_tok.attention_mask,
    }


# --- Dataset Preparation ---
print('Loading datasets...')
hf_train_dataset_full = get_the_datasets(
    tokenizer,
    max_length=MAX_LENGTH,
    data_dir='.',
    processed_cache_base='/tmp',  # Keep processed data fast if desired
)
hf_test_dataset = get_the_datasets(
    tokenizer,
    max_length=MAX_LENGTH,
    test=True,
    data_dir='.',
    processed_cache_base='/tmp',
)
print('Full datasets loaded.')


# --- Subsetting Logic ---
train_subset_size = TRAIN_DATASET_SIZE
if len(hf_train_dataset_full) > train_subset_size:
    hf_train_dataset = hf_train_dataset_full.select(range(train_subset_size))
    print(f'Using a subset of {train_subset_size} training examples.')
else:
    hf_train_dataset = hf_train_dataset_full
    print(f'Using full training dataset ({len(hf_train_dataset)} examples).')

# --- Create DataLoaders ---
collate_wrapper = partial(
    dpo_collate_fn, tokenizer=tokenizer, max_length=MAX_LENGTH
)

train_dataloader = DataLoader(
    hf_train_dataset,
    batch_size=MINIBATCH_SIZE,
    shuffle=True,
    collate_fn=collate_wrapper,
    num_workers=NUM_WORKERS,
    pin_memory=True if device == 'cuda' else False,  # Only pin if using GPU
)
val_dataloader = DataLoader(
    hf_test_dataset,
    batch_size=MINIBATCH_SIZE,
    shuffle=False,
    collate_fn=collate_wrapper,
    num_workers=NUM_WORKERS,
    pin_memory=True if device == 'cuda' else False,
)
# Calculate total steps considering potential resumption
# Note: This might slightly overestimate if the last epoch wasn't fully completed before stopping
estimated_total_steps = (
    math.ceil(len(train_dataloader) / ACCUMULATION_STEPS) * EPOCHS
)
print(
    f'Train DataLoader: {len(train_dataloader)} batches of size {MINIBATCH_SIZE}'
)
print(
    f'Validation DataLoader: {len(val_dataloader)} batches of size {MINIBATCH_SIZE}'
)
print(
    f'Estimated total optimizer steps for {EPOCHS} epochs: {estimated_total_steps}'
)


# --- Loss Calculation Functions ---
def extract_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    completion_mask: torch.Tensor,
    current_global_step: int,  # Pass step for context in prints
) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    batch_size, seq_len_shifted, vocab_size = shifted_logits.shape

    if torch.isnan(shifted_logits).any() or torch.isinf(shifted_logits).any():
        print(
            f'!!! WARNING: NaN/Inf detected in shifted_logits BEFORE log_softmax at step {current_global_step} !!!'
        )

    log_probs_all = F.log_softmax(shifted_logits, dim=-1)
    gathered_log_probs = torch.gather(
        log_probs_all, 2, shifted_labels.unsqueeze(-1)
    ).squeeze(-1)

    valid_completion_indices_shifted = completion_mask[:, 1:]

    masked_log_probs = gathered_log_probs * valid_completion_indices_shifted
    sum_log_probs_B = masked_log_probs.sum(dim=-1)
    num_completion_tokens_B = valid_completion_indices_shifted.sum(
        dim=-1
    ).float()

    # Handle division by zero safely
    mean_log_probs_B = torch.where(
        num_completion_tokens_B > 0,
        sum_log_probs_B / num_completion_tokens_B,
        torch.zeros_like(sum_log_probs_B),
    )

    # NaN Check with Debugging Info
    if (
        torch.isnan(mean_log_probs_B).any()
        or torch.isinf(mean_log_probs_B).any()
    ):
        print(
            f'--- DEBUG: NaN or Inf detected in mean_log_probs_B at Step {current_global_step}! ---'
        )
        problem_indices = torch.where(
            torch.isnan(mean_log_probs_B) | torch.isinf(mean_log_probs_B)
        )[0]
        print(f'Problematic indices in batch: {problem_indices.tolist()}')
        for idx in problem_indices[
            : min(len(problem_indices), 3)
        ]:   # Print details for first few
            print(f' Index {idx}:')
            print(
                f'  num_completion_tokens: {num_completion_tokens_B[idx].item()}'
            )
            print(f'  sum_log_probs: {sum_log_probs_B[idx].item()}')

    assert mean_log_probs_B.shape == (
        batch_size,
    ), f'Expected shape ({batch_size},) but got {mean_log_probs_B.shape}'
    return mean_log_probs_B


def dpo_loss_function(
    policy_chosen_logprobs: torch.Tensor,
    policy_rejected_logprobs: torch.Tensor,
    ref_chosen_logprobs: torch.Tensor,
    ref_rejected_logprobs: torch.Tensor,
    beta: float,
    current_global_step: int,  # Pass step for context
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    # Check for NaNs in inputs early
    if any(
        torch.isnan(t).any()
        for t in [
            policy_chosen_logprobs,
            policy_rejected_logprobs,
            ref_chosen_logprobs,
            ref_rejected_logprobs,
        ]
    ):
        print(
            f'!!! WARNING: NaN detected in logprob inputs to DPO loss at step {current_global_step} !!!'
        )

    pi_logratios = policy_chosen_logprobs - policy_rejected_logprobs
    ref_logratios = ref_chosen_logprobs - ref_rejected_logprobs
    logits = pi_logratios - ref_logratios

    loss = -F.logsigmoid(beta * logits).mean()

    # Check loss value
    if torch.isnan(loss):
        print(
            f'!!! WARNING: DPO loss is NaN at step {current_global_step} !!!'
        )

    chosen_rewards = (
        beta * (policy_chosen_logprobs - ref_chosen_logprobs).detach()
    )
    rejected_rewards = (
        beta * (policy_rejected_logprobs - ref_rejected_logprobs).detach()
    )

    return loss, chosen_rewards, rejected_rewards


def get_batch_loss(
    batch: dict[str, torch.Tensor],
    model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
    beta: float,
    current_global_step: int,  # Pass step for context
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # --- 1. Concatenate prompt and completions ---
    concat_chosen_ids = torch.cat(
        [batch['prompt_input_ids'], batch['chosen_input_ids']], dim=-1
    )
    concat_chosen_mask = torch.cat(
        [batch['prompt_attention_mask'], batch['chosen_attention_mask']],
        dim=-1,
    )
    concat_rejected_ids = torch.cat(
        [batch['prompt_input_ids'], batch['rejected_input_ids']], dim=-1
    )
    concat_rejected_mask = torch.cat(
        [batch['prompt_attention_mask'], batch['rejected_attention_mask']],
        dim=-1,
    )

    # --- Create masks needed for log prob extraction ---
    prompt_only_mask_chosen = torch.cat(
        [
            batch['prompt_attention_mask'],
            torch.zeros_like(batch['chosen_attention_mask']),
        ],
        dim=-1,
    )
    prompt_only_mask_rejected = torch.cat(
        [
            batch['prompt_attention_mask'],
            torch.zeros_like(batch['rejected_attention_mask']),
        ],
        dim=-1,
    )
    completion_only_mask_chosen = torch.cat(
        [
            torch.zeros_like(batch['prompt_attention_mask']),
            batch['chosen_attention_mask'],
        ],
        dim=-1,
    )
    completion_only_mask_rejected = torch.cat(
        [
            torch.zeros_like(batch['prompt_attention_mask']),
            batch['rejected_attention_mask'],
        ],
        dim=-1,
    )

    # --- 2. Get Model Outputs ---
    # Policy model
    policy_chosen_outputs = model(
        concat_chosen_ids, attention_mask=concat_chosen_mask, return_dict=True
    )
    if torch.isnan(policy_chosen_outputs.logits).any():
        print(
            f'!!! DEBUG: NaN detected in policy_chosen_outputs.logits at step {current_global_step} !!!'
        )
        raise RuntimeError(
            f'NaN in policy chosen logits at step {current_global_step}'
        )

    policy_rejected_outputs = model(
        concat_rejected_ids,
        attention_mask=concat_rejected_mask,
        return_dict=True,
    )
    if torch.isnan(policy_rejected_outputs.logits).any():
        print(
            f'!!! DEBUG: NaN detected in policy_rejected_outputs.logits at step {current_global_step} !!!'
        )
        raise RuntimeError(
            f'NaN in policy rejected logits at step {current_global_step}'
        )

    # Reference model
    with torch.no_grad():
        ref_chosen_outputs = ref_model(
            concat_chosen_ids,
            attention_mask=concat_chosen_mask,
            return_dict=True,
        )
        ref_rejected_outputs = ref_model(
            concat_rejected_ids,
            attention_mask=concat_rejected_mask,
            return_dict=True,
        )
        # Add NaN checks for ref model logits too, if concerned they might cause issues downstrea
        if (
            torch.isnan(ref_chosen_outputs.logits).any()
            or torch.isnan(ref_rejected_outputs.logits).any()
        ):
            print(
                f'!!! WARNING: NaN detected in REFERENCE model logits at step {current_global_step} !!!'
            )
            # Logprobs extraction might handle this, but good to know

    # --- 3. Extract Log Probabilities for Completions ---
    policy_chosen_logprobs = extract_log_probs(
        policy_chosen_outputs.logits,
        concat_chosen_ids,
        prompt_only_mask_chosen,
        completion_only_mask_chosen,
        current_global_step,
    )
    policy_rejected_logprobs = extract_log_probs(
        policy_rejected_outputs.logits,
        concat_rejected_ids,
        prompt_only_mask_rejected,
        completion_only_mask_rejected,
        current_global_step,
    )
    with torch.no_grad():   # Ensure ref logprobs don't track grads
        ref_chosen_logprobs = extract_log_probs(
            ref_chosen_outputs.logits,
            concat_chosen_ids,
            prompt_only_mask_chosen,
            completion_only_mask_chosen,
            current_global_step,
        )
        ref_rejected_logprobs = extract_log_probs(
            ref_rejected_outputs.logits,
            concat_rejected_ids,
            prompt_only_mask_rejected,
            completion_only_mask_rejected,
            current_global_step,
        )

    # --- 4. Compute DPO Loss ---
    loss, chosen_rewards, rejected_rewards = dpo_loss_function(
        policy_chosen_logprobs,
        policy_rejected_logprobs,
        ref_chosen_logprobs,
        ref_rejected_logprobs,
        beta=beta,
        current_global_step=current_global_step,
    )

    return loss, chosen_rewards, rejected_rewards


# --- DPO Training Loop ---
model.train()
optimizer.zero_grad()   # Zero gradients initially

print(
    f'Starting training from Epoch {start_epoch+1}, Global Step {global_step}...'
)   # Display 1-based start epoch

# Use range starting from start_epoch
for epoch in range(start_epoch, EPOCHS):
    print(f'--- Epoch {epoch+1}/{EPOCHS} ---')   # Display 1-based epoch
    model.train()   # Ensure model is in training mode at the start of each epoch

    num_batches_per_epoch = len(
        train_dataloader
    )   # Total microbatches in dataloader
    if num_batches_per_epoch == 0:
        print(
            f'Warning: Train dataloader for epoch {epoch+1} is empty. Skipping epoch.'
        )
        continue   # Skip to the next epoch if dataloader is empty

    num_optimizer_steps_per_epoch = math.ceil(
        num_batches_per_epoch / ACCUMULATION_STEPS
    )
    if (
        num_optimizer_steps_per_epoch == 0
    ):   # Should not happen if num_batches > 0, but safety check
        print(
            f'Warning: Calculated 0 optimizer steps for epoch {epoch+1}. Skipping epoch.'
        )
        continue

    microbatches_to_skip = 0
    if epoch == start_epoch and global_step > 0:
        # Calculate how many optimizer steps were completed *in this specific epoch*
        optimizer_steps_in_epoch = global_step % num_optimizer_steps_per_epoch
        if optimizer_steps_in_epoch > 0:
            # Calculate microbatches corresponding to completed optimizer steps
            microbatches_to_skip = (
                optimizer_steps_in_epoch * ACCUMULATION_STEPS
            )
            # Ensure we don't try to skip more than available
            microbatches_to_skip = min(
                microbatches_to_skip, num_batches_per_epoch
            )
            print(
                f'Resuming epoch {epoch+1}. Skipping first {microbatches_to_skip}/{num_batches_per_epoch} microbatches (based on global_step {global_step}).'
            )
        elif (
            global_step >= num_optimizer_steps_per_epoch
            and num_optimizer_steps_per_epoch > 0
        ):
            # Resuming, but previous epoch finished exactly or global_step is beyond this epoch's expected steps
            # This case should ideally be handled by `start_epoch` being incremented correctly upon load.
            # If we land here, it implies `start_epoch` might not have been incremented in the checkpoint,
            # or we are resuming precisely at an epoch boundary. Starting from batch 0 is correct.
            print(
                f'Resuming at the start of epoch {epoch+1} (global_step {global_step}). Skipping 0 batches.'
            )
        else:
            # Resuming near the beginning or exactly at the start
            print(
                f'Resuming near the start of epoch {epoch+1} (global_step {global_step}). Skipping 0 batches.'
            )

    # Initialize dataloader iterator
    dataloader_iter = iter(train_dataloader)

    # Skip batches if resuming mid-epoch
    if microbatches_to_skip > 0:
        # Use a simple loop for skipping, tqdm is optional here and can be verbose
        print(f'Skipping {microbatches_to_skip} batches...')
        for i in range(microbatches_to_skip):
            try:
                next(dataloader_iter)
            except StopIteration:
                print(
                    f'\nWarning: StopIteration encountered while skipping batch {i+1}/{microbatches_to_skip}. DataLoader exhausted early.'
                )
                # Adjust remaining count if needed, although loop below should handle empty iterator
                remaining_microbatches = 0   # No batches left to process
                break
        print('Finished skipping.')

    # Setup progress bar for the *entire* epoch, starting from the skipped count
    progress_bar = tqdm(
        total=num_batches_per_epoch,  # Show total microbatches for the epoch
        initial=microbatches_to_skip,  # Start the visual bar at the skipped count
        desc=f'Epoch {epoch+1} Training',
        leave=True,
        # Set mininterval high if updates are too frequent and causing issues
        mininterval=5.0,  # Update progress bar at most every 5 seconds
    )

    microbatch_step_in_epoch = 0   # Tracks microbatches processed *in this run* of the epoch loop (after skipping)
    accumulated_loss = 0.0
    accumulated_chosen_rewards = 0.0
    accumulated_rejected_rewards = 0.0
    accumulated_accurate_samples = 0
    accumulated_samples = 0

    # Iterate over the *remaining* batches using the iterator
    # Enumerate provides an index if needed, starting from the skip count
    for batch_idx, batch in enumerate(
        dataloader_iter, start=microbatches_to_skip
    ):

        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        current_microbatch_size = batch['prompt_input_ids'].size(0)

        # --- Forward Pass ---
        loss, chosen_rewards, rejected_rewards = get_batch_loss(
            batch,
            model,
            ref_model,
            BETA,
            global_step,  # Pass current global_step
        )

        # Check for NaN loss
        if torch.isnan(loss):
            print(
                f'WARNING: NaN loss detected at global step {global_step}, epoch {epoch+1}, batch index ~{batch_idx}. Skipping microbatch.'
            )
            # Skip update and optimizer step for this microbatch
            # Crucially, do not increment microbatch_step_in_epoch here
            # Update progress bar manually since we're skipping the normal update path
            progress_bar.update(1)
            continue   # Go to the next microbatch

        # Scale loss for gradient accumulation
        scaled_loss = loss / ACCUMULATION_STEPS

        # Accumulate metrics for logging
        accumulated_loss += loss.item() * current_microbatch_size
        accumulated_chosen_rewards += chosen_rewards.sum().item()
        accumulated_rejected_rewards += rejected_rewards.sum().item()
        accumulated_accurate_samples += (
            (chosen_rewards > rejected_rewards).sum().item()
        )
        accumulated_samples += current_microbatch_size

        # --- Backward Pass ---
        scaled_loss.backward(retain_graph=False)

        microbatch_step_in_epoch += 1   # Increment microbatch counter *after* successful forward/backward

        # --- Optimizer Step Logic ---
        # Determine the effective microbatch number within the epoch context
        # This ensures optimizer steps happen at correct absolute batch intervals
        effective_microbatch_num_in_epoch = (
            microbatches_to_skip + microbatch_step_in_epoch
        )
        # Check if it's time for an optimizer step
        if effective_microbatch_num_in_epoch % ACCUMULATION_STEPS == 0:
            # Gradient Clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            # Optional: Check gradient norm for NaNs/Infs before optimizer step
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(
                    f'WARNING: NaN/Inf gradient norm detected ({grad_norm:.2f}) at global step {global_step} before optimizer step. Skipping optimizer step.'
                )
                optimizer.zero_grad()   # Zero out potentially problematic gradients
            else:
                # Optimizer Step
                optimizer.step()
                optimizer.zero_grad()   # Clear gradients *after* successful step

            # Increment global step *only after* attempting (or skipping) an optimizer step
            global_step += 1

            # --- Logging, Validation, Checkpointing ---
            # These should trigger based on the updated global_step

            # --- Log Training Statistics ---
            if global_step > 0 and global_step % LOG_INTERVAL == 0:
                if accumulated_samples > 0:
                    avg_loss = accumulated_loss / accumulated_samples
                    avg_chosen_reward = (
                        accumulated_chosen_rewards / accumulated_samples
                    )
                    avg_rejected_reward = (
                        accumulated_rejected_rewards / accumulated_samples
                    )
                    reward_acc = (
                        accumulated_accurate_samples / accumulated_samples
                    )
                    print(
                        f'\n[TRAIN] Step: {global_step}, Epoch: {epoch+1}, Avg Loss: {avg_loss:.4f}, '
                        f'Avg Chosen Reward: {avg_chosen_reward:.3f}, Avg Rejected Reward: {avg_rejected_reward:.3f}, '
                        f'Reward Acc: {reward_acc:.3f}'
                    )
                    progress_bar.set_postfix(
                        {
                            'Loss': f'{avg_loss:.4f}',
                            'Acc': f'{reward_acc:.2f}',
                            'GradNorm': f'{grad_norm:.2f}',
                        }
                    )
                else:
                    print(
                        f'\n[TRAIN] Step: {global_step}, No samples processed in the last {LOG_INTERVAL} steps.'
                    )
                # Reset accumulators
                (
                    accumulated_loss,
                    accumulated_chosen_rewards,
                    accumulated_rejected_rewards,
                ) = (0.0, 0.0, 0.0)
                accumulated_accurate_samples, accumulated_samples = 0, 0

            # --- Validation Step --- (triggered by global_step and eval_steps)
            if global_step > 0 and global_step % EVAL_STEPS == 0:
                print(
                    f'\n--- Running Validation on approx. {MAX_EVAL_SAMPLES} samples at Step {global_step} ---'
                )
                model.eval()  # Switch to evaluation mode

                total_eval_loss = 0.0
                total_eval_chosen_rewards = 0.0
                total_eval_rejected_rewards = 0.0
                total_eval_accurate_samples = 0
                total_eval_samples = (
                    0  # Tracks samples processed *in this validation run*
                )

                val_progress_bar = tqdm(
                    val_dataloader,
                    desc=f'Validation (subset ~{MAX_EVAL_SAMPLES})',
                    leave=False,  # Make the bar disappear after completion
                    mininterval=5.0,
                )
                with torch.no_grad():  # Ensure no gradients are computed during validation
                    for val_batch in val_progress_bar:
                        # Check if we've processed enough samples already
                        if total_eval_samples >= MAX_EVAL_SAMPLES:
                            break  # Stop evaluating early

                        # Move batch to device
                        val_batch = {
                            k: v.to(device) for k, v in val_batch.items()
                        }
                        val_microbatch_size = val_batch[
                            'prompt_input_ids'
                        ].size(0)

                        # Get loss and rewards for the validation batch
                        (
                            eval_loss,
                            eval_chosen_rewards,
                            eval_rejected_rewards,
                        ) = get_batch_loss(
                            val_batch, model, ref_model, BETA, global_step
                        )

                        if not torch.isnan(eval_loss):  # Skip NaN eval losses
                            # Accumulate total loss weighted by samples in the microbatch
                            total_eval_loss += (
                                eval_loss.item() * val_microbatch_size
                            )
                            # Accumulate reward stats for validation set
                            total_eval_chosen_rewards += (
                                eval_chosen_rewards.sum().item()
                            )
                            total_eval_rejected_rewards += (
                                eval_rejected_rewards.sum().item()
                            )
                            total_eval_accurate_samples += (
                                (eval_chosen_rewards > eval_rejected_rewards)
                                .sum()
                                .item()
                            )
                            total_eval_samples += val_microbatch_size
                        else:
                            print(
                                'Warning: NaN detected in validation loss during evaluation.'
                            )

                # Calculate and print average validation metrics
                if total_eval_samples > 0:
                    avg_eval_loss = total_eval_loss / total_eval_samples
                    avg_eval_chosen_reward = (
                        total_eval_chosen_rewards / total_eval_samples
                    )
                    avg_eval_rejected_reward = (
                        total_eval_rejected_rewards / total_eval_samples
                    )
                    avg_eval_reward_acc = (
                        total_eval_accurate_samples / total_eval_samples
                    )

                    print(f'--- Validation Complete ---')
                    print(
                        f'[VAL] Step: {global_step}, Avg Loss: {avg_eval_loss:.4f}, '
                        f'Avg Chosen Reward: {avg_eval_chosen_reward:.3f}, '
                        f'Avg Rejected Reward: {avg_eval_rejected_reward:.3f}, '
                        f'Reward Acc: {avg_eval_reward_acc:.3f}'
                    )
                    # Add early stopping logic here based on avg_eval_loss or avg_eval_reward_acc if desired
                else:
                    print(
                        '--- Validation Warning: No valid samples processed (all losses might have been NaN) ---'
                    )

                model.train()  # Set back to training mode before continuing training loop

            # --- Checkpoint Saving ---
            if global_step > 0 and global_step % SAVE_INTERVAL == 0:
                step_checkpoint_path = os.path.join(
                    CHECKPOINT_DIR, f'step_{global_step}.pth'
                )
                latest_checkpoint_path = os.path.join(
                    CHECKPOINT_DIR, 'latest_checkpoint.pth'
                )
                state = {
                    'epoch': epoch,  # Save the CURRENT epoch index
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                try:
                    torch.save(state, step_checkpoint_path)
                    torch.save(
                        state, latest_checkpoint_path
                    )   # Overwrite latest
                    print(
                        f'\nCheckpoint saved at step {global_step} (during epoch {epoch+1}) to {step_checkpoint_path} and {latest_checkpoint_path}'
                    )
                except Exception as e:
                    print(
                        f'\nError saving checkpoint at step {global_step}: {e}'
                    )

        # Update the progress bar after processing each microbatch
        progress_bar.update(1)

    # --- End of Batch Loop ---
    progress_bar.close()   # Close the progress bar for the epoch
    # Ensure any remaining gradients are zeroed if epoch ends mid-accumulation cycle
    # Although zeroing happens after optimizer step, this is safe.
    optimizer.zero_grad(set_to_none=True)
    print(f'--- Epoch {epoch+1} Complete ---')
    # (Optional end-of-epoch checkpoint save logic could go here)

# --- End of Epoch Loop ---
print('Training finished.')

# Save the final trained model to a separate directory
print(f'Saving final model to {FINAL_MODEL_DIR}...')
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
try:
    # Use model.save_pretrained for Hugging Face compatibility
    model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    # You might also want to save other training args or metrics here
    print('Final model saved!')
except Exception as e:
    print(f'Error saving final model: {e}')

print('done!')

## LLM-GENERATED CODE END ##
