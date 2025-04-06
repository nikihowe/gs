import torch
from colorama import Fore, Style
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

from constants import ACCEPTANCE, REJECTION


# ------------------------------------------------------------------------------------------------
# Token attribution. You need to implement this
# ------------------------------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def token_attribution(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    completion: str,
) -> list[tuple[str, float]]:
    # Implement this function. It should take a model, tokenizer, prompt, and completion, and return a list
    # of tuples representing some token attributions for the prompt tokens.
    #
    # The list should be in the same order as the prompt tokens.
    # The scores can be any finite float value where higher numbers indicate higher importance of the
    # token, ideally with linear scale.

    # We're going to calculate attribution scores by masking out each token in the prompt
    # and seeing how that affects the probabilities of the completion.

    # B: batch size
    # T: time size (variable in this case)
    # V: vocab size

    # Set the model to eval mode
    model.eval()

    # Tokenize the prompt and completion
    prompt_tokens = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = prompt_tokens.input_ids.shape[1]
    completion_tokens = tokenizer(completion, return_tensors="pt").to(device)

    # Now we loop over the different prompt tokens and look at the output probs

    attribution_scores = []
    for i in range(prompt_tokens.input_ids.shape[1]):
        # Mask out the i-th token in the prompt
        # We do this by replacing it with a pad token
        # and setting the attention mask to 0 for that token
        masked_prompt_input_ids_BT = prompt_tokens.input_ids.clone().to(device)
        masked_prompt_input_ids_BT[:, i] = tokenizer.pad_token_id
        masked_prompt_att_mask_BT = prompt_tokens.attention_mask.clone().to(device)
        masked_prompt_att_mask_BT[:, i] = 0

        masked_completion_input_ids_BT = completion_tokens.input_ids.clone().to(device)
        masked_completion_att_mask_BT = completion_tokens.attention_mask.clone().to(device)

        full_input_ids_BT = torch.cat(
            (masked_prompt_input_ids_BT, masked_completion_input_ids_BT), dim=1
        )
        full_att_mask_BT = torch.cat(
            (masked_prompt_att_mask_BT, masked_completion_att_mask_BT), dim=1
        )

        # Get the model's output probabilities for the masked prompt
        with torch.no_grad():
            outputs = model(
                input_ids=full_input_ids_BT,
                attention_mask=full_att_mask_BT,
                return_dict=True,
            )
            logits_BTV = outputs.logits

        # Calculate the probability of the completion given the masked prompt
        # Note that the completion can have multiple tokens so we need to average over them
        completion_logits_BTV = logits_BTV[:, prompt_length:, :]
        completion_probs_BTV = torch.softmax(completion_logits_BTV, dim=-1)

        # Get the log probabilities of the completion tokens
        completion_probs_BT = completion_probs_BTV.gather(
            2, completion_tokens.input_ids.unsqueeze(-1)
        ).squeeze(-1)
        completion_probs_B = completion_probs_BT.mean(dim=-1)

        # Assume batch size is 1 for now
        # TODO: make work for bigger batch sizes
        assert completion_probs_B.shape == (1,)
        completion_probs_B = completion_probs_B.squeeze(0)

        # Get the str verson of the prompt token we masked
        # TODO: remove assumption that batch is 1
        prompt_token = prompt_tokens.input_ids[0, i]
        str_token = tokenizer.decode(prompt_token, skip_special_tokens=True)

        attribution_scores.append((str_token, completion_probs_B.item()))

    # Example:
    #   - Input prompt: "The capital of France"
    #   - Input completion: "is Paris"
    #   - Output: [("The", 0.1), ("capital", 0.9), ("of", 0.05), ("France", 1.8)]
    return attribution_scores


# ------------------------------------------------------------------------------------------------
# Helpers to visualize attribution scores. You shouldn't need to change anything below here.
# ------------------------------------------------------------------------------------------------


# Uses the attribution scores to colorize high-scoring tokens for visualization
def visualize_attribution(
    tokens_with_attributions: list[tuple[str, float]],
    max_attr: float,
) -> str:
    colored_text = ""
    for token, attribution in tokens_with_attributions:
        # Skip special tokens
        if token.startswith("<") and token.endswith(">"):
            continue

        # Normalize attribution value (0-1 scale)
        norm_attribution = min(1.0, attribution / max_attr)
        if norm_attribution > 0.7:
            color = Fore.RED  # High importance
        elif norm_attribution > 0.4:
            color = Fore.YELLOW  # Medium importance
        else:
            color = Fore.GREEN  # Low importance

        # Add space before token if needed
        if token.startswith("Ġ"):
            token = " " + token[1:]
        colored_text += f"{color}{token}{Style.RESET_ALL}"
    return colored_text


# Wrapper that adds attribution visualization to a model's completion
def create_attribution_display(model, tokenizer):
    def attribution_modifier(prompt, completion):
        # Colorize prompt tokens based on attribution scores
        completion_text = completion[len(prompt) :]
        attribution = token_attribution(model, tokenizer, prompt, completion_text)
        max_attr = max(attr for _, attr in attribution)
        colored_text = visualize_attribution(attribution, max_attr)
        return colored_text + completion_text

    return attribution_modifier


if __name__ == "__main__":
    # Load the models and tokenizer
    checkpoint = 'HuggingFaceTB/SmolLM2-135M-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    base_model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

    trained_model_path = "dpo_model"
    trained_model = AutoModelForCausalLM.from_pretrained(trained_model_path).to(device)

    # Make sure the trained model is in fact different from the base model
    assert trained_model != base_model
    # Set the models to eval mode
    base_model.eval()
    trained_model.eval()
    for param in trained_model.parameters():
        param.requires_grad = False
    for param in base_model.parameters():
        param.requires_grad = False

    base_attribution_modifier = create_attribution_display(base_model, tokenizer)
    tuned_attribution_modifier = create_attribution_display(trained_model, tokenizer)

    # Load the synthetic dataset
    with open('./datasets/dataset.json', 'r') as fp:
        dataset = json.load(fp)
    
    print("Loaded dataset with", len(dataset['text']), "examples")
    # Split the dataset into good and bad examples
    good_examples = [text for text, label in zip(dataset['text'], dataset['harmful']) if label == 1]
    good_examples = ["Human: " + text + " Assistant: " for text in good_examples]
    bad_examples = [text for text, label in zip(dataset['text'], dataset['harmful']) if label == 1]
    bad_examples = ["Human: " + text + " Assistant: " for text in bad_examples]

    for i in range(5):
        base_model_response = base_model.generate(
            input_ids=tokenizer(good_examples[i], return_tensors="pt").input_ids.to(device),
            attention_mask=tokenizer(good_examples[i], return_tensors="pt").attention_mask.to(device),
            max_length=50,
            num_return_sequences=1,
        )
        base_model_response = tokenizer.decode(base_model_response[0], skip_special_tokens=True)
        print("Base model response:", base_model_response)

        tuned_model_response = trained_model.generate(
            input_ids=tokenizer(good_examples[i], return_tensors="pt").input_ids.to(device),
            attention_mask=tokenizer(good_examples[i], return_tensors="pt").attention_mask.to(device),
            max_length=50,
            num_return_sequences=1,
        )
        tuned_model_response = tokenizer.decode(tuned_model_response[0], skip_special_tokens=True)
        print("Tuned model response:", tuned_model_response)

        base_model_good_attribution = token_attribution(base_model, tokenizer, good_examples[i], base_model_response)
        tuned_model_good_attribution = token_attribution(trained_model, tokenizer, good_examples[i], tuned_model_response)
        print("Base model good attribution:", base_model_good_attribution)
        print("Tuned model good attribution:", tuned_model_good_attribution)
        print("Base model good attribution visualization:", visualize_attribution(base_model_good_attribution, max_attr=1.0))
        print("Tuned model good attribution visualization:", visualize_attribution(tuned_model_good_attribution, max_attr=1.0))
        