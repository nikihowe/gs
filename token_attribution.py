import torch
import multiprocessing
from colorama import Fore, Style
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from params import TEST_PROMPTS
from utils import (
    device,
    load_models,
    print_model_comparison,
)


# ------------------------------------------------------------------------------------------------
# Token attribution. You need to implement this
# ------------------------------------------------------------------------------------------------


def token_attribution(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    completion: str,
) -> list[tuple[str, float]]:
    # Implement this function. It should take a model, tokenizer, prompt, and completion, and return a list
    # of tuples representing some token attributions for the prompt tokens.
    #
    # The list should be in the same order as the prompt tokens.
    # The scores can be any finite float value where higher numbers indicate higher importance of the
    # token, ideally with linear scale.
    #
    # Example:
    #   - Input prompt: "The capital of France"
    #   - Input completion: "is Paris"
    #   - Output: [("The", 0.1), ("capital", 0.9), ("of", 0.05), ("France", 1.8)]
    return []


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
    num_cores = multiprocessing.cpu_count()
    torch.set_num_threads(num_cores)
    torch.set_num_interop_threads(num_cores)
    model, base_model, tokenizer = load_models()
    print(
        f"{Fore.RED}Red = high impact{Style.RESET_ALL}, {Fore.YELLOW}Yellow = medium impact{Style.RESET_ALL}, {Fore.GREEN}Green = low impact{Style.RESET_ALL}"
    )
    base_attribution_modifier = create_attribution_display(base_model, tokenizer)
    tuned_attribution_modifier = create_attribution_display(model, tokenizer)
    for prompt in TEST_PROMPTS:
        print_model_comparison(
            model,
            base_model,
            tokenizer,
            prompt,
            base_modifier=base_attribution_modifier,
            tuned_modifier=tuned_attribution_modifier,
        )