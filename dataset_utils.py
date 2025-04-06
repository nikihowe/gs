from datasets import Dataset


# Inspired by https://github.com/phymhan/llm-dpo/blob/41ddeaea2782f7f5c6d79b9f2041a91921aadbd0/preference_datasets.py#L14  # noqa: E501
def extract_prompt_and_response(example: str) -> Dataset:
    # We need to look for a `\n\nAssistant:` from the right
    match_string = '\n\nAssistant: '
    assistant_start_idx = example.rfind(match_string)
    assert assistant_start_idx != -1, 'No Assistant found in the example'

    return (
        example[:assistant_start_idx + len(match_string)],
        example[assistant_start_idx + len(match_string) :],
    )


def dpoify_dataset(dataset: Dataset) -> list[dict]:
    """
    DPOifies the dataset by creating a new dataset with `prompt`, `chosen`, and `rejected`.

    Args:
        dataset (Dataset): The HH dataset.

    Returns:
        Dataset: The DPOified dataset.
    """
    # 'chosen': <SOME PROMPT> \n\nAssistant: <ASSISTANT RESPONSE>"
    # 'rejected': <SOME PROMPT> \n\nAssistant: <ASSISTANT RESPONSE>"
    # Note that "SOME PROMPT" is the same for both chosen and rejected

    new_dataset = {'prompt': [], 'chosen': [], 'rejected': []}
    num_skipped = 0
    for example in dataset:
        prompt, chosen = extract_prompt_and_response(example['chosen'])
        prompt2, rejected = extract_prompt_and_response(example['rejected'])
        if prompt != prompt2:
            num_skipped += 1
            continue
        new_dataset['prompt'].append(prompt)
        new_dataset['chosen'].append(chosen)
        new_dataset['rejected'].append(rejected)
    print('num skipped', num_skipped)
    return Dataset.from_dict(new_dataset)
