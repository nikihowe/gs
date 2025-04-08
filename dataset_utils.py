from transformers import PreTrainedTokenizer

from constants import TIME_SIZE
from datasets import Dataset, load_dataset, load_from_disk


# Inspired by https://github.com/phymhan/llm-dpo/blob/41ddeaea2782f7f5c6d79b9f2041a91921aadbd0/preference_datasets.py#L14  # noqa: E501
def extract_prompt_and_response(example: str) -> Dataset:
    # We need to look for a `\n\nAssistant:` from the right
    match_string = '\n\nAssistant: '
    assistant_start_idx = example.rfind(match_string)
    assert assistant_start_idx != -1, 'No Assistant found in the example'

    return (
        example[: assistant_start_idx + len(match_string)],
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


def filter_too_long(
    dataset: Dataset, tokenizer: PreTrainedTokenizer, max_length: int
) -> Dataset:
    """
    Filters the dataset to remove examples that are too long.

    Args:
        dataset (Dataset): The dataset to filter.
        max_length (int): The maximum length of the examples.

    Returns:
        Dataset: The filtered dataset.
    """
    # Tokenize the dataset first
    tokenized_chosen = dataset.map(
        lambda x: tokenizer(x['chosen']), batched=True
    )
    tokenized_rejected = dataset.map(
        lambda x: tokenizer(x['rejected']), batched=True
    )
    too_long_indices = []

    # Add an index to the dataset so we know what to remove
    dataset = dataset.add_column('idx', list(range(len(dataset['chosen']))))

    for i, (chosen, rejected) in enumerate(
        zip(tokenized_chosen['input_ids'], tokenized_rejected['input_ids'])
    ):
        if len(chosen) > max_length or len(rejected) > max_length:
            too_long_indices.append(i)
    print(
        f'Found {len(too_long_indices)} examples that are too long, removing them'
    )

    dataset = dataset.filter(lambda x: x['idx'] not in too_long_indices)
    dataset = dataset.remove_columns(['idx'])
    return dataset


def get_the_datasets(
    tokenizer: PreTrainedTokenizer, test: bool = False
) -> tuple[Dataset, Dataset, Dataset]:
    if test:
        dset_type = 'test'
    else:
        dset_type = 'train'
    try:
        tokenized_prompt = load_from_disk(f'tokenized_{dset_type}_prompt')
        tokenized_chosen = load_from_disk(f'tokenized_{dset_type}_chosen')
        tokenized_rejected = load_from_disk(f'tokenized_{dset_type}_rejected')
        print(f'Loaded {dset_type} datasets from disk')
    except:
        print(
            f'Unable to load {dset_type} datasets from disk, so getting originals and tokenizing them'
        )
        dataset = load_dataset(
            'Unified-Language-Model-Alignment/Anthropic_HH_Golden'
        )

        print(
            'originally, train dataset has',
            len(dataset['train']['chosen']),
            'examples',
        )
        print(
            'originally, test dataset has',
            len(dataset['test']['chosen']),
            'examples',
        )
        dataset['train'] = filter_too_long(
            dataset['train'],
            tokenizer,
            TIME_SIZE - 10,  # buffer for retokenization
        )
        dataset['test'] = filter_too_long(
            dataset['test'],
            tokenizer,
            TIME_SIZE - 10,  # buffer for retokenization
        )
        print(
            'now train dataset has',
            len(dataset['train']['chosen']),
            'examples',
        )
        print(
            'now test dataset has', len(dataset['test']['chosen']), 'examples'
        )

        dpoified_dataset = dpoify_dataset(dataset[dset_type])

        # print('first datapoint', dpoified_dataset[0])
        # print('second datapoint', dpoified_dataset[1])

        small = dpoified_dataset

        tokenized_prompt = small.map(
            lambda x: tokenizer(
                x['prompt'],
                truncation=True,
                padding='max_length',
                max_length=TIME_SIZE,
            ),
            batched=True,
        )
        tokenized_prompt = tokenized_prompt.remove_columns(
            ['chosen', 'rejected']
        )
        tokenized_chosen = small.map(
            lambda x: tokenizer(
                x['chosen'],
                truncation=True,
                padding='max_length',
                max_length=TIME_SIZE,
            ),
            batched=True,
        )
        tokenized_chosen = tokenized_chosen.remove_columns(
            ['prompt', 'rejected']
        )

        tokenized_rejected = small.map(
            lambda x: tokenizer(
                x['rejected'],
                truncation=True,
                padding='max_length',
                max_length=TIME_SIZE,
            ),
            batched=True,
        )
        tokenized_rejected = tokenized_rejected.remove_columns(
            ['prompt', 'chosen']
        )

        # Save the tokenized datasets so we don't have to re-tokenize them every time
        tokenized_prompt.save_to_disk(f'tokenized_{dset_type}_prompt')
        tokenized_chosen.save_to_disk(f'tokenized_{dset_type}_chosen')
        tokenized_rejected.save_to_disk(f'tokenized_{dset_type}_rejected')

    print(f'the {dset_type} dataset has {len(tokenized_prompt)} examples')
    return (
        tokenized_prompt,
        tokenized_chosen,
        tokenized_rejected,
    )
