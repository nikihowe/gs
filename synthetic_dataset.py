import json

from vllm import LLM, SamplingParams

REPLACE_TOKEN = '<|REPLACE|>'
SAMPLING_PARAMS = SamplingParams(
    seed=0,
    n=2000,  # How many good and bad samples to generate
    max_tokens=50,
)


# For generative models (task=generate) only
# NOTE: some this code was copied from the vllm documentation and modified
llm = LLM(model='allenai/OLMo-2-1124-13B-Instruct', task='generate')

with open('./prompt.txt', 'r') as fp:
    prompt = fp.read()

# Now, replace the token with benign and halmful
good_prompt = prompt.replace('<|REPLACE|>', 'benign')
bad_prompt = prompt.replace('<|REPLACE|>', 'harmful')

# Now generate the actual datasets
good_outputs = llm.generate(good_prompt, sampling_params=SAMPLING_PARAMS)[
    0
].outputs
good_prompts = [output.text for output in good_outputs]

bad_outputs = llm.generate(bad_prompt, sampling_params=SAMPLING_PARAMS)[
    0
].outputs
bad_prompts = [output.text for output in bad_outputs]

dataset = {
    'text': good_prompts + bad_prompts,
    'harmful': [0] * len(good_prompts) + [1] * len(bad_prompts),
}
print('together', dataset)

# Now save the datasets
with open('./datasets/dataset.json', 'w') as fp:
    json.dump(dataset, fp)

# Check that the datasets can be loaded correctly and are the same
# as the ones that we generated
with open('./datasets/dataset.json', 'r') as fp:
    dataset_2 = json.load(fp)
assert dataset == dataset_2
