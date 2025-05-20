import pandas as pd
from vllm import LLM, SamplingParams
from tqdm import tqdm

# Load original dataset
train_df = pd.read_parquet("train-00000-of-00001.parquet")

# Initialize vLLM with full Phi-4 model
llm = LLM(model="microsoft/phi-4")
sampling_params = SamplingParams(temperature=1.0, max_tokens=512)

# Define paraphrasing prompt templates
claim_template = """
<|im_start|>system<|im_sep|>You are a professional writer. Please paraphrase the following claim while keeping its meaning intact.<|im_end|><|im_start|>user<|im_sep|>The claim: {claim}\nYour paraphrased claim: <|im_end|><|im_start|>assistant<|im_sep|>
"""

abstract_template = """
<|im_start|>system<|im_sep|>You are a professional writer. Please paraphrase the following abstract while keeping its meaning intact.<|im_end|><|im_start|>user<|im_sep|>The abstract: {abstract}\nYour paraphrased abstract: <|im_end|><|im_start|>assistant<|im_sep|>
"""

batch_size = 32

# Batched inference function
def batch_generate(prompts, batch_size=batch_size):
    results = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)
        results.extend([output.outputs[0].text.strip() for output in outputs])
    return results

# Prepare prompts
augmented_rows = []
for i in tqdm(range(0, len(train_df), batch_size)):
    batch = train_df.iloc[i:i+batch_size]
    claim_prompts = [claim_template.format(claim=row.claim) for _, row in batch.iterrows()]
    abstract_prompts = [abstract_template.format(abstract=row.abstract) for _, row in batch.iterrows()]

    try:
        paraphrased_claims = batch_generate(claim_prompts)
        paraphrased_abstracts = batch_generate(abstract_prompts)

        # First round of augmentation
        for j, row in enumerate(batch.itertuples()):
            new_row = row._asdict()
            new_row['claim'] = paraphrased_claims[j]
            new_row['abstract'] = paraphrased_abstracts[j]
            augmented_rows.append(pd.DataFrame([new_row]))

        paraphrased_claims = batch_generate(claim_prompts)
        paraphrased_abstracts = batch_generate(abstract_prompts)

        # First round of augmentation
        for j, row in enumerate(batch.itertuples()):
            new_row = row._asdict()
            new_row['claim'] = paraphrased_claims[j]
            new_row['abstract'] = paraphrased_abstracts[j]
            augmented_rows.append(pd.DataFrame([new_row]))

    except Exception as e:
        print("Batch failed:", e)

# Combine and save
augmented_df = pd.concat([train_df] + augmented_rows, ignore_index=True)
augmented_df.to_parquet("train_augmented_para_abs_both.parquet")
