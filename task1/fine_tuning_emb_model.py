import pandas as pd
import random
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
from transformers import EarlyStoppingCallback
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

train_df = pd.read_parquet("train-00000-of-00001.parquet")
#train_df = pd.read_parquet("train_augmented_para_abs_both.parquet")
train_df = train_df.dropna(subset=["claim", "abstract"])

train_df["claim"] = "query: " + train_df["claim"]
train_df["abstract"] = "passage: " + train_df["abstract"]

train_sample = train_df.sample(frac=0.99, random_state=42)
eval_sample = train_df.drop(train_sample.index)

embedder = SentenceTransformer("intfloat/e5-large-v2")

abstract_list = train_sample["abstract"].tolist()
abstract_embeddings = embedder.encode(abstract_list, convert_to_tensor=True, show_progress_bar=True)
claim_list = train_sample["claim"].tolist()
claim_embeddings = embedder.encode(claim_list, convert_to_tensor=True, show_progress_bar=True)

hard_negatives = []
for idx, claim_emb in enumerate(claim_embeddings):
    scores = cosine_similarity(claim_emb.cpu().numpy().reshape(1, -1), abstract_embeddings.cpu().numpy())[0]
    scores[idx] = -1  # Mask true positive
    top_negatives = np.argsort(scores)[-3:]
    hard_negatives.append([abstract_list[i] for i in top_negatives])

triplet_data = Dataset.from_dict({
    "anchor": claim_list,
    "positive": abstract_list,
    "negative": hard_negatives,
})

eval_triplet_data = Dataset.from_dict({
    "anchor": eval_sample["claim"].tolist(),
    "positive": eval_sample["abstract"].tolist(),
    "negative": random.choices(eval_sample["abstract"].tolist(), k=len(eval_sample)),
})

m_name = "thenlper/gte-large"
#m_name = "intfloat/e5-large-v2"
model = SentenceTransformer(m_name)

loss = MultipleNegativesRankingLoss(model)

r_label = "intfloat/gte-large/final"
args = SentenceTransformerTrainingArguments(
    output_dir=r_label,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,
    bf16=False,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

dev_evaluator = TripletEvaluator(
    anchors=eval_triplet_data["anchor"],
    positives=eval_triplet_data["positive"],
    negatives=eval_triplet_data["negative"],
    name="claims-abstracts-dev",
    main_similarity_function="cosine",
)
dev_evaluator(model)

early_stopper = EarlyStoppingCallback(
    early_stopping_patience=5,
    #early_stopping_threshold=0.05,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=triplet_data,
    eval_dataset=eval_triplet_data,
    loss=loss,
    evaluator=dev_evaluator,
    callbacks=[early_stopper],
)
trainer.train()

model.save_pretrained(r_label)
