import json
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import numpy as np
import os
from sklearn.metrics import classification_report, accuracy_score

#train_df = pd.read_parquet("train-00000-of-00001.parquet")
train_df = pd.read_parquet("train_augmented_para_abs_both.parquet")
train_df, dev_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['annotation'])

def construct_input_text(row):
    return row['claim'] + "[SEP]" + row['abstract']

train_df['text'] = train_df.apply(construct_input_text, axis=1)
dev_df['text']   = dev_df.apply(construct_input_text, axis=1)

label_encoder = LabelEncoder()
train_df['label'] = label_encoder.fit_transform(train_df['annotation'])
dev_df['label']   = label_encoder.transform(dev_df['annotation'])
num_labels = len(label_encoder.classes_)

with open("label_mapping.json", "w") as f:
    json.dump({i: label for i, label in enumerate(label_encoder.classes_)}, f)

class ClaimDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

model_name = "cross-encoder/nli-deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

train_dataset = ClaimDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer)
dev_dataset = ClaimDataset(dev_df["text"].tolist(), dev_df["label"].tolist(), tokenizer)

TARGET_CLASS_N = "Not Enough Information"
TARGET_CLASS_N_IDX = label_encoder.transform([TARGET_CLASS_N])[0]
TARGET_CLASS_R = "Refutes"
TARGET_CLASS_R_IDX = label_encoder.transform([TARGET_CLASS_R])[0]
TARGET_CLASS_S = "Supports"
TARGET_CLASS_S_IDX = label_encoder.transform([TARGET_CLASS_S])[0]



def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, preds)

    report = classification_report(labels, preds, target_names=label_encoder.classes_, output_dict=True, zero_division=0)

    n_accuracy = report[TARGET_CLASS_N]['recall']
    r_accuracy = report[TARGET_CLASS_R]['recall']
    s_accuracy = report[TARGET_CLASS_S]['recall']
    r_f1 = report[TARGET_CLASS_R]['f1-score']

    n_recall = report[TARGET_CLASS_N]['recall']
    r_recall = report[TARGET_CLASS_R]['recall']
    s_recall = report[TARGET_CLASS_S]['recall']
    
    min_recall = min(n_recall, r_recall, s_recall)

    print("-----------")
    print(f"N Recall: {n_accuracy:.4f} | R Recall: {r_accuracy:.4f} | S Recall: {s_accuracy:.4f}")
    print(f"Overall Accuracy: {accuracy:.4f} | R F1-Score: {r_f1:.4f}")
    print("-----------")
    print(f"min recall {min_recall}")
    print("-----------")

    return {
        "accuracy": accuracy,
        "target_accuracy": r_accuracy,
        "refutes_f1": r_f1,
        "supports_recall": s_accuracy,
        "nei_recall": n_accuracy,
        "min_recall": min_recall
    }

model_final_path = "./nli/nli-deberta"

training_args = TrainingArguments(
    output_dir=model_final_path,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model=f"min_recall",
    greater_is_better=True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    num_train_epochs=20,
    learning_rate=1e-5,
    report_to="none",
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
)

trainer.train()
trainer.save_model(model_final_path + "/best_model")
tokenizer.save_pretrained(model_final_path + "/best_model")
print(f"Training complete. Best model saved in: {model_final_path}/best_model")
