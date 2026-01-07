import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
import nltk
from sentence_transformers import SentenceTransformer, CrossEncoder 
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch
import time
import os
import pickle
from codecarbon import EmissionsTracker

tracker = EmissionsTracker()
tracker.start()

nltk.download('punkt')
nltk.download('punkt_tab')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "mps"

test_df_path = "test-00000-of-00001.parquet"
results_label = "preds2"

rer_model = "cross-encoder/ms-marco-MiniLM-L12-v2"
reranker = CrossEncoder(rer_model, device=device)
def rerank_candidates(claim, candidate_abstracts):
    pairs = [(claim, ab) for ab in candidate_abstracts]
    scores = reranker.predict(pairs)
    return np.argsort(scores)[::-1]

class EvidenceRetriever:
    def __init__(self, emb_model_name="intfloat/e5-large-v2", device=device):
        self.bm25_top_k = 1500
        self.cos_sim_top_k = 150
        self.embedding_model = SentenceTransformer(emb_model_name, device=device)
        self.corpus = None
        self.corpus_embeddings = None
        self.tokenized_corpus = None
        self.bm25 = None

    def load_corpus(self, corpus_path):
        corpus_df = pd.read_parquet(corpus_path)
        corpus_df = corpus_df[corpus_df["abstract"].notnull()]
        corpus_df["abstract"] = "passage: " + corpus_df["abstract"]

        self.corpus = {
            "abstracts": corpus_df["abstract"].tolist(),
            "ids": corpus_df["abstract_id"].tolist()
        }

        cache_path = "tokenized_corpus.pkl"
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.tokenized_corpus = pickle.load(f)
        else:
            #token_re = re.compile(r"\b\w+\b")
            #self.tokenized_corpus = [token_re.findall(a.lower()) for a in self.corpus["abstracts"]]
            self.tokenized_corpus = [nltk.word_tokenize(abstract) for abstract in self.corpus["abstracts"]]
            with open(cache_path, "wb") as f:
                pickle.dump(self.tokenized_corpus, f)
        
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print("Encoding corpus abstracts...")
        self.corpus_embeddings = self.embedding_model.encode(self.corpus["abstracts"], show_progress_bar=True, batch_size=256)
        self.corpus_embeddings = np.array(self.corpus_embeddings)

    def retrieve_evidence(self, claim):
        bm25_results = self.bm25_retrieve(claim, top_k=self.bm25_top_k)
        claim = f"query: {claim}"
        semantic_results = self.semantic_retrieve(claim, bm25_results)
        reranked_indices = rerank_candidates(claim, [r[1] for r in semantic_results])
        reranked = [semantic_results[i] for i in reranked_indices]
        return reranked[:10]

    def bm25_retrieve(self, query, top_k):
        tokenized_query = nltk.word_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return {
            "indices": top_indices,
            "abstracts": [self.corpus["abstracts"][i] for i in top_indices],
            "ids": [self.corpus["ids"][i] for i in top_indices]
        }

    def semantic_retrieve(self, query, bm25_results):
        query_embedding = self.embedding_model.encode([query], batch_size=256)
        bm25_embeddings = self.corpus_embeddings[bm25_results["indices"]]
        similarities = cosine_similarity(query_embedding, bm25_embeddings)[0]
        results = list(zip(
            bm25_results["ids"],
            bm25_results["abstracts"],
            similarities
        ))
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:self.cos_sim_top_k]

def generate_submission_file(retriever, test_df, output_csv=f"{results_label}.csv"):
    records = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating Predictions"):
        claim_id = row["claim_id"]
        claim_text = row["claim"]

        retrieved = retriever.retrieve_evidence(claim_text)

        for rank, (abstract_id, _, _) in enumerate(retrieved, start=1):
            records.append({
                "claim_id": claim_id,
                "abstract_id": abstract_id,
                "rank": rank,
            })

    submission_df = pd.DataFrame(records)
    submission_df.to_csv(output_csv, index=False)


retriever = EvidenceRetriever(emb_model_name="xplainlp/e5-large-v2-climatecheck")
retriever.load_corpus("climatecheck_publications_corpus.parquet")

test_df = pd.read_parquet(test_df_path)
test_df = test_df[test_df["claim"].notnull()]

generate_submission_file(retriever, test_df)
print(f"Submission file {results_label} is ready.")


"""Run inference with fine tuned bert-based model"""
import pandas as pd
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "xplainlp/DeBERTa-v3-large-mnli-fever-anli-ling-wanli-climatecheck"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

with open("label_mapping.json") as f:
    label_map = json.load(f)

def predict_label(claim, abstract):
    text = claim + "[SEP]" + abstract
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    return label_map[str(prediction)]

predictions_df = pd.read_csv(f"{results_label}.csv")
test_df = pd.read_parquet(test_df_path)
abstracts_df = pd.read_parquet("climatecheck_publications_corpus.parquet")

claim_map = test_df.set_index("claim_id")["claim"].to_dict()
abstract_map = abstracts_df.set_index("abstract_id")["abstract"].to_dict()

start_time = time.time()

labels = []
for _, row in tqdm(predictions_df.iterrows(), total=len(predictions_df), desc="Predicting labels"):
    claim = claim_map[row["claim_id"]]
    abstract = abstract_map[row["abstract_id"]]
    label = predict_label(claim, abstract)
    labels.append(label)

end_time = time.time()
time_delta_seconds = end_time - start_time
print(f"Time delta (seconds): {time_delta_seconds}")
filename = "time_log_deberta.txt"
with open(filename, "w") as f:
    f.write(f"Start Time: {start_time}\n")
    f.write(f"End Time: {end_time}\n")
    f.write(f"Time Delta (seconds): {time_delta_seconds}\n")

predictions_df["label"] = labels
predictions_df.to_csv(f"{results_label}_both.csv", index=False)
print("Saved predictions with labels to predictions_new.csv")

tracker.stop()