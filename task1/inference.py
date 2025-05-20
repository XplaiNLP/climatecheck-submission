import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
import nltk
from sentence_transformers import SentenceTransformer, CrossEncoder 
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch

nltk.download('punkt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#rer_model = "BAAI/bge-reranker-large"
rer_model = "cross-encoder/ms-marco-MiniLM-L12-v2"
bge_reranker = CrossEncoder(rer_model, device=device)

def rerank_candidates(claim, candidate_abstracts):
    pairs = [(claim, ab) for ab in candidate_abstracts]
    scores = bge_reranker.predict(pairs)
    return np.argsort(scores)[::-1]

results_label = "preds"
class EvidenceRetriever:
    def __init__(self, emb_model_name="intfloat/e5-large-v2", device="cuda"):
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

        self.tokenized_corpus = [nltk.word_tokenize(abstract) for abstract in self.corpus["abstracts"]]
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
        query_embedding = self.embedding_model.encode([query])
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

retriever = EvidenceRetriever()

test_df = pd.read_parquet("test-00000-of-00001.parquet")
test_df = test_df[test_df["claim"].notnull()]

retriever.load_corpus("climatecheck_publications_corpus.parquet")

generate_submission_file(retriever, test_df)
print(f"Submission file {results_label} is ready.")