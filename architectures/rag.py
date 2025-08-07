import time
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Load models globally to avoid reloading on each call
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def run(question: str, chunks: List[str]) -> Tuple[str, List[str], Dict]:
    start_time = time.time()

    # Embed chunks
    chunk_embeddings = embedding_model.encode(chunks, convert_to_tensor=True)
    
    # Embed the question
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)

    # Compute cosine similarity
    scores = util.cos_sim(question_embedding, chunk_embeddings)[0]

    # Get top 3 passages
    top_k = min(3, len(chunks))
    top_results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

    top_passages = [chunks[i] for i, _ in top_results]

    # Use QA pipeline on the top passage (best match)
    answer_result = qa_pipeline(question=question, context=top_passages[0])
    answer = answer_result["answer"]

    latency = round((time.time() - start_time) * 1000, 2)  # in ms

    metrics = {
        "Similarity Score (Top)": round(float(top_results[0][1]), 4),
        "Latency (ms)": latency,
        "Retrieved Passages": top_k
    }

    return answer, top_passages, metrics
