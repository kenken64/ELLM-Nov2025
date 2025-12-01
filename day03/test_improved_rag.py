#!/usr/bin/env python3
"""
Test the improved RAG pipeline
"""
import pickle
import os
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification

# Configuration
FAISS_DB_PATH = '../faiss_db'
MODEL_CACHE_DIR = './model_cache'

os.environ['HF_HOME'] = MODEL_CACHE_DIR

print("Loading models...")

# Embedding model
embed_model_name = "BAAI/bge-small-en-v1.5"
embeddings = HuggingFaceEmbeddings(
    model_name=embed_model_name,
    cache_folder=MODEL_CACHE_DIR
)

# LLM
llm_model_name = "google/flan-t5-large"
llm_model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name, cache_dir=MODEL_CACHE_DIR)
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name, cache_dir=MODEL_CACHE_DIR)

# Reranker
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name, cache_dir=MODEL_CACHE_DIR)
reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name, cache_dir=MODEL_CACHE_DIR)

# Load vector store
vector_store = FAISS.load_local(
    FAISS_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

with open(os.path.join(FAISS_DB_PATH, 'metadata.pkl'), 'rb') as f:
    metadata_store = pickle.load(f)

def rerank_results(query, documents, top_k=5):
    """Rerank documents using cross-encoder"""
    pairs = [[query, doc] for doc in documents]

    inputs = reranker_tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )

    with torch.no_grad():
        scores = reranker_model(**inputs).logits.squeeze(-1)

    scored_docs = [(score.item(), i) for i, score in enumerate(scores)]
    scored_docs.sort(reverse=True, key=lambda x: x[0])

    return scored_docs[:top_k]

# Test query
question = "What is the name of Bob Cratchit's youngest son who is ill?"

print("\n" + "="*80)
print(f"QUESTION: {question}")
print("="*80)

# Step 1: FAISS retrieval - IMPROVED: retrieve_k=20
print("\nStep 1: FAISS Similarity Search (retrieve_k=20)")
results = vector_store.similarity_search(question, k=20)
documents = [doc.page_content for doc in results]

print(f"Retrieved {len(documents)} chunks")

# Step 2: Reranking - IMPROVED: use_k=3
print("\n" + "="*80)
print("Step 2: Reranking (use_k=3)")
print("="*80)

reranked = rerank_results(question, documents, top_k=3)

top_docs = [documents[idx] for score, idx in reranked]
top_scores = [score for score, idx in reranked]

for i, (score, idx) in enumerate(reranked):
    print(f"\n--- Reranked Position {i+1} (Score: {score:.4f}) ---")
    print(documents[idx][:300] + "...")

# Step 3: Context formation - IMPROVED: simpler separator
print("\n" + "="*80)
print("Step 3: Forming Context")
print("="*80)

context = "\n\n".join(top_docs)
print(f"\nContext length: {len(context)} characters")
print(f"\nFull context:")
print("="*80)
print(context)
print("="*80)

# Step 4: Create prompt - IMPROVED: simpler format
print("\n" + "="*80)
print("Step 4: Creating Prompt")
print("="*80)

question_prompt = f"""Context: {context}

Question: {question}

Answer:"""

print(f"Prompt length: {len(question_prompt)} characters")

# Step 5: Generate answer - IMPROVED: better params
print("\n" + "="*80)
print("Step 5: Generating Answer")
print("="*80)

enc_prompt = llm_tokenizer(
    question_prompt,
    return_tensors='pt',
    max_length=2048,  # Increased
    truncation=True
)

print(f"Tokenized prompt length: {enc_prompt.input_ids.shape[1]} tokens")

enc_answer = llm_model.generate(
    enc_prompt.input_ids,
    max_length=200,
    num_beams=5,
    early_stopping=True,
    no_repeat_ngram_size=3,
    temperature=0.7,
    top_p=0.9
)

answer = llm_tokenizer.decode(enc_answer[0], skip_special_tokens=True)

print("\n" + "="*80)
print("FINAL ANSWER:")
print("="*80)
print(answer)
print("\n" + "="*80)

# Check if Tiny Tim is in the context
if "Tiny Tim" in context:
    print("✓ 'Tiny Tim' IS in the context")
    # Count occurrences
    count = context.count("Tiny Tim")
    print(f"  Mentioned {count} times")
else:
    print("✗ 'Tiny Tim' is NOT in the context")

if "crutch" in context:
    print("✓ 'crutch' IS in the context")
else:
    print("✗ 'crutch' is NOT in the context")

# Check for the critical chunk
if "bore a little crutch" in context:
    print("✓ The chunk describing Tiny Tim with a crutch IS in the context")
else:
    print("✗ The chunk describing Tiny Tim with a crutch is NOT in the context")
