#!/usr/bin/env python3
"""
Test the multi-query improved RAG pipeline
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

# Multi-query retrieval
related_queries = [question]

if "who is" in question.lower() or "name of" in question.lower():
    words = question.split()
    for i, word in enumerate(words):
        if word[0].isupper() and len(word) > 2:
            related_queries.append(f"Describe {word} and their physical appearance")
            related_queries.append(f"{word} character description")

print("\nRelated queries generated:")
for i, q in enumerate(related_queries):
    print(f"{i+1}. {q}")

# Retrieve chunks using multiple queries
all_docs = []
seen_texts = set()
retrieve_k = 30

print("\n" + "="*80)
print("Multi-Query Retrieval")
print("="*80)

for query in related_queries:
    print(f"\nQuery: {query}")
    results = vector_store.similarity_search(query, k=retrieve_k)
    new_count = 0
    for doc in results:
        if doc.page_content not in seen_texts:
            all_docs.append(doc)
            seen_texts.add(doc.page_content)
            new_count += 1
    print(f"  Retrieved {len(results)} chunks, {new_count} new unique chunks")

print(f"\nTotal unique chunks collected: {len(all_docs)}")

documents = [doc.page_content for doc in all_docs]

# Rerank
print("\n" + "="*80)
print("Reranking (use_k=5)")
print("="*80)

reranked = rerank_results(question, documents, top_k=5)

top_docs = [documents[idx] for score, idx in reranked]

for i, (score, idx) in enumerate(reranked):
    print(f"\n--- Reranked Position {i+1} (Score: {score:.4f}) ---")
    chunk = documents[idx]
    print(chunk[:300] + "...")
    if "crutch" in chunk.lower():
        print("✓✓✓ CONTAINS 'CRUTCH' ✓✓✓")
    if "tiny tim" in chunk.lower():
        print("✓ Contains 'Tiny Tim'")

# Context formation
context = "\n\n".join(top_docs)

print("\n" + "="*80)
print("Context Analysis")
print("="*80)
print(f"Context length: {len(context)} characters")

if "crutch" in context.lower():
    print("✓ 'crutch' IS in the context")
else:
    print("✗ 'crutch' is NOT in the context")

if "bore a little crutch" in context.lower():
    print("✓✓✓ The chunk describing Tiny Tim with a crutch IS in the context!")
else:
    print("✗ The specific 'bore a little crutch' phrase is not in context")

# Generate answer
print("\n" + "="*80)
print("Generating Answer")
print("="*80)

question_prompt = f"""Context: {context}

Question: {question}

Provide a detailed answer using information from the context:"""

enc_prompt = llm_tokenizer(
    question_prompt,
    return_tensors='pt',
    max_length=2048,
    truncation=True
)

print(f"Tokenized prompt length: {enc_prompt.input_ids.shape[1]} tokens")

enc_answer = llm_model.generate(
    enc_prompt.input_ids,
    max_length=250,
    num_beams=5,
    early_stopping=True,
    no_repeat_ngram_size=3,
    temperature=0.8,
    top_p=0.95
)

answer = llm_tokenizer.decode(enc_answer[0], skip_special_tokens=True)

print("\n" + "="*80)
print("FINAL ANSWER:")
print("="*80)
print(answer)
print("\n" + "="*80)
