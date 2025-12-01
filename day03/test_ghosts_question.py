#!/usr/bin/env python3
"""
Test the ghosts question to see what answer we get
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
question = "How many ghosts visit Scrooge in total?"

print("\n" + "="*80)
print(f"QUESTION: {question}")
print("="*80)

# Hybrid retrieval
retrieve_k = 30
results = vector_store.similarity_search(question, k=retrieve_k)

print(f"\nRetrieved {len(results)} chunks from FAISS")

# Keyword boosting (currently only for "who is" and "name of" questions)
question_lower = question.lower()
keywords = []

if "who is" in question_lower or "name of" in question_lower or "describe" in question_lower:
    descriptive_keywords = ["crutch", "walked", "limbs", "appearance", "looked like",
                           "described as", "wore", "carried", "had", "iron frame",
                           "fragile", "sickly", "weak", "ill", "sick", "shoulder"]
    keywords.extend(descriptive_keywords)

if keywords:
    print(f"\nKeyword boost active with: {keywords[:5]}...")

    # Boost chunks
    boosted_docs = []
    regular_docs = []

    for doc in results:
        doc_lower = doc.page_content.lower()
        has_keyword = any(kw in doc_lower for kw in keywords)
        if has_keyword:
            boosted_docs.append(doc)
        else:
            regular_docs.append(doc)

    print(f"\nBoosted chunks: {len(boosted_docs)}")
    print(f"Regular chunks: {len(regular_docs)}")

    # Combine and deduplicate
    all_docs = boosted_docs + regular_docs
    seen_texts = set()
    unique_docs = []
    for doc in all_docs:
        if doc.page_content not in seen_texts:
            unique_docs.append(doc)
            seen_texts.add(doc.page_content)

    all_docs = unique_docs
else:
    print("\nNo keyword boosting for this question type")
    all_docs = results

documents = [doc.page_content for doc in all_docs]

# Rerank
print("\n" + "="*80)
print("Reranking (use_k=5)")
print("="*80)

rerank_count = min(len(documents), max(5 * 2, 10))
reranked = rerank_results(question, documents[:rerank_count], top_k=5)

top_docs = [documents[idx] for score, idx in reranked]

for i, (score, idx) in enumerate(reranked):
    print(f"\n--- Reranked Position {i+1} (Score: {score:.4f}) ---")
    chunk = documents[idx]
    print(chunk[:400] + "...")

    # Check for ghost mentions
    if "ghost" in chunk.lower():
        ghost_count = chunk.lower().count("ghost")
        print(f"✓ Contains 'ghost' {ghost_count} times")
    if "marley" in chunk.lower():
        print("✓ Contains 'Marley'")
    if "christmas past" in chunk.lower():
        print("✓ Contains 'Christmas Past'")
    if "christmas present" in chunk.lower():
        print("✓ Contains 'Christmas Present'")
    if "christmas yet to come" in chunk.lower() or "christmas future" in chunk.lower():
        print("✓ Contains 'Christmas Yet to Come' or 'Future'")

# Context formation
context = "\n\n".join(top_docs)

print("\n" + "="*80)
print("Context Analysis")
print("="*80)
print(f"Context length: {len(context)} characters")

ghost_mentions = context.lower().count("ghost")
print(f"\nTotal 'ghost' mentions in context: {ghost_mentions}")

if "marley" in context.lower():
    print("✓ Marley is in the context")
if "christmas past" in context.lower():
    print("✓ Christmas Past is in the context")
if "christmas present" in context.lower():
    print("✓ Christmas Present is in the context")
if "christmas yet to come" in context.lower() or "christmas future" in context.lower():
    print("✓ Christmas Yet to Come/Future is in the context")

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
    do_sample=True,
    temperature=0.8,
    top_p=0.95
)

answer = llm_tokenizer.decode(enc_answer[0], skip_special_tokens=True)

print("\n" + "="*80)
print("FINAL ANSWER:")
print("="*80)
print(answer)
print("\n" + "="*80)

print("\nEXPECTED ANSWER:")
print("Four ghosts - Jacob Marley's ghost, followed by the Ghost of Christmas Past,")
print("the Ghost of Christmas Present, and the Ghost of Christmas Yet to Come (or Future).")
print("="*80)
