#!/usr/bin/env python3
"""
Test script to inspect FAISS database and test retrieval
"""
import pickle
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configuration
FAISS_DB_PATH = '../faiss_db'
MODEL_CACHE_DIR = './model_cache'

os.environ['HF_HOME'] = MODEL_CACHE_DIR

print("Loading embeddings model...")
embed_model_name = "BAAI/bge-small-en-v1.5"
embeddings = HuggingFaceEmbeddings(
    model_name=embed_model_name,
    cache_folder=MODEL_CACHE_DIR
)

print("Loading FAISS vector store...")
vector_store = FAISS.load_local(
    FAISS_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

print("Loading metadata...")
with open(os.path.join(FAISS_DB_PATH, 'metadata.pkl'), 'rb') as f:
    metadata_store = pickle.load(f)

print(f"\nTotal chunks in database: {len(metadata_store)}")
print(f"\nFirst 3 chunks preview:")
for i, meta in enumerate(metadata_store[:3]):
    print(f"\n--- Chunk {i} ---")
    print(f"Source: {meta.get('source', 'unknown')}")
    print(f"Preview: {meta.get('text_preview', '')}")

# Test the exact query
print("\n" + "="*80)
print("TESTING QUERY: 'What is the name of Bob Cratchit's youngest son who is ill?'")
print("="*80)

question = "What is the name of Bob Cratchit's youngest son who is ill?"

print("\n1. FAISS Similarity Search (top 10):")
results = vector_store.similarity_search(question, k=10)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Content preview: {doc.page_content[:200]}...")
    if "Tiny Tim" in doc.page_content or "Bob Cratchit" in doc.page_content:
        print("✓ CONTAINS RELEVANT INFO!")

print("\n" + "="*80)
print("Searching for chunks containing 'Tiny Tim':")
print("="*80)

tiny_tim_chunks = [meta for meta in metadata_store if 'Tiny Tim' in meta['full_text']]
print(f"\nFound {len(tiny_tim_chunks)} chunks containing 'Tiny Tim'")

for i, meta in enumerate(tiny_tim_chunks[:3]):
    print(f"\n--- Tiny Tim Chunk {i+1} ---")
    print(f"Chunk ID: {meta.get('chunk_id')}")
    print(f"Source: {meta.get('source')}")
    print(f"Full text: {meta['full_text'][:300]}...")

print("\n" + "="*80)
print("Searching for chunks containing 'Bob Cratchit':")
print("="*80)

bob_chunks = [meta for meta in metadata_store if 'Bob Cratchit' in meta['full_text']]
print(f"\nFound {len(bob_chunks)} chunks containing 'Bob Cratchit'")

for i, meta in enumerate(bob_chunks[:3]):
    print(f"\n--- Bob Cratchit Chunk {i+1} ---")
    print(f"Chunk ID: {meta.get('chunk_id')}")
    print(f"Source: {meta.get('source')}")
    print(f"Full text: {meta['full_text'][:300]}...")
