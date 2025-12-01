#!/usr/bin/env python3
"""
Find chunks that mention all the ghosts
"""
import pickle
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configuration
FAISS_DB_PATH = '../faiss_db'
MODEL_CACHE_DIR = './model_cache'

os.environ['HF_HOME'] = MODEL_CACHE_DIR

# Embedding model
embed_model_name = "BAAI/bge-small-en-v1.5"
embeddings = HuggingFaceEmbeddings(
    model_name=embed_model_name,
    cache_folder=MODEL_CACHE_DIR
)

# Load vector store
vector_store = FAISS.load_local(
    FAISS_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

with open(os.path.join(FAISS_DB_PATH, 'metadata.pkl'), 'rb') as f:
    metadata_store = pickle.load(f)

print("="*80)
print("Searching for chunks that mention Jacob Marley")
print("="*80)

marley_chunks = []
for i, meta in enumerate(metadata_store):
    text = meta['full_text']
    text_lower = text.lower()

    if "marley" in text_lower and "ghost" in text_lower:
        marley_chunks.append((i, meta['chunk_id'], text))

print(f"\nFound {len(marley_chunks)} chunks with Marley + ghost")

print("\n" + "="*80)
print("Chunks that mention counting or listing ghosts:")
print("="*80)

for i, chunk_id, text in marley_chunks:
    text_lower = text.lower()
    if any(keyword in text_lower for keyword in ["three", "four", "visit", "haunted", "spirits"]):
        print(f"\n--- Chunk {chunk_id} ---")
        print(text[:500])
        print("...")
        if "three" in text_lower:
            print("✓ Contains 'three'")
        if "four" in text_lower:
            print("✓ Contains 'four'")
        if "visit" in text_lower or "haunted" in text_lower:
            print("✓ Contains 'visit' or 'haunted'")

print("\n" + "="*80)
print("Searching for chunks mentioning all ghost names:")
print("="*80)

ghost_names = ["marley", "christmas past", "christmas present", "christmas yet to come", "christmas future"]

for i, meta in enumerate(metadata_store):
    text = meta['full_text']
    text_lower = text.lower()

    ghost_count = sum(1 for name in ghost_names if name in text_lower)

    if ghost_count >= 2:
        print(f"\n--- Chunk {meta['chunk_id']} (mentions {ghost_count} ghosts) ---")
        print(text[:500])
        print("...")
        for name in ghost_names:
            if name in text_lower:
                print(f"✓ Contains '{name}'")
