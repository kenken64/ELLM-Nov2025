#!/usr/bin/env python3
"""
Find chunks that describe Tiny Tim's condition
"""
import pickle
import os

# Load metadata
FAISS_DB_PATH = '../faiss_db'
with open(os.path.join(FAISS_DB_PATH, 'metadata.pkl'), 'rb') as f:
    metadata_store = pickle.load(f)

print("Searching for chunks with 'Tiny Tim' AND ('crutch' OR 'ill' OR 'sick' OR 'fragile')...")
print("="*80)

relevant_chunks = []
for meta in metadata_store:
    text = meta['full_text'].lower()
    if 'tiny tim' in text and any(keyword in text for keyword in ['crutch', 'ill', 'sick', 'fragile', 'weak', 'little child']):
        relevant_chunks.append(meta)

print(f"\nFound {len(relevant_chunks)} relevant chunks\n")

for i, meta in enumerate(relevant_chunks[:5]):
    print(f"\n{'='*80}")
    print(f"CHUNK {i+1} (ID: {meta.get('chunk_id')})")
    print(f"Source: {meta.get('source')}")
    print(f"{'='*80}")
    print(meta['full_text'])
    print()

# Also search for any mention of Bob Cratchit's son
print("\n" + "="*80)
print("Searching for chunks with 'son' and 'Bob Cratchit'...")
print("="*80)

son_chunks = []
for meta in metadata_store:
    text = meta['full_text'].lower()
    if 'son' in text and 'cratchit' in text:
        son_chunks.append(meta)

print(f"\nFound {len(son_chunks)} chunks mentioning son and Cratchit\n")

for i, meta in enumerate(son_chunks[:3]):
    print(f"\n{'='*80}")
    print(f"CHUNK {i+1} (ID: {meta.get('chunk_id')})")
    print(f"{'='*80}")
    print(meta['full_text'])
