#!/usr/bin/env python3
"""
Re-upload Christmas Carol to rebuild FAISS database with new chunk size
"""
import requests

SERVER_URL = "http://localhost:5001"
EPUB_PATH = "/Users/kennethphang/Projects/ELLM-Nov2025/day03/docs/charles-dickens_a-christmas-carol.epub"

print("Uploading A Christmas Carol with new chunk size (1536)...")
print(f"File: {EPUB_PATH}")

with open(EPUB_PATH, 'rb') as f:
    files = {'file': ('charles-dickens_a-christmas-carol.epub', f, 'application/epub+zip')}
    response = requests.post(f"{SERVER_URL}/upload", files=files, timeout=60)

if response.status_code == 200:
    data = response.json()
    print(f"\n✓ Upload successful!")
    print(f"  Chunks created: {data.get('num_chunks')}")
    print(f"  Total documents in DB: {data.get('total_docs')}")
else:
    print(f"\n✗ Upload failed: {response.status_code}")
    print(response.text)
