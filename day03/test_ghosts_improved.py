#!/usr/bin/env python3
"""
Test the improved ghost counting logic
"""
import sys
sys.path.insert(0, '/Users/kennethphang/Projects/ELLM-Nov2025/day03')

from app import answer_question

# Test the question
question = "How many ghosts visit Scrooge in total?"

print("\n" + "="*80)
print(f"QUESTION: {question}")
print("="*80)

result = answer_question(question, retrieve_k=30, use_k=5)

print("\n" + "="*80)
print("ANSWER:")
print("="*80)
print(result['answer'])

print("\n" + "="*80)
print("SOURCES:")
print("="*80)
for source in result['sources']:
    print(f"\nRank {source['rank']} (Score: {source['rerank_score']}):")
    print(f"  File: {source['source_file']}")
    print(f"  Preview: {source['text_preview'][:200]}...")

print("\n" + "="*80)
print("EXPECTED ANSWER:")
print("Four ghosts - Jacob Marley's ghost, followed by the Ghost of Christmas Past,")
print("the Ghost of Christmas Present, and the Ghost of Christmas Yet to Come (or Future).")
print("="*80)
