#!/usr/bin/env python3
"""
Test Llama 3.2 3B Instruct RAG system
"""
import sys
sys.path.insert(0, '/Users/kennethphang/Projects/ELLM-Nov2025/day03')

from app_llama import answer_question
import time

# Test questions with expected answers
test_cases = [
    {
        "question": "Who was Scrooge's deceased business partner?",
        "expected": "Jacob Marley",
        "key_words": ["Jacob Marley", "Marley"]
    },
    {
        "question": "What is the name of Scrooge's underpaid clerk?",
        "expected": "Bob Cratchit",
        "key_words": ["Bob Cratchit", "Cratchit"]
    },
    {
        "question": "How many ghosts visit Scrooge in total?",
        "expected": "Four",
        "key_words": ["four", "4"]
    },
    {
        "question": "What is the name of Bob Cratchit's youngest son who is ill?",
        "expected": "Tiny Tim",
        "key_words": ["Tiny Tim", "Tim"]
    },
    {
        "question": "Who was Scrooge engaged to in his youth, and why did she leave him?",
        "expected": "Belle",
        "key_words": ["Belle", "money", "love"]
    }
]

print("="*80)
print("TESTING LLAMA 3.2 3B INSTRUCT RAG SYSTEM")
print("="*80)
print()

passed = 0
total = len(test_cases)

for i, test in enumerate(test_cases, 1):
    print(f"\n{'='*80}")
    print(f"TEST {i}/{total}")
    print(f"{'='*80}")
    print(f"Question: {test['question']}")
    print(f"Expected: {test['expected']}")

    start = time.time()
    result = answer_question(test['question'], retrieve_k=30, use_k=5)
    elapsed = time.time() - start

    answer = result['answer']
    print(f"\nActual Answer: {answer}")
    print(f"Time: {elapsed:.2f}s")

    # Check if any key words are in the answer
    answer_lower = answer.lower()
    found_keywords = [kw for kw in test['key_words'] if kw.lower() in answer_lower]

    if found_keywords:
        print(f"✓ PASS - Found keywords: {found_keywords}")
        passed += 1
    else:
        print(f"✗ FAIL - Missing expected keywords: {test['key_words']}")

print(f"\n{'='*80}")
print(f"RESULTS: {passed}/{total} passed ({passed/total*100:.1f}%)")
print(f"{'='*80}")
