#!/usr/bin/env python3
"""
Test all validated questions against the RAG system
"""
import sys
sys.path.insert(0, '/Users/kennethphang/Projects/ELLM-Nov2025/day03')

from app import answer_question
import time

# Define all questions with validated answers
test_cases = [
    {
        "question": "Who was Scrooge's deceased business partner?",
        "expected": "Jacob Marley. He had been dead for seven years at the start of the story, dying on Christmas Eve.",
        "key_points": ["Jacob Marley", "seven years", "Christmas Eve"]
    },
    {
        "question": "What is the name of Scrooge's underpaid clerk?",
        "expected": "Bob Cratchit, who works in a cold office and has a large family to support.",
        "key_points": ["Bob Cratchit", "clerk"]
    },
    {
        "question": "How many ghosts visit Scrooge in total?",
        "expected": "Four ghosts - Jacob Marley's ghost, followed by the Ghost of Christmas Past, the Ghost of Christmas Present, and the Ghost of Christmas Yet to Come (or Future).",
        "key_points": ["Four", "Jacob Marley", "Christmas Past", "Christmas Present", "Christmas Yet to Come"]
    },
    {
        "question": "What is the name of Bob Cratchit's youngest son who is ill?",
        "expected": "Tiny Tim, who walks with a crutch and is described as fragile and sickly.",
        "key_points": ["Tiny Tim", "crutch", "ill"]
    },
    {
        "question": "Who was Scrooge engaged to in his youth, and why did she leave him?",
        "expected": "Belle (or \"Bell\"). She left him because his love of money had replaced his love for her, saying that he feared poverty too much and that a golden idol had displaced her.",
        "key_points": ["Belle", "money", "left"]
    },
    {
        "question": "What does Scrooge see written on the gravestone that frightens him into changing his ways?",
        "expected": "His own name - \"EBENEZER SCROOGE\" - showing him his own lonely, unmourned death.",
        "key_points": ["own name", "Ebenezer Scrooge", "gravestone"]
    },
    {
        "question": "What is Scrooge's response when his nephew Fred invites him to Christmas dinner at the beginning of the story?",
        "expected": "Scrooge responds with \"Bah! Humbug!\" and refuses the invitation, saying Christmas is a \"humbug\" and asking to be left alone.",
        "key_points": ["Bah", "Humbug", "refuses"]
    },
    {
        "question": "What does Scrooge do on Christmas morning after his transformation?",
        "expected": "He sends a prize turkey to the Cratchit family anonymously, raises Bob Cratchit's salary, and goes to his nephew Fred's house for Christmas dinner.",
        "key_points": ["turkey", "Cratchit", "salary", "Fred"]
    },
    {
        "question": "What are the two children shown to Scrooge by the Ghost of Christmas Present, hidden under the ghost's robes?",
        "expected": "Ignorance (a boy) and Want (a girl), representing society's neglected children and social problems.",
        "key_points": ["Ignorance", "Want", "children"]
    },
    {
        "question": "What was Scrooge's first name as a boy, revealed by the Ghost of Christmas Past?",
        "expected": "Ebenezer. The Ghost of Christmas Past shows him scenes from his childhood when he was a lonely boy at boarding school.",
        "key_points": ["Ebenezer"]
    },
    {
        "question": "What specific, generous act does Scrooge perform for the Cratchit family on Christmas morning?",
        "expected": "He buys and sends them a large prize turkey.",
        "key_points": ["turkey", "Cratchit"]
    }
]

def check_answer_quality(answer, key_points):
    """Check if answer contains key points"""
    answer_lower = answer.lower()
    found_points = []
    missing_points = []

    for point in key_points:
        if point.lower() in answer_lower:
            found_points.append(point)
        else:
            missing_points.append(point)

    score = len(found_points) / len(key_points) * 100
    return score, found_points, missing_points

def run_tests():
    """Run all test cases"""
    print("="*80)
    print("TESTING RAG SYSTEM WITH VALIDATED QUESTIONS")
    print("="*80)
    print()

    results = []
    total_score = 0

    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        expected = test_case["expected"]
        key_points = test_case["key_points"]

        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(test_cases)}")
        print(f"{'='*80}")
        print(f"Question: {question}")
        print(f"\nExpected Answer:\n{expected}")

        # Get answer from RAG system
        start_time = time.time()
        result = answer_question(question, retrieve_k=30, use_k=5)
        elapsed_time = time.time() - start_time

        answer = result['answer']

        print(f"\nActual Answer:\n{answer}")

        # Check quality
        score, found, missing = check_answer_quality(answer, key_points)

        print(f"\n--- Evaluation ---")
        print(f"Score: {score:.1f}%")
        print(f"Found key points: {found}")
        if missing:
            print(f"Missing key points: {missing}")
        print(f"Time: {elapsed_time:.2f}s")

        # Determine pass/fail (>= 50% key points = pass)
        passed = score >= 50
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"Status: {status}")

        results.append({
            'question': question,
            'score': score,
            'passed': passed,
            'answer': answer,
            'time': elapsed_time
        })

        total_score += score

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed_count = sum(1 for r in results if r['passed'])
    avg_score = total_score / len(test_cases)
    avg_time = sum(r['time'] for r in results) / len(results)

    print(f"\nTotal Tests: {len(test_cases)}")
    print(f"Passed: {passed_count}/{len(test_cases)} ({passed_count/len(test_cases)*100:.1f}%)")
    print(f"Average Score: {avg_score:.1f}%")
    print(f"Average Time: {avg_time:.2f}s")

    print("\n" + "-"*80)
    print("Individual Results:")
    print("-"*80)

    for i, r in enumerate(results, 1):
        status_symbol = "✓" if r['passed'] else "✗"
        print(f"{status_symbol} Test {i}: {r['score']:.1f}% - {r['question'][:60]}...")

    print("\n" + "="*80)

    # Detailed failures
    failures = [r for r in results if not r['passed']]
    if failures:
        print("\nDETAILED FAILURE ANALYSIS")
        print("="*80)
        for i, failure in enumerate(failures, 1):
            print(f"\nFailure {i}: {failure['question']}")
            print(f"Score: {failure['score']:.1f}%")
            print(f"Answer: {failure['answer'][:200]}...")
            print()

    return results

if __name__ == "__main__":
    results = run_tests()
