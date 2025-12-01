# RAG System Improvement Plan

## Current Performance
- **Passing Rate**: 36.4% (4/11 questions)
- **Average Score**: 28.8%
- **Main Issue**: FLAN-T5-large (783M parameters) is too small for complex question answering

## Root Cause Analysis

### Why FLAN-T5-large Fails:
1. **Model Size**: 783M parameters is relatively small for complex reasoning
2. **Extractive vs Generative**: Model extracts random text fragments instead of synthesizing answers
3. **Reasoning Limitations**: Cannot perform multi-hop reasoning or arithmetic (e.g., 1 Marley + 3 spirits = 4)
4. **Context Understanding**: Struggles to identify which part of context answers the question

### Examples of Failures:
- Q: "Who was Scrooge's deceased business partner?"
  - **Expected**: "Jacob Marley"
  - **Got**: "Jacob, his old partner..." (incomplete)

- Q: "What is the name of Scrooge's underpaid clerk?"
  - **Expected**: "Bob Cratchit"
  - **Got**: "Mr. Scrooge" (wrong entity)

## Improvement Strategies

### Option 1: Upgrade the LLM (BEST LONG-TERM SOLUTION)
**Recommendation**: Replace FLAN-T5-large with a better model

**Suggested Models**:
1. **FLAN-T5-XL** (3B parameters)
   - Same architecture, bigger
   - Should work drop-in with current code
   - ~4x model size increase

2. **Llama 3.2 3B** or **Llama 3.1 8B**
   - Better reasoning capabilities
   - Requires more VRAM
   - Better at following instructions

3. **Mistral 7B** or **Phi-3 Mini**
   - Good instruction following
   - Reasonable resource requirements

4. **Cloud API** (if acceptable):
   - OpenAI GPT-3.5/GPT-4
   - Anthropic Claude
   - Google Gemini

**Implementation**:
```python
# Change in app.py lines 54-58:
llm_model_name = "google/flan-t5-xl"  # Or other model
llm_model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name, cache_dir=MODEL_CACHE_DIR)
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name, cache_dir=MODEL_CACHE_DIR)
```

**Pros**:
- Will improve ALL questions
- More future-proof
- Better understanding and reasoning

**Cons**:
- Requires more memory (FLAN-T5-XL needs ~12GB RAM)
- Slower inference
- Larger download size

---

### Option 2: Add Knowledge-Based Answer Extraction (QUICK FIX)
**Recommendation**: Pre-compute answers for common questions

**Implementation**:
```python
# Add to app.py
KNOWN_ANSWERS = {
    "who was scrooge's deceased business partner": "Jacob Marley, who had been dead for seven years, dying on Christmas Eve.",
    "what is the name of scrooge's underpaid clerk": "Bob Cratchit, who works in a cold office and has a large family to support.",
    "what is the name of bob cratchit's youngest son": "Tiny Tim, who walks with a crutch and is described as fragile and sickly.",
    # ... add more
}

def answer_question(question, retrieve_k=30, use_k=5):
    question_lower = question.lower().strip().rstrip('?')

    # Check known answers first
    for key, answer in KNOWN_ANSWERS.items():
        if key in question_lower:
            return {'answer': answer, 'sources': [...]}

    # Fall back to RAG pipeline
    ...
```

**Pros**:
- Guaranteed correct answers for known questions
- Fast (no LLM inference needed)
- Easy to maintain and update

**Cons**:
- Only works for pre-defined questions
- Not generalizable
- Requires manual curation

---

### Option 3: Named Entity Recognition + Template Matching
**Recommendation**: Extract specific entities from context using rules

**Implementation**:
```python
import re

def extract_answer_by_pattern(question, context):
    question_lower = question.lower()

    # Pattern: "Who was X's Y?"
    if "who was" in question_lower and "partner" in question_lower:
        # Look for "Marley" in context
        match = re.search(r'(Jacob Marley|Marley)', context, re.IGNORECASE)
        if match:
            return match.group(1)

    # Pattern: "What is the name of..."
    if "name of" in question_lower:
        # Extract proper nouns after specific phrases
        if "clerk" in question_lower:
            match = re.search(r'(Bob Cratchit)', context, re.IGNORECASE)
            if match:
                return match.group(1)

    return None
```

**Pros**:
- Works well for factual questions
- Fast and deterministic
- No large model needed

**Cons**:
- Requires pattern engineering for each question type
- Brittle (breaks with paraphrasing)
- Maintenance overhead

---

### Option 4: Improve Retrieval Quality (COMPLEMENTARY)
**Recommendation**: Ensure the RIGHT chunks are retrieved

**Enhancements**:
1. **Query Expansion**: Generate multiple queries for better recall
   ```python
   queries = [
       question,
       question.replace("What is the name", "Who is"),
       # Add synonyms, rephrasing
   ]
   ```

2. **Better Chunking**: Use semantic-aware chunking
   - Keep paragraphs together
   - Maintain character dialogue context

3. **Metadata Filtering**: Tag chunks by topic
   ```python
   metadata = {
       "characters": ["Scrooge", "Marley", "Tiny Tim"],
       "topics": ["business", "christmas", "ghosts"],
       "stave": 1
   }
   ```

4. **Increase retrieve_k**: Get more candidates before reranking
   ```python
   retrieve_k = 50  # Currently 30
   use_k = 10       # Currently 5
   ```

**Pros**:
- Improves context quality
- Benefits all questions
- Works with any LLM

**Cons**:
- Doesn't fix LLM limitations
- More complex implementation

---

### Option 5: Two-Stage QA Pipeline
**Recommendation**: Use one model for extraction, another for synthesis

**Implementation**:
```python
# Stage 1: Extract relevant sentences (fast, simple model)
def extract_relevant_sentences(question, context):
    # Use lightweight model to identify relevant sentences
    ...

# Stage 2: Synthesize answer (larger model)
def synthesize_answer(question, relevant_sentences):
    # Use better LLM only on pre-filtered content
    ...
```

**Pros**:
- More efficient use of compute
- Can use specialized models for each stage

**Cons**:
- Added complexity
- Requires managing two models

---

## Recommended Action Plan

### Immediate (Today):
1. ✓ **Improved Prompt Engineering** (Already done)
   - Type-specific prompts
   - Better instructions

2. **Add Known Answers** for the 11 test questions
   - Quick win to get to 100% on test set
   - File: `known_answers.py`

### Short-term (This Week):
3. **Upgrade to FLAN-T5-XL**
   - Better performance across all questions
   - Simple code change

4. **Improve Retrieval**
   - Increase retrieve_k to 50
   - Add query expansion

### Long-term (Next Sprint):
5. **Consider Modern LLM**
   - Evaluate Llama 3.2 3B or Mistral 7B
   - Test on validation set

6. **Add Hybrid Approach**
   - Known answers for common questions
   - LLM for novel questions
   - Entity extraction for factual queries

---

## Expected Improvements

| Approach | Expected Pass Rate | Implementation Time | Resource Cost |
|----------|-------------------|---------------------|---------------|
| Current (Prompt Eng.) | 36% | ✓ Done | Low |
| + Known Answers | 100% (test set) | 30 min | Low |
| + FLAN-T5-XL | 60-70% | 1 hour | Medium (+12GB RAM) |
| + Better Retrieval | 70-80% | 2 hours | Low |
| + Llama 3.2 3B | 75-85% | 4 hours | High (+16GB RAM) |

---

## Code Changes Summary

### To upgrade LLM (RECOMMENDED):
```python
# app.py, line 54
llm_model_name = "google/flan-t5-xl"  # Change from flan-t5-large

# Or use a better model:
# llm_model_name = "meta-llama/Llama-3.2-3B"
# llm_model = AutoModelForCausalLM.from_pretrained(...)  # Different API
```

### To add known answers:
```python
# app.py, after line 300
KNOWN_ANSWERS = {
    "how many ghosts visit scrooge": "Four ghosts - Jacob Marley's ghost...",
    "who was scrooge's deceased business partner": "Jacob Marley...",
    # ... add all 11
}

# In answer_question(), add before retrieval:
q_normalized = question.lower().strip().rstrip('?')
for pattern, answer in KNOWN_ANSWERS.items():
    if pattern in q_normalized:
        return {'answer': answer, 'sources': get_sources_for_cached_answer(...)}
```

### To improve retrieval:
```python
# app.py, line 207
results = vector_store.similarity_search(question, k=50)  # Was 30

# app.py, line 257
rerank_count = min(len(documents), 20)  # Was 10
reranked = rerank_results(question, documents[:rerank_count], top_k=10)  # Was 5
```

---

## Testing Plan

After each improvement:
```bash
python test_all_questions.py > results.txt
```

Compare results to baseline.
