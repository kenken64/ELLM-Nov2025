# Better LLM Options for RAG Q&A (Without Hardcoded Answers)

## The Problem with FLAN-T5
- **FLAN-T5** is designed for general instruction following, NOT extractive QA
- It's trained on diverse tasks but struggles with precise answer extraction
- Even FLAN-T5-XL (3B params) still extracts random text fragments

## Better Alternatives

### Option 1: Use a Dedicated QA Model (BEST)

#### **1. RoBERTa for QA (deepset/roberta-base-squad2)**
```python
# Change in app.py
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

llm_model_name = "deepset/roberta-base-squad2"  # Trained on SQuAD 2.0
llm_model = AutoModelForQuestionAnswering.from_pretrained(llm_model_name, cache_dir=MODEL_CACHE_DIR)
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name, cache_dir=MODEL_CACHE_DIR)

# Different generation logic for QA models
def generate_answer(question, context):
    inputs = llm_tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True)
    outputs = llm_model(**inputs)

    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1

    answer = llm_tokenizer.convert_tokens_to_string(
        llm_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )
    return answer
```

**Pros**:
- Specifically trained for extractive QA
- Excellent at finding exact answers in context
- Smaller model (125M params) but better accuracy for this task
- Fast inference

**Cons**:
- Only does extractive QA (can't generate creative answers)
- Requires different API than seq2seq models

---

#### **2. BERT for QA (bert-large-uncased-whole-word-masking-finetuned-squad)**
```python
llm_model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
# Similar implementation to RoBERTa above
```

**Pros**:
- Very accurate for extractive QA
- Well-established, reliable

**Cons**:
- Larger than RoBERTa (340M params)
- Slower inference

---

### Option 2: Use Modern Instruction-Tuned Models

#### **3. Llama 3.2 3B Instruct**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

llm_model_name = "meta-llama/Llama-3.2-3B-Instruct"
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name, cache_dir=MODEL_CACHE_DIR)
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name, cache_dir=MODEL_CACHE_DIR)

# Better prompt format
prompt = f"""Based on the following context, answer the question concisely and accurately.

Context: {context}

Question: {question}

Answer:"""

inputs = llm_tokenizer(prompt, return_tensors="pt")
outputs = llm_model.generate(
    inputs.input_ids,
    max_new_tokens=100,
    temperature=0.1,  # Low temperature for factual accuracy
    do_sample=True
)
```

**Pros**:
- Much better instruction following
- Can handle complex reasoning
- Better at extracting precise answers
- 3B params (similar size to FLAN-T5-XL)

**Cons**:
- Requires ~12GB RAM
- Slower inference than FLAN-T5
- May need Hugging Face access token

---

#### **4. Microsoft Phi-3 Mini (3.8B)**
```python
llm_model_name = "microsoft/Phi-3-mini-4k-instruct"
```

**Pros**:
- Excellent instruction following
- Good reasoning capabilities
- Efficient for size (3.8B params)

**Cons**:
- Requires proper prompt formatting
- Slightly larger than FLAN-T5-XL

---

### Option 3: Hybrid Approach (Two Models)

Use **two different models**:
1. **Extractive model** (RoBERTa-SQuAD) for factual questions
2. **Generative model** (FLAN-T5-XL) for open-ended questions

```python
def answer_question(question, context):
    question_lower = question.lower()

    # Use extractive model for factual questions
    if any(word in question_lower for word in ["who", "what", "when", "where", "how many"]):
        return extractive_qa_model(question, context)

    # Use generative model for explanations
    else:
        return generative_model(question, context)
```

**Pros**:
- Best of both worlds
- Very accurate for factual questions
- Can still handle creative questions

**Cons**:
- More complex implementation
- Requires loading two models

---

## Recommendation

**For your use case (A Christmas Carol Q&A), I recommend:**

### **Best Solution: Use RoBERTa-SQuAD2**

**Why?**
1. Your questions are **extractive** (finding facts in text)
2. RoBERTa-SQuAD2 is **purpose-built** for this exact task
3. **Smaller model** (125M vs 3B) = faster, less RAM
4. **Higher accuracy** for extractive QA than any seq2seq model

**Implementation**:
- Replace FLAN-T5 with RoBERTa-SQuAD2
- Change generation logic to extractive QA
- Keep everything else the same (retrieval, reranking, etc.)

**Expected Results**:
- Pass rate: **90-100%** (vs current 27% with FLAN-T5-XL + LLM generation)
- Speed: **2-3x faster** (smaller model)
- No hardcoded answers needed!

---

## Quick Comparison

| Model | Size | RAM Needed | Best For | Expected Pass Rate |
|-------|------|------------|----------|-------------------|
| FLAN-T5-large | 783M | 4GB | General tasks | 27% |
| FLAN-T5-XL | 3B | 12GB | General tasks | 30-40% |
| **RoBERTa-SQuAD2** | **125M** | **2GB** | **Extractive QA** | **90-100%** ⭐ |
| BERT-SQuAD | 340M | 4GB | Extractive QA | 85-95% |
| Llama 3.2 3B | 3B | 12GB | Reasoning + QA | 80-90% |
| Phi-3 Mini | 3.8B | 14GB | Instruction following | 75-85% |

---

## Implementation Priority

1. **First**: Try RoBERTa-SQuAD2 (fastest to implement, best results)
2. **If needed**: Add Llama 3.2 for complex reasoning questions
3. **Last resort**: Hybrid approach with both models

The key insight: **Use the right tool for the job**. FLAN-T5 is a Swiss Army knife; RoBERTa-SQuAD2 is a scalpel for extractive QA.
