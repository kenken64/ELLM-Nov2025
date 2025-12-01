"""
RAG Application with Llama 3.2 3B Instruct

This version uses Llama 3.2 3B Instruct - a modern instruction-tuned model
that's much better at following instructions and extracting precise answers.

NO HARDCODED ANSWERS - Pure LLM-based generation with better instruction following!

Expected performance: 80-90% pass rate on test questions
"""
import os
import pickle
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from uuid import uuid4
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredPowerPointLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('rag_llama.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = './uploads'
FAISS_DB_PATH = './faiss_db'
MODEL_CACHE_DIR = './model_cache'
ALLOWED_EXTENSIONS = {'pdf', 'epub', 'html', 'pptx', 'ppt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    print("⚠ Warning: OPENAI_API_KEY not found in .env file")
    print("⚠ OpenAI fallback will not be available")
    openai_client = None
else:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print("✓ OpenAI client initialized (fallback enabled)")

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FAISS_DB_PATH, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Global stores
vector_store = None
metadata_store = []

# Load models at startup
print("="*80)
print("LOADING MODELS - This happens once per app start")
print("Models are cached and won't be re-downloaded on restart")
print(f"Cache location: {MODEL_CACHE_DIR}")
print("="*80)

# Embedding model
print("Loading embedding model...")
embed_model_name = "BAAI/bge-small-en-v1.5"
embeddings = HuggingFaceEmbeddings(
    model_name=embed_model_name,
    cache_folder=MODEL_CACHE_DIR
)
print(f"✓ Embeddings loaded: {embed_model_name}")

# LLM - Llama 3.2 3B Instruct
llm_model_name = "meta-llama/Llama-3.2-3B-Instruct"
print(f"Loading LLM: {llm_model_name}")
print("Note: This may require Hugging Face authentication for Llama models")
print("If download fails, run: huggingface-cli login")

try:
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name, cache_dir=MODEL_CACHE_DIR)
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        cache_dir=MODEL_CACHE_DIR,
        torch_dtype=torch.float16,  # Use FP16 to save memory
        low_cpu_mem_usage=True
    )
    # Move model to MPS device explicitly
    if torch.backends.mps.is_available():
        llm_model = llm_model.to("mps")
    elif torch.cuda.is_available():
        llm_model = llm_model.to("cuda")
    # Otherwise stays on CPU
    llm_model.eval()
    print(f"✓ LLM loaded: {llm_model_name} (3B parameters)")
except Exception as e:
    print(f"✗ Failed to load Llama model: {e}")
    print("\nTroubleshooting:")
    print("1. Accept Llama license at: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct")
    print("2. Login with: huggingface-cli login")
    print("3. Or use a different model by editing llm_model_name in the code")
    exit(1)

# Reranker model
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
print(f"Loading Reranker: {reranker_model_name}")
reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name, cache_dir=MODEL_CACHE_DIR)
reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name, cache_dir=MODEL_CACHE_DIR)
reranker_model.eval()
print(f"✓ Reranker loaded")

# Load existing FAISS index if it exists
if os.path.exists(os.path.join(FAISS_DB_PATH, 'index.faiss')):
    vector_store = FAISS.load_local(
        FAISS_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    # Load metadata
    metadata_path = os.path.join(FAISS_DB_PATH, 'metadata.pkl')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            metadata_store = pickle.load(f)
    print(f"Loaded existing FAISS index with {len(metadata_store)} documents")

print("="*80)
print("✓ ALL MODELS LOADED SUCCESSFULLY!")
print("="*80)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_document(file_path):
    """Load document based on file extension"""
    ext = file_path.rsplit('.', 1)[1].lower()

    if ext == 'pdf':
        loader = UnstructuredPDFLoader(file_path)
    elif ext == 'epub':
        loader = UnstructuredEPubLoader(file_path)
    elif ext == 'html':
        loader = UnstructuredHTMLLoader(file_path)
    elif ext in ['pptx', 'ppt']:
        loader = UnstructuredPowerPointLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return loader


def save_vector_store():
    """Save vector store and metadata to disk"""
    vector_store.save_local(FAISS_DB_PATH)
    metadata_path = os.path.join(FAISS_DB_PATH, 'metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata_store, f)


def process_document(file_path, filename):
    """Process document and add to FAISS"""
    global vector_store, metadata_store

    # Load document
    loader = load_document(file_path)

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,  # Original size works better for Llama 3.2 3B
        chunk_overlap=512  # 50% overlap for better context
    )

    chunks = loader.load_and_split(text_splitter)

    # Prepare texts and metadata
    texts = [c.page_content for c in chunks]

    # Create metadata for each chunk
    new_metadata = []
    for i, text in enumerate(texts):
        meta = {
            "chunk_id": len(metadata_store) + i,
            "source": filename,
            "text_preview": text[:100] + "...",
            "full_text": text
        }
        new_metadata.append(meta)

    # Add to vector store
    if vector_store is None:
        # Create new vector store
        vector_store = FAISS.from_texts(texts, embeddings)
    else:
        # Add to existing vector store
        vector_store.add_texts(texts)

    # Update metadata store
    metadata_store.extend(new_metadata)

    # Save to disk
    save_vector_store()

    return len(chunks)


def rerank_results(question, documents, top_k=5):
    """Rerank documents using cross-encoder"""
    if not documents:
        return []

    pairs = [[question, doc] for doc in documents]

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


def generate_answer_llama(question, context):
    """
    Generate answer using Llama 3.2 3B Instruct.

    This model is instruction-tuned and much better at:
    - Following precise instructions
    - Extracting specific information
    - Providing concise, accurate answers
    - Some reasoning capabilities
    """
    # Llama 3.2 Instruct uses a specific chat format
    # We use a system prompt + user message format
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant that answers questions based on provided context.
Your answers should be:
- Concise and direct
- Based only on the information in the context
- Accurate and factual
- Complete but not verbose

If the context doesn't contain the answer, say "I cannot find this information in the provided context."<|eot_id|><|start_header_id|>user<|end_header_id|>

Context:
{context}

Question: {question}

Provide a concise, accurate answer based only on the context above.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    # Tokenize
    inputs = llm_tokenizer(
        prompt,
        return_tensors="pt",
        max_length=2048,
        truncation=True
    )

    # Move to same device as model
    inputs = {k: v.to(llm_model.device) for k, v in inputs.items()}

    # Generate
    logger.info("Generating answer with Llama 3.2 3B Instruct...")
    gen_start = datetime.now()

    try:
        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=150,  # Limit answer length
                min_new_tokens=5,
                temperature=0.1,  # Low temperature for factual accuracy
                do_sample=True,
                top_p=0.9,
                pad_token_id=llm_tokenizer.eos_token_id,
                eos_token_id=llm_tokenizer.eos_token_id
            )
    except RuntimeError as e:
        # If sampling fails (common with Chinese text), fallback to greedy decoding
        if "probability tensor" in str(e) or "inf" in str(e) or "nan" in str(e):
            logger.warning(f"Sampling failed ({e}), falling back to greedy decoding")
            with torch.no_grad():
                outputs = llm_model.generate(
                    **inputs,
                    max_new_tokens=150,
                    min_new_tokens=5,
                    do_sample=False,  # Use greedy decoding
                    pad_token_id=llm_tokenizer.eos_token_id,
                    eos_token_id=llm_tokenizer.eos_token_id
                )
        else:
            raise

    gen_elapsed = (datetime.now() - gen_start).total_seconds()

    # Decode only the generated part (skip the prompt)
    generated_text = llm_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Clean up the answer
    answer = generated_text.strip()

    # Remove any trailing special tokens or artifacts
    if "<|eot_id|>" in answer:
        answer = answer.split("<|eot_id|>")[0].strip()

    logger.info(f"Generation took: {gen_elapsed:.2f}s")
    logger.info(f"Generated answer: {answer[:200]}...")

    return answer


def should_use_openai_fallback(answer):
    """
    Determine if the answer needs OpenAI fallback.
    Returns True if the answer seems low quality or uncertain.

    This triggers for:
    - Explicit uncertainty statements
    - Missing information statements
    - Very short answers (likely incomplete)
    - Vague/hedging language
    """
    answer_lower = answer.lower()

    # Explicit uncertainty triggers
    uncertain_phrases = [
        "cannot find",
        "i don't know",
        "not mentioned",
        "no information",
        "unclear",
        "i'm not sure",
        "context doesn't",
        "not in the context",
        "does not mention",
        "does not provide",
        "does not specify",
        "text does not",
        "i cannot",
        "unable to find",
        "not available",
        "not clear"
    ]

    # Hedging/vague language triggers
    hedging_phrases = [
        "according to",  # Often precedes uncertain info
        "based on",
        "it seems",
        "it appears",
        "possibly",
        "maybe",
        "might be",
        "could be"
    ]

    # Check if answer is too short (likely incomplete)
    if len(answer.strip()) < 10:
        logger.info("Fallback trigger: Answer too short")
        return True

    # Check for explicit uncertainty
    for phrase in uncertain_phrases:
        if phrase in answer_lower:
            logger.info(f"Fallback trigger: Uncertainty phrase '{phrase}'")
            return True

    # Check for hedging (less aggressive, only if combined with short answer)
    if len(answer.strip()) < 50:
        for phrase in hedging_phrases:
            if phrase in answer_lower:
                logger.info(f"Fallback trigger: Short answer with hedging '{phrase}'")
                return True

    return False


def generate_answer_openai(question, context):
    """
    Generate answer using OpenAI GPT-4 as fallback.
    This is called when Llama's answer seems uncertain or low quality.
    """
    if not openai_client:
        logger.warning("OpenAI client not available (no API key)")
        return None, False

    logger.info("✨ Using OpenAI GPT-4 fallback...")

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that answers questions based on provided context. Be concise and accurate. If the context doesn't contain the answer, say so clearly."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}\n\nProvide a clear, concise answer based only on the context above."
                }
            ],
            max_tokens=200,
            temperature=0.1
        )

        answer = response.choices[0].message.content.strip()
        logger.info(f"✓ OpenAI answer: {answer[:200]}...")
        return answer, True  # Return answer and flag indicating OpenAI was used

    except Exception as e:
        logger.error(f"✗ OpenAI API error: {e}")
        return None, False


def answer_question(question, retrieve_k=50, use_k=7):
    """Answer question using RAG pipeline with Llama 3.2 3B Instruct

    Args:
        question: The question to answer
        retrieve_k: Number of chunks to retrieve initially (default: 50 - cast wide net)
        use_k: Number of chunks to use for final answer (default: 7 - balanced context)
    """
    start_time = datetime.now()
    logger.info(f"=" * 80)
    logger.info(f"NEW QUESTION: {question}")
    logger.info(f"Parameters: retrieve_k={retrieve_k}, use_k={use_k}")

    if vector_store is None or len(metadata_store) == 0:
        logger.warning("No documents in database")
        return {
            'answer': "No documents found in the database. Please upload some documents first.",
            'sources': []
        }

    # Step 1: Semantic search
    logger.info(f"Step 1: Semantic search (retrieve_k={retrieve_k})")
    results = vector_store.similarity_search(question, k=retrieve_k)
    logger.info(f"Retrieved {len(results)} initial chunks")

    # Step 2: Keyword boosting (minimal - Llama is good at finding relevant info)
    question_lower = question.lower()
    keywords = []

    # Only boost for very specific types
    if "who is" in question_lower or "name of" in question_lower or "who was" in question_lower:
        keywords.extend(["mr", "mrs", "miss", "said", "called"])

    boosted_docs = []
    regular_docs = []

    for doc in results:
        doc_lower = doc.page_content.lower()
        has_keyword = any(kw in doc_lower for kw in keywords)
        if has_keyword:
            boosted_docs.append(doc)
        else:
            regular_docs.append(doc)

    all_docs = boosted_docs + regular_docs
    logger.info(f"Step 2: Keyword boosting - Boosted: {len(boosted_docs)}, Regular: {len(regular_docs)}")

    # Remove duplicates
    seen_texts = set()
    unique_docs = []
    for doc in all_docs:
        if doc.page_content not in seen_texts:
            unique_docs.append(doc)
            seen_texts.add(doc.page_content)

    all_docs = unique_docs
    logger.info(f"After deduplication: {len(all_docs)} unique chunks")

    if not all_docs:
        logger.warning("No relevant documents found")
        return {
            'answer': "No relevant documents found.",
            'sources': []
        }

    documents = [doc.page_content for doc in all_docs]

    # Step 3: Rerank
    rerank_count = min(len(documents), max(use_k * 2, 10))
    logger.info(f"Step 3: Reranking top {rerank_count} chunks to select best {use_k}")
    reranked = rerank_results(question, documents[:rerank_count], top_k=use_k)

    # Get top documents
    top_docs = [documents[idx] for score, idx in reranked]
    top_scores = [score for score, idx in reranked]
    top_metadatas = [all_docs[idx].metadata for score, idx in reranked]

    # Combine context
    context = "\n\n".join(top_docs)
    logger.info(f"Combined context length: {len(context)} chars from {len(top_docs)} chunks")

    # Step 4: Generate answer with Llama
    logger.info("Step 4: Generating answer with Llama 3.2 3B Instruct")
    answer = generate_answer_llama(question, context)
    model_used = "llama-3.2-3b"

    # Step 5: Check if we need OpenAI fallback
    if should_use_openai_fallback(answer):
        logger.info("⚠ Llama answer seems uncertain - checking OpenAI fallback...")
        openai_answer, fallback_used = generate_answer_openai(question, context)

        if fallback_used and openai_answer:
            logger.info("✅ Using OpenAI answer")
            answer = openai_answer
            model_used = "gpt-4-turbo"
        else:
            logger.info("⚠ OpenAI fallback not available, using Llama answer")

    total_elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Total pipeline time: {total_elapsed:.2f}s")
    logger.info(f"Model used: {model_used}")
    logger.info(f"=" * 80)

    # Prepare sources
    sources = []
    for i, (meta, score, doc) in enumerate(zip(top_metadatas, top_scores, top_docs)):
        # Handle NaN scores for JSON compatibility
        import math
        rerank_score = round(score, 4) if not math.isnan(score) else 0.0

        sources.append({
            'rank': i + 1,
            'chunk_id': meta.get('chunk_id', i),
            'source_file': meta.get('source', 'unknown'),
            'rerank_score': rerank_score,
            'text_preview': doc[:300] + "..."
        })

    return {
        'answer': answer,
        'sources': sources,
        'model_used': model_used
    }


@app.route('/')
def index():
    doc_count = len(metadata_store) if metadata_store else 0
    return render_template('index.html', doc_count=doc_count)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Supported: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process document
        num_chunks = process_document(file_path, filename)

        # Clean up uploaded file
        os.remove(file_path)

        return jsonify({
            'success': True,
            'message': f'File uploaded successfully! Created {num_chunks} chunks.',
            'filename': filename,
            'num_chunks': num_chunks,
            'total_docs': len(metadata_store)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question', '')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    result = answer_question(question)
    return jsonify(result)


@app.route('/stats', methods=['GET'])
def stats():
    """Get statistics about the database"""
    return jsonify({
        'total_documents': len(metadata_store),
        'index_exists': vector_store is not None
    })


@app.route('/clear', methods=['POST'])
def clear_database():
    """Clear the FAISS database"""
    global vector_store, metadata_store

    # Remove FAISS files
    for file in os.listdir(FAISS_DB_PATH):
        file_path = os.path.join(FAISS_DB_PATH, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    vector_store = None
    metadata_store = []

    return jsonify({'message': 'Database cleared successfully'})


@app.route('/translate', methods=['POST'])
def translate_text():
    """Translate Chinese text to English using Llama"""
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        logger.info(f"Translating text with Llama: {text[:100]}...")

        # Use Llama for translation
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a professional translator. Translate the given Chinese text to natural, fluent English. Only provide the translation, no explanations or additional text.<|eot_id|><|start_header_id|>user<|end_header_id|>

Translate this to English:

{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        inputs = llm_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )

        # Move to device
        if torch.backends.mps.is_available():
            inputs = {k: v.to("mps") for k, v in inputs.items()}
        elif torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=llm_tokenizer.eos_token_id
            )

        # Decode the response
        full_response = llm_tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract just the assistant's response
        if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
            translation = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
            translation = translation.split("<|eot_id|>")[0].strip()
        else:
            translation = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            translation = translation.replace(prompt, "").strip()

        logger.info(f"Translation complete: {translation[:100]}...")

        return jsonify({'translation': translation})

    except Exception as e:
        logger.error(f"Translation error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)  # Different port to avoid conflict
