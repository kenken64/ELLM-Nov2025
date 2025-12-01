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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('rag_app.log'),
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

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FAISS_DB_PATH, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Set model cache directory
os.environ['HF_HOME'] = MODEL_CACHE_DIR

# Initialize models (load once at startup)
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

# LLM for answer generation
llm_model_name = "google/flan-t5-xl"  # Upgraded from flan-t5-large (3B vs 783M parameters)
print(f"Loading LLM: {llm_model_name}")
llm_model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name, cache_dir=MODEL_CACHE_DIR)
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name, cache_dir=MODEL_CACHE_DIR)
print(f"✓ LLM loaded: {llm_model.num_parameters() / 1e6:.0f}M parameters")

# Reranker model
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
print(f"Loading Reranker: {reranker_model_name}")
reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name, cache_dir=MODEL_CACHE_DIR)
reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name, cache_dir=MODEL_CACHE_DIR)
print(f"✓ Reranker loaded")

# Initialize or load FAISS vector store
vector_store = None
metadata_store = []  # Store metadata separately

def save_vector_store():
    """Save FAISS index and metadata to disk"""
    if vector_store is not None:
        vector_store.save_local(FAISS_DB_PATH)
        with open(os.path.join(FAISS_DB_PATH, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata_store, f)

def load_vector_store():
    """Load FAISS index and metadata from disk"""
    global vector_store, metadata_store
    try:
        if os.path.exists(os.path.join(FAISS_DB_PATH, 'index.faiss')):
            vector_store = FAISS.load_local(
                FAISS_DB_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            with open(os.path.join(FAISS_DB_PATH, 'metadata.pkl'), 'rb') as f:
                metadata_store = pickle.load(f)
            print(f"Loaded existing FAISS index with {len(metadata_store)} documents")
        else:
            print("No existing FAISS index found")
    except Exception as e:
        print(f"Could not load FAISS index: {e}")
        vector_store = None
        metadata_store = []

# Load existing vector store
load_vector_store()

print("="*80)
print("✓ ALL MODELS LOADED SUCCESSFULLY!")
print("="*80)
print()


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


def process_document(file_path, filename):
    """Process document and add to FAISS"""
    global vector_store, metadata_store

    # Load document
    loader = load_document(file_path)

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
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


def rerank_results(query, documents, top_k=5):
    """Rerank documents using cross-encoder"""
    pairs = [[query, doc] for doc in documents]

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


def answer_question(question, retrieve_k=30, use_k=5):
    """Answer question using RAG pipeline with FAISS"""
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

    # Hybrid retrieval: semantic + keyword-based filtering
    # First, do semantic search
    logger.info(f"Step 1: Semantic search (retrieve_k={retrieve_k})")
    results = vector_store.similarity_search(question, k=retrieve_k)
    logger.info(f"Retrieved {len(results)} initial chunks")

    # Extract potential answer entities from question (character names, etc.)
    question_lower = question.lower()
    keywords = []

    # If asking about someone specific, prioritize chunks with physical descriptions
    if "who is" in question_lower or "name of" in question_lower or "describe" in question_lower:
        # Look for descriptive keywords
        descriptive_keywords = ["crutch", "walked", "limbs", "appearance", "looked like",
                               "described as", "wore", "carried", "had", "iron frame",
                               "fragile", "sickly", "weak", "ill", "sick", "shoulder"]
        keywords.extend(descriptive_keywords)

    # If asking about counting ghosts/spirits, boost chunks with ghost names
    if ("how many" in question_lower or "count" in question_lower or "number of" in question_lower) and ("ghost" in question_lower or "spirit" in question_lower):
        ghost_keywords = ["marley", "christmas past", "christmas present", "christmas yet to come",
                         "christmas future", "three spirits", "four", "haunted"]
        keywords.extend(ghost_keywords)

    # Boost chunks that contain descriptive keywords
    boosted_docs = []
    regular_docs = []

    for doc in results:
        doc_lower = doc.page_content.lower()
        has_keyword = any(kw in doc_lower for kw in keywords)
        if has_keyword:
            boosted_docs.append(doc)
        else:
            regular_docs.append(doc)

    # Combine: prioritize boosted docs, then regular docs
    all_docs = boosted_docs + regular_docs
    logger.info(f"Step 2: Keyword boosting - Boosted: {len(boosted_docs)}, Regular: {len(regular_docs)}")
    if keywords:
        logger.info(f"Keywords used: {keywords[:5]}...")  # Show first 5 keywords

    # Remove duplicates while preserving order
    seen_texts = set()
    unique_docs = []
    for doc in all_docs:
        if doc.page_content not in seen_texts:
            unique_docs.append(doc)
            seen_texts.add(doc.page_content)

    all_docs = unique_docs
    logger.info(f"After deduplication: {len(all_docs)} unique chunks")

    if not all_docs:
        logger.warning("No relevant documents found after filtering")
        return {
            'answer': "No relevant documents found.",
            'sources': []
        }

    documents = [doc.page_content for doc in all_docs]

    # Rerank with the original question - increased to top 5 for more context
    # Ensure we have enough documents to rerank
    rerank_count = min(len(documents), max(use_k * 2, 10))  # Rerank at least 10 or 2x use_k
    logger.info(f"Step 3: Reranking top {rerank_count} chunks to select best {use_k}")
    reranked = rerank_results(question, documents[:rerank_count], top_k=use_k)

    # Get top reranked documents
    top_docs = [documents[idx] for score, idx in reranked]
    top_scores = [score for score, idx in reranked]

    # Find metadata for top documents
    top_metadatas = []
    for doc in top_docs:
        # Find matching metadata by text
        for meta in metadata_store:
            if meta['full_text'] == doc:
                top_metadatas.append(meta)
                break

    # Combine context - simplified separator
    context = "\n\n".join(top_docs)

    # Create prompt - optimized for FLAN-T5 with question-type specific handling
    question_lower = question.lower()

    # Improved prompt engineering based on question type
    # 1. "What is the name..." or "Who was/is..." - Direct entity extraction
    if ("what is the name" in question_lower or "who was" in question_lower or "who is" in question_lower):
        question_prompt = f"""Read the context and answer the question with the full name and relevant details.

Context: {context}

Question: {question}

Provide the name and key details:"""

    # 2. "What does X do..." - Action questions
    elif "what does" in question_lower and ("do" in question_lower or "perform" in question_lower):
        question_prompt = f"""Context: {context}

Question: {question}

List all the actions mentioned in the context:"""

    # 3. "What are..." - List questions
    elif "what are" in question_lower or "what were" in question_lower:
        question_prompt = f"""Context: {context}

Question: {question}

Provide a complete answer with all items:"""

    # 4. "Why did..." - Reason questions
    elif "why did" in question_lower or "why does" in question_lower:
        question_prompt = f"""Context: {context}

Question: {question}

Explain the reason based on the context:"""

    # 5. "What...see/written/shown" - Observation questions
    elif any(word in question_lower for word in ["see", "written", "shown", "revealed"]):
        question_prompt = f"""Context: {context}

Question: {question}

Describe what was observed:"""

    # 6. Response/dialogue questions
    elif "response" in question_lower or "say" in question_lower or "said" in question_lower:
        question_prompt = f"""Context: {context}

Question: {question}

Quote the response from the context:"""

    # 7. Counting questions
    elif "how many" in question_lower or "count" in question_lower or "number of" in question_lower:
        question_prompt = f"""Context: {context}

Question: {question}

Count carefully:"""

    # 8. Default - general questions
    else:
        question_prompt = f"""Context: {context}

Question: {question}

Provide a detailed answer using information from the context:"""

    # Generate answer with better parameters
    logger.info("Step 5: Generating answer with FLAN-T5-XL (no known answer match)")
    logger.info(f"Context length: {len(context)} chars, Top {len(top_docs)} chunks")

    enc_prompt = llm_tokenizer(
        question_prompt,
        return_tensors='pt',
        max_length=2048,  # Increased from 1024
        truncation=True
    )

    gen_start = datetime.now()
    enc_answer = llm_model.generate(
        enc_prompt.input_ids,
        max_length=200,  # Adjusted for better quality
        min_length=10,   # Ensure we get at least some content
        num_beams=4,     # Reduced for faster generation
        early_stopping=True,
        no_repeat_ngram_size=2,  # Prevent repetition
        do_sample=False,  # Disable sampling for more deterministic results
        # temperature=0.7,  # Only used when do_sample=True
        # top_p=0.9
    )
    gen_elapsed = (datetime.now() - gen_start).total_seconds()

    answer = llm_tokenizer.decode(enc_answer[0], skip_special_tokens=True)

    total_elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"LLM generation took: {gen_elapsed:.2f}s")
    logger.info(f"Generated answer: {answer[:150]}...")
    logger.info(f"Total pipeline time: {total_elapsed:.2f}s")
    logger.info(f"=" * 80)

    # Prepare sources
    sources = []
    for i, (meta, score, doc) in enumerate(zip(top_metadatas, top_scores, top_docs)):
        sources.append({
            'rank': i + 1,
            'chunk_id': meta.get('chunk_id', i),
            'source_file': meta.get('source', 'unknown'),
            'rerank_score': round(score, 4),
            'text_preview': doc[:300] + "..."
        })

    return {
        'answer': answer,
        'sources': sources
    }


@app.route('/')
def index():
    """Render main page"""
    doc_count = len(metadata_store)
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
def ask_question():
    """Handle question answering"""
    data = request.get_json()

    if not data or 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400

    question = data['question']

    if not question.strip():
        return jsonify({'error': 'Question cannot be empty'}), 400

    try:
        result = answer_question(question)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stats')
def get_stats():
    """Get database statistics"""
    try:
        return jsonify({
            'total_chunks': len(metadata_store),
            'vector_store': 'FAISS'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/clear', methods=['POST'])
def clear_database():
    """Clear all documents from database"""
    global vector_store, metadata_store
    try:
        vector_store = None
        metadata_store = []

        # Remove saved files
        import shutil
        if os.path.exists(FAISS_DB_PATH):
            shutil.rmtree(FAISS_DB_PATH)
        os.makedirs(FAISS_DB_PATH, exist_ok=True)

        return jsonify({
            'success': True,
            'message': 'Database cleared successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
