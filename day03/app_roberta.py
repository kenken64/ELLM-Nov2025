"""
RAG Application with RoBERTa-SQuAD2 for Extractive Question Answering

This version uses RoBERTa specifically trained on SQuAD 2.0 for extractive QA,
which is much better than FLAN-T5 for finding precise answers in context.

NO HARDCODED ANSWERS - Pure LLM-based extraction!
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
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('rag_roberta.log'),
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

# QA Model - RoBERTa trained on SQuAD 2.0 for extractive QA
qa_model_name = "deepset/roberta-base-squad2"
print(f"Loading QA Model: {qa_model_name}")
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name, cache_dir=MODEL_CACHE_DIR)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name, cache_dir=MODEL_CACHE_DIR)
qa_model.eval()  # Set to evaluation mode
print(f"✓ QA Model loaded: {qa_model_name} (125M parameters)")

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


def extract_answer_roberta(question, context, max_answer_length=100):
    """
    Extract answer using RoBERTa-SQuAD2 model.

    This is extractive QA - it finds the exact span in the context that answers the question.
    No generation, just extraction of existing text.
    """
    # Tokenize input - separate question and context
    inputs = qa_tokenizer(
        question,
        context,
        return_tensors="pt",
        max_length=512,
        truncation="only_second",  # Only truncate context, not question
        padding=True,
        return_offsets_mapping=True  # Get character offsets
    )

    offset_mapping = inputs.pop("offset_mapping")

    # Get model outputs
    with torch.no_grad():
        outputs = qa_model(**inputs)

    # Get start and end positions
    answer_start_idx = torch.argmax(outputs.start_logits)
    answer_end_idx = torch.argmax(outputs.end_logits)

    # Get confidence scores
    start_score = torch.max(outputs.start_logits).item()
    end_score = torch.max(outputs.end_logits).item()
    confidence = (start_score + end_score) / 2

    # Check if answer exists (SQuAD2.0 can say "no answer")
    if answer_start_idx >= answer_end_idx or confidence < 0:
        logger.warning(f"Model could not find answer (confidence: {confidence:.2f})")
        return "I cannot find a clear answer in the provided context."

    # Get character-level offsets in the CONTEXT ONLY
    # The offset_mapping includes [CLS] question [SEP] context [SEP]
    # We need to find where the context starts
    sequence_ids = inputs.sequence_ids(0)

    # Find first token that belongs to context (sequence_id == 1)
    context_start = 0
    for idx, seq_id in enumerate(sequence_ids):
        if seq_id == 1:  # context
            context_start = idx
            break

    # Make sure answer is in the context part, not the question
    if answer_start_idx < context_start:
        answer_start_idx = context_start

    # Extract answer using character offsets
    start_char = offset_mapping[0][answer_start_idx][0].item()
    end_char = offset_mapping[0][answer_end_idx][1].item()

    answer = context[start_char:end_char]

    logger.info(f"Extracted answer: '{answer}' (confidence: {confidence:.2f})")

    return answer.strip()


def answer_question(question, retrieve_k=30, use_k=3):
    """Answer question using RAG pipeline with RoBERTa extractive QA"""
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

    # Step 2: Extract keywords for boosting
    question_lower = question.lower()
    keywords = []

    if "who is" in question_lower or "name of" in question_lower or "who was" in question_lower:
        keywords.extend(["mr", "mrs", "miss", "said", "called", "named"])

    if "how many" in question_lower or "count" in question_lower:
        keywords.extend(["one", "two", "three", "four", "five", "first", "second", "third"])

    # Keyword boosting
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
    rerank_count = min(len(documents), max(use_k * 3, 9))
    logger.info(f"Step 3: Reranking top {rerank_count} chunks to select best {use_k}")
    reranked = rerank_results(question, documents[:rerank_count], top_k=use_k)

    # Get top documents
    top_docs = [documents[idx] for score, idx in reranked]
    top_scores = [score for score, idx in reranked]
    top_metadatas = [all_docs[idx].metadata for score, idx in reranked]

    # Combine context
    context = "\n\n".join(top_docs)
    logger.info(f"Combined context length: {len(context)} chars from {len(top_docs)} chunks")

    # Step 4: Extract answer using RoBERTa
    logger.info("Step 4: Extracting answer with RoBERTa-SQuAD2 (extractive QA)")
    extract_start = datetime.now()

    answer = extract_answer_roberta(question, context)

    extract_elapsed = (datetime.now() - extract_start).total_seconds()
    total_elapsed = (datetime.now() - start_time).total_seconds()

    logger.info(f"Answer extraction took: {extract_elapsed:.2f}s")
    logger.info(f"Final answer: {answer}")
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
    return render_template('index.html')


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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
