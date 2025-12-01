# RAG Document Q&A Web Application

A web application that allows users to upload documents (PDF, EPUB, HTML, PPTX) and ask questions about their content using an improved RAG (Retrieval-Augmented Generation) system.

## Features

- **Multi-format Support**: Upload PDF, EPUB, HTML, and PowerPoint files
- **Persistent Storage**: Documents are stored in ChromaDB and persist across sessions
- **Advanced RAG Pipeline**:
  - Better chunking with 50% overlap (512/1024 characters)
  - Retrieves 10 chunks initially
  - Reranks using cross-encoder for better relevance
  - Uses top 5 reranked chunks for answer generation
  - Source attribution showing which chunks were used
- **Improved Answer Quality**:
  - FLAN-T5-Large model (780M parameters)
  - Cross-encoder reranking for better relevance
  - Handles "unknown" cases gracefully
- **User-Friendly Interface**:
  - Drag-and-drop file upload
  - Real-time question answering
  - Source citations with relevance scores

## Architecture

Based on the improved RAG system from `day03-rag-v3-improved.ipynb`:

1. **Document Processing**: Files are split into 1024-character chunks with 512-character overlap
2. **Embeddings**: Uses BAAI/bge-small-en-v1.5 for semantic embeddings
3. **Retrieval**: ChromaDB with persistent storage
4. **Reranking**: cross-encoder/ms-marco-MiniLM-L-6-v2 for better relevance
5. **Generation**: google/flan-t5-large for answer generation

## Installation

### Prerequisites

- **Python 3.9 - 3.12** (Python 3.13+ not yet supported by dependencies)
- pip

**Important**: If you have Python 3.13, you'll need to install Python 3.12:
- macOS: `brew install python@3.12`
- Or download from [python.org](https://www.python.org/downloads/)

### Quick Start (Recommended)

Use the provided startup script that handles everything automatically:

```bash
./start.sh
```

This script will:
- Create a virtual environment if needed
- Install/update all dependencies
- Create necessary directories
- Start the Flask application

### Manual Setup

If you prefer manual setup:

1. Clone or navigate to the project directory:
```bash
cd /Users/kennethphang/Projects/ELLM-Nov2025/day03
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to:
```
http://localhost:5000
```

### First Run vs Subsequent Runs

**First Run** (~5-10 minutes):
- Downloads ~4GB of AI models from Hugging Face
- Models are cached to `./model_cache/` directory
- You'll see progress bars for each model download

**Subsequent Runs** (~30-60 seconds):
- Models load from cache (no download)
- Much faster startup
- Models are loaded into memory each time but NOT re-downloaded

### Model Caching

Models are automatically cached in the `./model_cache/` directory. This means:
- First run downloads models once
- Every restart after that loads from cache
- No internet required after first run
- You can backup/share the `model_cache` folder to avoid re-downloading

To use the default Hugging Face cache location (`~/.cache/huggingface`) instead:
1. Open `app.py`
2. Comment out line 35: `# os.environ['HF_HOME'] = MODEL_CACHE_DIR`

## Usage

### Uploading Documents

1. Click the upload area or drag and drop a file
2. Supported formats: PDF, EPUB, HTML, PPTX
3. The file will be automatically processed and chunked
4. Progress will be shown with the number of chunks created

### Asking Questions

1. Type your question in the input field
2. Click "Ask Question" or press Enter
3. The system will:
   - Retrieve relevant chunks from all uploaded documents
   - Rerank them for relevance
   - Generate an answer using the most relevant context
   - Show source citations with relevance scores

### Managing Database

- **Document Count**: Displayed at the top showing total chunks in database
- **Clear Database**: Button to remove all documents (requires confirmation)

## API Endpoints

### `POST /upload`
Upload and process a document file.

**Request**: multipart/form-data with file
**Response**:
```json
{
  "success": true,
  "message": "File uploaded successfully! Created 274 chunks.",
  "filename": "document.pdf",
  "num_chunks": 274,
  "total_docs": 274
}
```

### `POST /ask`
Ask a question about uploaded documents.

**Request**:
```json
{
  "question": "What happened to Marley?"
}
```

**Response**:
```json
{
  "answer": "Marley was dead.",
  "sources": [
    {
      "rank": 1,
      "chunk_id": 5,
      "doc_id": "abc123",
      "source_file": "document.pdf",
      "rerank_score": 0.8542,
      "text_preview": "Marley was dead, to begin with..."
    }
  ]
}
```

### `GET /stats`
Get database statistics.

**Response**:
```json
{
  "total_chunks": 274,
  "collection_name": "documents"
}
```

### `POST /clear`
Clear all documents from the database.

**Response**:
```json
{
  "success": true,
  "message": "Database cleared successfully"
}
```

## Configuration

Edit `app.py` to modify:

- `UPLOAD_FOLDER`: Directory for temporary file storage (default: `./uploads`)
- `CHROMA_DB_PATH`: ChromaDB persistence path (default: `./chroma_db`)
- `MODEL_CACHE_DIR`: Model cache directory (default: `./model_cache`)
- `MAX_CONTENT_LENGTH`: Maximum file size (default: 50MB)
- `ALLOWED_EXTENSIONS`: Supported file types
- Model parameters in `answer_question()`:
  - `retrieve_k`: Number of chunks to retrieve (default: 10)
  - `use_k`: Number of chunks to use after reranking (default: 5)

## Improvements Over Basic RAG

This implementation includes all improvements from the notebook:

1. **Skip Reformulation** - Direct search (simpler, faster)
2. **Increased Context** - 10 chunks retrieved, 5 used after reranking
3. **Better Chunking** - 512-character overlap (50%)
4. **Persistent Storage** - ChromaDB PersistentClient
5. **Improved Prompt** - Instructions for handling unknown cases
6. **Larger Model** - FLAN-T5-large (780M vs 250M)
7. **Reranking** - Cross-encoder for better relevance
8. **Source Attribution** - Chunk IDs and metadata with answers

## Troubleshooting

### Models downloading slowly
On first run, the application will download several models (~4GB total). This is a one-time process. The models include:
- `BAAI/bge-small-en-v1.5` (~130MB) - Embeddings
- `google/flan-t5-large` (~3GB) - Answer generation
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90MB) - Reranking

After first download, models are cached in `./model_cache/` and won't be re-downloaded.

### Models are being re-downloaded every time
This shouldn't happen. Check:
- The `./model_cache/` directory exists and has files
- You haven't deleted the cache folder
- You have write permissions in the project directory

### Out of memory
If you encounter memory issues:
- Reduce `retrieve_k` and `use_k` in the `answer_question()` function
- Use a smaller model (change `google/flan-t5-large` to `google/flan-t5-base`)
- Close other applications to free up RAM

### Slow startup even with cached models
The models still need to be loaded into memory each time (30-60 seconds). This is normal. Only the download is cached, not the in-memory loading.

### File upload fails
- Check file size (must be under 50MB)
- Verify file format is supported
- Check server logs for detailed error messages

## Project Structure

```
day03/
├── app.py                          # Flask application
├── start.sh                        # Startup script (recommended)
├── templates/
│   └── index.html                  # Frontend interface
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── venv/                           # Virtual environment (auto-created)
├── uploads/                        # Temporary file storage (auto-created)
├── chroma_db/                      # Persistent vector database (auto-created)
└── model_cache/                    # Cached AI models (auto-created, ~4GB)
```

## Technologies Used

- **Flask**: Web framework
- **ChromaDB**: Vector database for embeddings
- **LangChain**: Document loaders and text splitting
- **Transformers**: LLM models (FLAN-T5, BGE embeddings, cross-encoder)
- **PyTorch**: Model inference
- **Unstructured**: Document parsing (PDF, EPUB, HTML, PPTX)

## License

MIT License

## Credits

Based on the improved RAG system from `day03-rag-v3-improved.ipynb`.
