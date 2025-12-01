# Quick Start Guide

## First Time Setup

```bash
# Make the startup script executable (if not already)
chmod +x start.sh

# Run the app
./start.sh
```

That's it! The script handles everything.

## What Happens on First Run

1. **Virtual environment created** (~10 seconds)
2. **Dependencies installed** (~2-3 minutes)
3. **AI models downloaded** (~5-10 minutes, ~4GB)
   - BAAI/bge-small-en-v1.5 (embeddings)
   - google/flan-t5-large (answer generation)
   - cross-encoder/ms-marco-MiniLM-L-6-v2 (reranking)
4. **Server starts** at http://localhost:5000

## Subsequent Runs

```bash
./start.sh
```

Much faster (~30-60 seconds) - only loads models from cache, no downloads!

## Using the App

1. **Open browser**: http://localhost:5000
2. **Upload documents**: Drag & drop PDF, EPUB, HTML, or PPTX files
3. **Ask questions**: Type in the question box and hit Enter
4. **View answers**: See AI-generated answers with source citations

## Stopping the App

Press `Ctrl+C` in the terminal

## Troubleshooting

### Permission denied when running start.sh
```bash
chmod +x start.sh
```

### Port 5000 already in use
Edit `app.py` line 285 and change port:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Out of memory
Close other applications or use a smaller model:
- Edit `app.py` line 51
- Change `google/flan-t5-large` to `google/flan-t5-base`

## File Locations

- **Uploaded docs stored in**: `./chroma_db/` (persists across restarts)
- **AI models cached in**: `./model_cache/` (~4GB, one-time download)
- **Temp files**: `./uploads/` (cleaned after processing)

## Need Help?

See the full [README.md](README.md) for detailed documentation.
