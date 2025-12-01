# Installation Notes

## ⚠️ Current Status

There are dependency conflicts with the current Python version and system architecture. ChromaDB has specific binary requirements that may not be available for Python 3.12+.

## Recommended Solution

**Use Python 3.11** for best compatibility with all dependencies.

### Step 1: Install Python 3.11

```bash
# macOS with Homebrew
brew install python@3.11

# Or download from python.org
```

### Step 2: Create Virtual Environment with Python 3.11

```bash
# Remove existing venv if present
rm -rf venv

# Create new venv with Python 3.11
python3.11 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 3: Install Dependencies

```bash
pip install flask==3.0.3
pip install chromadb==0.4.22
pip install langchain-community==0.2.16
pip install transformers==4.40.0
pip install torch
pip install sentence-transformers==2.7.0
pip install "unstructured[pdf,pptx]==0.11.8"
```

###Step 4: Run the Application

```bash
python app.py
```

## Alternative: Use Docker (Recommended for Production)

If you continue to have dependency issues, Docker is the most reliable option:

```bash
# Coming soon: Docker setup
```

## Known Issues

1. **Python 3.13**: Not supported - use Python 3.9-3.11
2. **Python 3.12**: Limited support - ChromaDB binary dependencies may not be available
3. **ARM Mac (M1/M2/M3)**: Some binary packages may have limited availability

## Quick Test

To verify your environment is working:

```python
python3 -c "import chromadb; import transformers; import flask; print('All core dependencies OK!')"
```

## Need Help?

If you're still having issues:
1. Check Python version: `python3 --version` (should be 3.9-3.11)
2. Check architecture: `uname -m`
3. Try the manual installation steps above with Python 3.11
