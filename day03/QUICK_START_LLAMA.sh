#!/bin/bash

# Quick Start Script for Llama 3.2 3B RAG System
# Run this to set up and test the Llama version

echo "========================================="
echo "Llama 3.2 3B RAG System - Quick Start"
echo "========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./start.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

echo "Step 1: Checking Hugging Face login..."
echo "----------------------------------------"

# Check if already logged in
if ! huggingface-cli whoami > /dev/null 2>&1; then
    echo "❌ Not logged in to Hugging Face"
    echo ""
    echo "Please complete these steps:"
    echo "1. Go to: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct"
    echo "2. Click 'Agree and access repository'"
    echo "3. Get your token from: https://huggingface.co/settings/tokens"
    echo "4. Run: huggingface-cli login"
    echo ""
    read -p "Press Enter after completing login, or Ctrl+C to exit..."

    # Check again
    if ! huggingface-cli whoami > /dev/null 2>&1; then
        echo "❌ Still not logged in. Please run: huggingface-cli login"
        exit 1
    fi
fi

USER=$(huggingface-cli whoami | head -n 1)
echo "✓ Logged in as: $USER"
echo ""

echo "Step 2: Testing Llama 3.2 3B model..."
echo "----------------------------------------"
echo "This may take 5-10 minutes on first run (downloading 8GB model)"
echo "Subsequent runs will be fast (model is cached)"
echo ""

# Test with a simple question
python -c "
import sys
sys.path.insert(0, '/Users/kennethphang/Projects/ELLM-Nov2025/day03')

print('Loading Llama 3.2 3B model...')
print('(This may take a few minutes on first run)')
print('')

try:
    from app_llama import answer_question

    print('✓ Model loaded successfully!')
    print('')
    print('Testing with sample question...')
    print('-' * 60)

    question = 'Who was Scrooge\\'s deceased business partner?'
    print(f'Question: {question}')
    print('')

    result = answer_question(question, retrieve_k=30, use_k=5)

    print(f'Answer: {result[\"answer\"]}')
    print('-' * 60)
    print('')
    print('✓ Test successful! Llama 3.2 3B is working.')

except Exception as e:
    print(f'❌ Error: {e}')
    print('')
    print('Troubleshooting:')
    print('1. Make sure you accepted the Llama license')
    print('2. Check your Hugging Face login: huggingface-cli whoami')
    print('3. Check you have enough RAM (12GB+ needed)')
    sys.exit(1)
" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ SUCCESS! Llama 3.2 3B is ready to use"
    echo "========================================="
    echo ""
    echo "Next steps:"
    echo "1. Run full test suite: python test_llama.py"
    echo "2. Start Flask app: python app_llama.py"
    echo "3. Read detailed docs: cat LLAMA_SETUP.md"
    echo ""
    echo "Expected performance: 80-90% accuracy WITHOUT hardcoded answers!"
else
    echo ""
    echo "========================================="
    echo "❌ Setup failed - see error above"
    echo "========================================="
    echo ""
    echo "Common fixes:"
    echo "1. Accept license: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct"
    echo "2. Login: huggingface-cli login"
    echo "3. Check RAM: You need 12GB+ available"
    echo ""
    exit 1
fi
