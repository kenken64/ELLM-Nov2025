#!/bin/bash

# RAG Document Q&A Web Application Startup Script
# This script sets up the virtual environment and starts the Flask app

set -e  # Exit on error

echo "=================================="
echo "RAG Document Q&A - Startup Script"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Find the best Python version (prefer 3.11)
PYTHON_CMD=""
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3.9 &> /dev/null; then
    PYTHON_CMD="python3.9"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    echo "Please install Python 3.9 - 3.11"
    echo "  - macOS: brew install python@3.11"
    echo "  - Or download from: https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version | cut -d' ' -f2 | cut -d'.' -f1,2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

echo -e "${GREEN}✓${NC} Found Python ${PYTHON_VERSION} (using ${PYTHON_CMD})"

# Check Python version compatibility
if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 13 ]; then
    echo -e "${RED}Error: Python 3.13+ is not yet supported${NC}"
    echo "This application requires Python 3.9 - 3.11"
    echo "Please install Python 3.11:"
    echo "  - macOS: brew install python@3.11"
    echo "  - Or download from: https://www.python.org/downloads/"
    exit 1
fi

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]; then
    echo -e "${RED}Error: Python version too old${NC}"
    echo "This application requires Python 3.9 - 3.11"
    echo "Please upgrade Python"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo -e "${YELLOW}Virtual environment not found. Creating one...${NC}"
    $PYTHON_CMD -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
else
    echo -e "${GREEN}✓${NC} Virtual environment exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip -q

# Install/upgrade requirements
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}Error: requirements.txt not found${NC}"
    exit 1
fi

echo ""
echo "Installing dependencies..."
echo "(This may take 5-10 minutes on first run)"
pip install -r requirements.txt

echo ""
echo -e "${GREEN}✓${NC} Dependencies installed"

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p uploads
mkdir -p faiss_db
mkdir -p model_cache
echo -e "${GREEN}✓${NC} Directories ready"

# Check if this is first run (no models cached)
echo ""
if [ -z "$(ls -A model_cache 2>/dev/null)" ]; then
    echo -e "${YELLOW}⚠ First run detected${NC}"
    echo "The application will download ~4GB of AI models on first startup."
    echo "This is a one-time process and will take 5-10 minutes."
    echo "Subsequent runs will be much faster (~30-60 seconds)."
    echo ""
    read -p "Press Enter to continue..."
else
    echo -e "${GREEN}✓${NC} Model cache found - startup will be fast!"
fi

# Start the Flask application
echo ""
echo "=================================="
echo "Starting Flask Application..."
echo "=================================="
echo ""
echo -e "${GREEN}The app will be available at:${NC} http://localhost:5000"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Run the app
python app.py
