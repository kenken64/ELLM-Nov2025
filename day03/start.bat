@echo off
REM RAG Document Q&A Web Application Startup Script for Windows

echo ==================================
echo RAG Document Q&A - Startup Script
echo ==================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python 3 is not installed
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

echo [OK] Found Python
python --version

REM Create virtual environment if it doesn't exist
if not exist "venv\" (
    echo.
    echo Virtual environment not found. Creating one...
    python -m venv venv
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment exists
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip -q

REM Install/upgrade requirements
if not exist "requirements.txt" (
    echo Error: requirements.txt not found
    pause
    exit /b 1
)

echo.
echo Installing dependencies...
echo (This may take 5-10 minutes on first run)
pip install -r requirements.txt

echo.
echo [OK] Dependencies installed

REM Create necessary directories
echo.
echo Creating directories...
if not exist "uploads\" mkdir uploads
if not exist "chroma_db\" mkdir chroma_db
if not exist "model_cache\" mkdir model_cache
echo [OK] Directories ready

REM Check if this is first run
echo.
if not exist "model_cache\*" (
    echo [!] First run detected
    echo The application will download ~4GB of AI models on first startup.
    echo This is a one-time process and will take 5-10 minutes.
    echo Subsequent runs will be much faster (~30-60 seconds).
    echo.
    pause
) else (
    echo [OK] Model cache found - startup will be fast!
)

REM Start the Flask application
echo.
echo ==================================
echo Starting Flask Application...
echo ==================================
echo.
echo The app will be available at: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.

REM Run the app
python app.py
