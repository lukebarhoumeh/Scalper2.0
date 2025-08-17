@echo off
title Unified ScalperBot v2.0
cls

echo ====================================================
echo           UNIFIED SCALPERBOT v2.0 LAUNCHER
echo          Production-Grade HFT Trading System         
echo ====================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

:: Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Creating...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing requirements...
    pip install -r requirements_production.txt
)

:: Set environment variables
echo.
echo Setting environment variables...
set PAPER_TRADING=true
set CLOSE_ON_EXIT=false
set PYTHONUNBUFFERED=1

:: Check if API keys are set
if not defined CB_API_KEY (
    echo.
    echo WARNING: CB_API_KEY not set - Running in simulation mode
    echo To trade live, set your Coinbase API credentials:
    echo   set CB_API_KEY=your_api_key
    echo   set CB_API_SECRET=your_api_secret
    echo.
)

:: Launch the bot
echo.
echo Starting Unified ScalperBot v2.0...
echo Mode: PAPER TRADING
echo.
echo ====================================================
echo.

python unified_scalperbot_v2.py

:: Keep window open on exit
echo.
echo Bot terminated.
pause 