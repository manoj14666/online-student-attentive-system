#!/bin/bash
# Online Class Facial Emotion Detection - Linux/Mac Startup Script

echo "Online Class Facial Emotion Detection"
echo "===================================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.7 or higher"
    exit 1
fi

echo "Python found. Starting application..."
echo
echo "The application will start on: http://localhost:5000"
echo "Press Ctrl+C to stop the application"
echo

# Start the application
python3 app.py
