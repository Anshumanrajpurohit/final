@echo off
echo Testing MediaPipe Face Detection
echo ==============================

if "%~1"=="" (
    echo Error: Please provide an image path
    echo Usage: test_mediapipe.bat path\to\image.jpg
    exit /b 1
)

python test_mediapipe.py %1

echo Test completed.
pause
