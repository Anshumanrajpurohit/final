@echo off
echo ===================================
echo Face Recognition System Update
echo ===================================
echo.

REM Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

REM Install MediaPipe
echo Installing MediaPipe for improved face detection...
pip install mediapipe
if %ERRORLEVEL% NEQ 0 (
    echo Warning: Failed to install MediaPipe. Face detection may not be optimal.
    echo You can try installing it manually: pip install mediapipe
)

echo.
echo Running face recognition system update...
echo.

REM Run the update script
python update_face_recognition.py

echo.
echo Update completed!
echo.
echo Now you can run the improved face recognition system:
echo python main_enhanced.py
echo.

pause
