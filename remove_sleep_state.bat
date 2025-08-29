@echo off
echo ===================================
echo Remove Sleep State Functionality
echo ===================================
echo This script will remove sleep state functionality from the face recognition system
echo to ensure real-time detection without delays.
echo.

REM Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

REM Install required packages if needed
echo Checking required packages...
python -c "import mysql.connector" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing mysql-connector-python...
    pip install mysql-connector-python
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to install mysql-connector-python
        pause
        exit /b 1
    )
)

python -c "import dotenv" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing python-dotenv...
    pip install python-dotenv
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to install python-dotenv
        pause
        exit /b 1
    )
)

echo.
echo Select an option:
echo 1. Remove sleep state from database only
echo 2. Apply code patches to disable sleep functionality
echo 3. Do both (recommended)
echo.

set /p option="Enter option (1-3): "

if "%option%"=="1" (
    echo.
    echo Removing sleep state from database...
    python remove_sleep_state.py
) else if "%option%"=="2" (
    echo.
    echo Applying code patches...
    python patch_disable_sleep.py
) else if "%option%"=="3" (
    echo.
    echo Removing sleep state from database...
    python remove_sleep_state.py
    echo.
    echo Applying code patches...
    python patch_disable_sleep.py
) else (
    echo Invalid option selected
    pause
    exit /b 1
)

echo.
echo ===================================
echo Process completed!
echo ===================================
echo The system will now perform real-time detection without sleep state delays.
echo To apply these changes, restart the face recognition system:
echo python main_enhanced.py
echo.

pause
