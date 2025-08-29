@echo off
echo ===================================
echo Face Recognition Database Cleanup
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

REM Check if required packages are installed
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
echo Starting database cleanup...
echo.

REM Run the database cleanup script
python cleanup_database.py

echo.
echo Database cleanup completed!
echo.

pause
