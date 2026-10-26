@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv
  if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -c "import flask, cv2, numpy" >nul 2>&1
if errorlevel 1 (
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  if errorlevel 1 goto failed
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" download_model.py
if errorlevel 1 goto failed
".venv\Scripts\python.exe" app.py
pause
exit /b
:failed
echo Setup failed. Check Python 3.12 installation and your internet connection.
pause
