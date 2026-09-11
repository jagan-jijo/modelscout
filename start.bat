@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>&1
    if not errorlevel 1 goto venv
)
py -3 -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>&1
if not errorlevel 1 goto py
python -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>&1
if not errorlevel 1 goto python
where uv >nul 2>&1
if not errorlevel 1 goto uv

echo ModelScout needs Python 3.11+ or uv. Install Python from python.org and run start.bat again. 1>&2
exit /b 1

:venv
".venv\Scripts\python.exe" "scripts\launch.py" %*
exit /b %errorlevel%
:py
py -3 "scripts\launch.py" %*
exit /b %errorlevel%
:python
python "scripts\launch.py" %*
exit /b %errorlevel%
:uv
uv run --no-project --python 3.12 python "scripts\launch.py" %*
exit /b %errorlevel%
