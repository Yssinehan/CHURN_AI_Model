@echo off
cd /d "c:\Users\Elitebook\ChurnAi"
call venv\Scripts\activate.bat
streamlit run src\app_fixed.py

echo.
echo API Server aussi disponible:
echo uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
echo.
pause
pause

