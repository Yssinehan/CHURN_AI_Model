@echo off
cd /d "c:\Users\Elitebook\ChurnAi"
call venv\Scripts\activate.bat
streamlit run src\app_fixed.py
pause

