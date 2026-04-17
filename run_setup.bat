@echo off
cd /d "c:\Users\Elitebook\ChurnAi"
echo Activating venv...
call venv\Scripts\activate.bat
echo Installing requirements...
pip install -r docs\requirements.txt
echo Setup complete! venv activated.
echo Run: streamlit run src\app_fixed.py
pause

