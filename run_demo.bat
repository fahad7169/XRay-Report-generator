@echo off
echo ========================================
echo   AI Radiology Report Generator
echo   Professional Demo Application
echo ========================================
echo.
echo Starting Streamlit app...
echo.
echo Once the app opens in your browser:
echo   1. Enter patient information
echo   2. Upload chest X-ray images
echo   3. Click "Generate Radiology Report"
echo   4. Download PDF report
echo.
echo Press Ctrl+C to stop the server
echo.
echo ========================================

.\venv\Scripts\python.exe -m streamlit run app.py

