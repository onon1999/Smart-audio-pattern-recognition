@echo off
echo ==========================================================
echo  EchoPulse AI - Push to GitHub
echo ==========================================================
python "%~dp0upload_to_github.py" %*
pause

