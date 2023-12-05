@echo off
python --version
echo Starting virtual environment setup.
echo.
python -m venv __venv__
call __venv__\Scripts\activate.bat
echo.
echo Virtual environment setup completed.
echo.
echo Start training...
call train.py
pause
echo Start testing...
call test.py
pause
