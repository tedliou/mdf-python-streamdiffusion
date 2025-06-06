@echo off
WHERE uv
IF %ERRORLEVEL% NEQ 0 CALL "Tools\install.bat"
uv run main.py
