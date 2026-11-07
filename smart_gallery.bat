@echo off
rem Opens the Smart Gallery for an indexed run.
rem   smart_gallery.bat                 the photos folder, final model
rem   smart_gallery.bat photos-fast     another run under data\runs
chcp 65001 >nul
cd /d "%~dp0"
set RUN=%1
if "%RUN%"=="" set RUN=photos-final
if not exist "data\runs\%RUN%\config.yaml" (
  echo No indexed run at data\runs\%RUN%. Index it first with run_all.ipynb.
  pause
  exit /b 1
)
set PYTHONIOENCODING=utf-8
set PYTHONWARNINGS=ignore
".venv\Scripts\python.exe" -m vie.gallery --config "data\runs\%RUN%\config.yaml"
pause
