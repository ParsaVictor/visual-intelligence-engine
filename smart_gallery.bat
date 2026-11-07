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
rem Every model is already on disk; without this each start checks the Hub online,
rem which on a slow connection held the page back for minutes.
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set YOLO_OFFLINE=True
set YOLO_AUTOINSTALL=False
set GRADIO_ANALYTICS_ENABLED=False
set HF_HUB_DISABLE_TELEMETRY=1
rem Python would otherwise route even this page's own requests through the Windows
rem system proxy (a VPN), and the page would not start while the VPN is off.
set NO_PROXY=*
set no_proxy=*
".venv\Scripts\python.exe" -m vie.gallery --config "data\runs\%RUN%\config.yaml"
pause
