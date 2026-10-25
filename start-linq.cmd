@echo off
setlocal EnableExtensions
rem Запуск сайта с инференсом людей на удалённом LinQ TPU (сервис хакатона).
rem Требования: подключён VPN LinQ, программа .tpu загружена на стенд,
rem пакет tpu_pycloud_format установлен из ZIP организаторов (см. README).
rem Подробно: docs/linq-site-integration.ru.md
cd /d "%~dp0"
set "LINQ_ADAPTER=my_linq:create_backend"
if "%LINQ_LOGIN%"=="" set "LINQ_LOGIN=hackathon74291"
if "%LINQ_PROGRAM%"=="" set "LINQ_PROGRAM=person-yolo12m-head-b1-o1-h1m.tpu"
echo LinQ adapter: %LINQ_ADAPTER% / login %LINQ_LOGIN% / program %LINQ_PROGRAM%
set "_LINQ_PYTHON=python"
if exist "%~dp0.venv\Scripts\python.exe" set "_LINQ_PYTHON=%~dp0.venv\Scripts\python.exe"
"%_LINQ_PYTHON%" -c "import pika, tpu_pycloud_format"
if errorlevel 1 (
  echo See README: install requirements-linq.txt, then run scripts/install_linq_runtime.py with the organizer ZIP.
  pause
  exit /b 1
)
"%_LINQ_PYTHON%" -m uvicorn app.server:app --host 127.0.0.1 --port 8765
pause
