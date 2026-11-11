@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "WEBOTS_EXE=C:\Program Files\Webots\msys64\mingw64\bin\webotsw.exe"
set "WORLD_FILE=%PROJECT_DIR%worlds\turtlebot_random_tracks.wbt"
set "MODEL_FILE=%PROJECT_DIR%runs\mobilenet_v3_small\best_model.onnx"

if not exist "%WEBOTS_EXE%" (
  echo No se encontro Webots en: %WEBOTS_EXE%
  pause
  exit /b 1
)

if not exist "%WORLD_FILE%" (
  echo No se encontro el mundo en: %WORLD_FILE%
  pause
  exit /b 1
)

if not exist "%MODEL_FILE%" (
  echo No se encontro el modelo ONNX en: %MODEL_FILE%
  echo Ejecuta export_onnx.py despues del entrenamiento.
  pause
  exit /b 1
)

set "WEBOTS_TRACK_SEED=%RANDOM%"
if not "%~1"=="" set "WEBOTS_TRACK_SEED=%~1"
set "WEBOTS_TRACK_ROWS=4"
set "WEBOTS_TRACK_COLS=4"
set "WEBOTS_TURTLEBOT_BATCH=0"
set "WEBOTS_CONFIDENCE_THRESHOLD=0.80"
set "WEBOTS_MODEL_PATH=%MODEL_FILE%"
set "WEBOTS_STABLE_PREDICTIONS=3"
set "WEBOTS_FORWARD_DISTANCE=0.42"
set "WEBOTS_TURN_ANGLE=90"
set "WEBOTS_STOP_STEPS=24"
set "WEBOTS_GOAL_TOLERANCE=0.04"
set "WEBOTS_START_ATTEMPT=0"

echo Semilla de la pista: %WEBOTS_TRACK_SEED%

:START_WEBOTS
set /a WEBOTS_START_ATTEMPT+=1
pushd "%PROJECT_DIR%"
start "" /wait "%WEBOTS_EXE%" --mode=realtime "%WORLD_FILE%"
set "WEBOTS_EXIT_CODE=%ERRORLEVEL%"
popd

if "%WEBOTS_EXIT_CODE%"=="-1073741819" goto RETRY_ACCESS_VIOLATION
goto CHECK_EXIT_CODE

:RETRY_ACCESS_VIOLATION
if %WEBOTS_START_ATTEMPT% GEQ 2 goto CHECK_EXIT_CODE
echo.
echo Webots tuvo un fallo grafico al iniciar. Reintentando una vez...
timeout /t 2 /nobreak >nul
goto START_WEBOTS

:CHECK_EXIT_CODE
if not "%WEBOTS_EXIT_CODE%"=="0" (
  echo.
  echo Webots termino con el codigo %WEBOTS_EXIT_CODE%.
  echo Revisa los mensajes anteriores para identificar el error.
  pause
)

endlocal
