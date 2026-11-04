# Instalación del entorno de deep learning

Este entorno se utiliza para entrenar y evaluar redes neuronales de visión. Es independiente del entorno de visión clásica del curso.

| Entorno | Uso | Versión de Python |
|---|---|---|
| `.venv` | Visión clásica, OpenCV y controladores ligeros de Webots | 3.13 |
| `vision_robotica_dl` | Entrenamiento con PyTorch, redes convolucionales y transformers | 3.11 o 3.12 |

No desinstales Python 3.13 ni elimines el entorno `.venv`.

## 1. Actualizar el repositorio

Abre la carpeta `Vision_en_Robotica` en Visual Studio Code. Selecciona **Terminal > New Terminal** y ejecuta:

```powershell
git pull
```

## 2. Instalar Python 3.11

Verifica si Python 3.11 está disponible:

```powershell
py -3.11 --version
```

Si el comando no muestra una versión de Python 3.11, ejecuta:

```powershell
winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
```

Cierra y vuelve a abrir Visual Studio Code después de instalar Python.

## 3. Crear el entorno

Desde la raíz del repositorio, ejecuta:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\deep_learning\setup_dl.ps1 -Backend Auto
```

El instalador crea el entorno en:

```text
%LOCALAPPDATA%\VisionRobotica\vision_robotica_dl
```

El modo `Auto` instala soporte CUDA cuando detecta una GPU NVIDIA compatible. Si CUDA no está disponible, instala la versión para CPU.

La instalación termina correctamente cuando aparece:

```text
Environment ready.
```

## 4. Activar el entorno

```powershell
& "$env:LOCALAPPDATA\VisionRobotica\vision_robotica_dl\Scripts\Activate.ps1"
```

## 5. Verificar desde la terminal

```powershell
python .\deep_learning\check_dl_environment.py
```

El resultado correcto termina con `GPU test: OK` o `CPU test: OK`. Ambos resultados indican que el entorno funciona.

## 6. Ejecutar el cuaderno de prueba

1. Abre `deep_learning/hello_deep_learning.ipynb`.
2. Selecciona el kernel **Python (Vision Robotica DL)**.
3. Selecciona **Run All**.

El cuaderno verifica:

- operaciones con tensores en GPU o CPU;
- inferencia con una red convolucional;
- inferencia con un vision transformer pequeño;
- procesamiento y visualización con OpenCV;
- ejecución de un modelo ONNX.

La última celda debe mostrar:

```text
Entorno listo: 6 de 6 pruebas aprobadas.
```

## Solución de problemas

- **No se encuentra Python 3.11:** cierra y vuelve a abrir Visual Studio Code; después ejecuta `py -3.11 --version`.
- **El instalador selecciona CPU:** el equipo no tiene una GPU NVIDIA compatible o CUDA no pudo activarse. El entorno puede utilizarse con CPU.
- **Falla la instalación de paquetes:** verifica la conexión a Internet y ejecuta nuevamente el instalador.
- **No aparece el kernel:** termina la instalación y reinicia Visual Studio Code.
- **Memoria insuficiente en GPU durante prácticas posteriores:** reduce el tamaño del lote o utiliza CPU.
