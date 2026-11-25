# YOLO: entrenamiento y ejecución

## 1. Abrir la práctica y activar el entorno

Desde la raíz del repositorio:

```powershell
cd .\deep_learning\03_deteccion_epp_yolo
& "$env:LOCALAPPDATA\VisionRobotica\vision_robotica_dl\Scripts\Activate.ps1"
python -m pip install -r requirements_training.txt
```

## 2. Colocar y revisar el dataset

Desde el Explorador de archivos, extraer `ppe_four.zip` en la carpeta `data/ppe_four` de la práctica. Crear las carpetas si no existen. La estructura debe quedar así:

```text
03_deteccion_epp_yolo/
    data/
        ppe_four/
            images/
                train/
                val/
                test/
            labels/
                train/
                val/
                test/
            dataset.yaml
            manifest.json
        pretrained/
```

`images`, `labels` y `dataset.yaml` deben quedar directamente dentro de `ppe_four`, sin otra carpeta `ppe_four` intermedia. `pretrained` guarda los pesos iniciales y se crea automáticamente al entrenar si no existe.

Verificar y analizar el dataset:

```powershell
python prepare_dataset.py --audit
python analyze_dataset.py
```

## 3. Entrenar

```powershell
python train.py --head-epochs 5 --epochs 40 --batch-size 8 --output runs/yolo11n_clase
```

El entrenamiento puede terminar antes si deja de mejorar en validación. Si falta memoria, usar `--batch-size 4`. Para repetir el experimento, cambiar el nombre de salida.

## 4. Evaluar y revisar resultados

```powershell
python evaluate.py --run runs/yolo11n_clase --split val
python evaluate.py --run runs/yolo11n_clase --split test
python analyze_results.py --run runs/yolo11n_clase
Invoke-Item .\runs\yolo11n_clase\operating_test_matrix.png
Invoke-Item .\runs\yolo11n_clase\analysis\learning_curves.png
code .\supplement_ppe_results.ipynb
```

En el notebook, usar `RUN_NAME = 'yolo11n_clase'` y ejecutar todas las celdas con el kernel **Python (Vision Robotica DL)**. Revisar mAP50, mAP50-95, precisión, recall y errores por clase.

## 5. Exportar a ONNX

```powershell
python export_onnx.py --run runs/yolo11n_clase --output runs/pc_clase
Get-Content .\runs\yolo11n_clase\onnx_validation.json
python check_environment.py --model runs/pc_clase/model.onnx
```

Conservar `model.onnx` y `model.json` juntos. La validación debe indicar `PASS`. Después de exportar, volver a ejecutar la última celda del notebook para revisar ONNX.

## 6. Ejecutar con imágenes

```powershell
python infer.py --model runs/pc_clase/model.onnx --source data/ppe_four/images/test --output runs/prueba_imagenes --show
```

Cualquier tecla avanza; **Q** o **Esc** termina. Para una imagen propia:

```powershell
python infer.py --model runs/pc_clase/model.onnx --source "$env:USERPROFILE\Pictures\prueba.jpg" --output runs/imagen_propia --show
```

## 7. Ejecutar con webcam

```powershell
python infer.py --model runs/pc_clase/model.onnx --source 0 --output runs/prueba_webcam --show
```

**Q** o **Esc** termina. Si se abre otra cámara, cambiar `--source 0` por `--source 1`.

## Modelo de referencia ya entrenado

Estos comandos permiten probar el modelo proporcionado por el profesor, sin repetir el entrenamiento. Requieren el paquete de referencia extraído en `runs/pc_ppe`, con `model.onnx`, `model.json` y la carpeta `samples`.

```powershell
python infer.py --model runs/pc_ppe/model.onnx --source runs/pc_ppe/samples --show
python infer.py --model runs/pc_ppe/model.onnx --source 0 --show
```

Las detecciones e imágenes anotadas se guardan en la carpeta indicada por `--output`; si se omite, se utiliza `runs/onnx_check`. EPP completo se refiere a casco y chaleco visibles.
