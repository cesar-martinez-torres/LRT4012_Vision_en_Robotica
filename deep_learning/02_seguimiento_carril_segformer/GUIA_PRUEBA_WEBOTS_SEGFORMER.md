# SegFormer: entrenamiento y ejecución en Webots

Con el dataset preparado, ejecutar los pasos en PowerShell. Si ya terminó el
entrenamiento `segformer_b0_clase`, continuar desde el paso 3.

## 1. Activar el ambiente y entrar al proyecto

```powershell
& "$env:LOCALAPPDATA\VisionRobotica\vision_robotica_dl\Scripts\Activate.ps1"
cd "C:\Users\20808\Documents\Repositorios\Vision_en_Robotica\deep_learning\02_seguimiento_carril_segformer"
```

## 2. Entrenar

```powershell
python .\train.py --epochs 15 --head-epochs 3 --batch-size 8 --output .\runs\segformer_b0_clase
```

## 3. Verificar los resultados

```powershell
python .\scripts\analyze_results.py --run .\runs\segformer_b0_clase
```

## 4. Abrir el cuaderno de resultados

```powershell
code .\supplement_segformer_fine_tuning.ipynb
```

Seleccionar **Python (Vision Robotica DL)** y ejecutar **Run All**.

## 5. Exportar a ONNX

```powershell
python .\export_onnx.py --run .\runs\segformer_b0_clase
```

## 6. Verificar la conversión

```powershell
Get-Content .\runs\segformer_b0_clase\onnx_validation.json
```

Revisar que `mask_agreement` sea cercano a 1.

## 7. Ejecutar Webots

Cerrar Webots antes de ejecutar el comando.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_fixed_track.ps1 -Python python -Model .\runs\segformer_b0_clase\best_model.onnx -Track 1
```

Para probar otra pista, cerrar Webots y repetir cambiando `-Track 1` por
`-Track 2`, `-Track 3`, `-Track 4` o `-Track 5`.

## 8. Abrir los resultados del recorrido

```powershell
Invoke-Item .\runs\fixed_track
```
