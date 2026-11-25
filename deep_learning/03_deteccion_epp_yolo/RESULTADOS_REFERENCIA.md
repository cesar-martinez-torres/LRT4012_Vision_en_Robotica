# Resultado de referencia: estado de EPP por persona

YOLO11n preentrenado en COCO, entrada 640 x 640, FP32. Entrenamiento en NVIDIA T1000 de 8 GB: 5 épocas de adaptación y 40 épocas de ajuste completo (máximo 40). Selección del checkpoint: finetune, por mAP50–95 de validación. Confianza: 0.65, elegida por macro-F1 de validación y aplicada sin modificar en prueba.

Dataset preparado: 873 imágenes; entrenamiento 611, validación 131, prueba 131. Fuente mixta de fotografías e imágenes generadas. Estados derivados de anotaciones explícitas, con exclusión de imágenes ambiguas y revisión visual de 96 personas. La partición de prueba es interna; no es una evaluación de una webcam independiente.

mAP50 de prueba: **0.884**. mAP50–95 de prueba: **0.728**. Macro-F1 al umbral seleccionado: **0.822**.

| Clase | Personas en prueba | AP50 | AP50–95 | Precisión | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| persona | 30 | 0.809 | 0.653 | 1.000 | 0.667 | 0.800 |
| persona_con_casco | 34 | 0.900 | 0.781 | 0.867 | 0.765 | 0.812 |
| persona_con_chaleco | 53 | 0.869 | 0.702 | 0.878 | 0.679 | 0.766 |
| persona_epp_completo | 151 | 0.957 | 0.777 | 0.956 | 0.868 | 0.910 |

AP usa un barrido de confianza; precisión, recall y F1 de esta tabla usan el umbral fijo indicado. Las filas de `operating_test_matrix.png` son referencia y las columnas predicción. Fondo representa omisiones o falsas detecciones, sin ser una quinta clase.

## Archivos

- `runs/yolo11n_ppe`: selección, curvas, métricas y matriz de confusión.
- `runs/pc_ppe/model.onnx`: modelo de inferencia con metadatos en `model.json`.
- `runs/pc_ppe.zip`: paquete autónomo de inferencia en PC.
- `runs/ppe_four.zip`: dataset preparado para distribuir; checksum en `runs/ppe_four.sha256`.
- `runs/yolo11n_ppe/onnx_validation.json`: equivalencia PyTorch/ONNX en validación.
- `runs/yolo11n_ppe/portable_validation.json`: ejecución del paquete extraído fuera del proyecto.
- `supplement_ppe_results.ipynb`: análisis reproducible, sin resultados incrustados en Git.

«EPP completo» describe únicamente casco y chaleco visibles. La clase de chaleco incluye otras prendas de alta visibilidad según la fuente. Una ausencia de detección no demuestra ausencia de personas ni de riesgo. Consultar `FUENTES.md` para procedencia, licencia y límites de las etiquetas.
