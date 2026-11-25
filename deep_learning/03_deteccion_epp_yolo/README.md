# DetecciÃ³n del estado de EPP con YOLO

Una caja por persona y cuatro clases:

| ID | Clase | Equipo visible |
|---|---|---|
| 0 | persona | Sin casco ni chaleco de seguridad |
| 1 | persona_con_casco | Casco, sin chaleco |
| 2 | persona_con_chaleco | Chaleco, sin casco |
| 3 | persona_epp_completo | Casco y chaleco |

La clase de chaleco hereda la definicion de la fuente: chaleco, chaqueta o camiseta de alta visibilidad.

Seguir `GUIA_COMANDOS.md` para entrenar, evaluar, exportar a ONNX y ejecutar con imÃ¡genes o webcam. El entorno utiliza CUDA cuando estÃ¡ disponible y CPU en los demÃ¡s equipos.

El dataset preparado se distribuye como `ppe_four.zip`. Su contenido se extrae en `data/ppe_four`. `prepare_dataset.py --audit` verifica imÃ¡genes, etiquetas y separaciÃ³n de duplicados. `prepare_dataset.py` permite reconstruirlo desde la fuente pÃºblica; no es necesario descargar la fuente si se dispone del ZIP preparado.

`train.py` adapta cuello/cabeza con el extractor congelado y despuÃ©s ajusta toda la red. Selecciona el checkpoint por mAP50â€“95 de validaciÃ³n. `evaluate.py` selecciona la confianza en validaciÃ³n y evalÃºa prueba con ese valor fijo. La fila/columna de fondo representa detecciones u objetos sin correspondencia; no es una clase entrenada.

`supplement_ppe_results.ipynb` presenta curvas, mÃ©tricas por clase, matriz de confusiÃ³n y ejemplos de aciertos y errores. `export_onnx.py` comprueba la equivalencia entre PyTorch y ONNX antes de preparar el paquete de inferencia.

Los resultados de referencia estÃ¡n en `runs/yolo11n_ppe`, el paquete ONNX en `runs/pc_ppe` y el dataset para distribuciÃ³n en `runs/ppe_four.zip`. Los experimentos del grupo usan otros nombres para conservar la referencia. Datos, pesos y resultados estÃ¡n excluidos de Git.

Consultar `FUENTES.md` para atribuciÃ³n y lÃ­mites de las etiquetas. El dataset combina fotografÃ­as e imÃ¡genes generadas; el rendimiento en una webcam debe comprobarse con imÃ¡genes del entorno de uso.
