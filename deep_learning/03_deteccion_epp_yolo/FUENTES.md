# Fuentes y preparaciÃ³n

- Dataset: [Ultralytics Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe/), ediciÃ³n de noviembre de 2026, 1,000 imÃ¡genes. Combina fotografÃ­as e imÃ¡genes generadas; no identifica el origen de cada imagen. Licencia CC BY 4.0, atribuciÃ³n al equipo de Ultralytics.
- [Archivo original](https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip). SHA-256: `e5817aa9c73f33e525d76ca6bc6af4e4eac9dcea9bcfef1cfd1a71c09fbc1232`.
- Modelo inicial: [YOLO11n preentrenado en COCO](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt). ImplementaciÃ³n y pesos sujetos a las condiciones de Ultralytics (AGPL-3.0 o licencia empresarial).

La clase original `vest` incluye chalecos, chaquetas y camisetas de alta visibilidad; las clases con chaleco heredan ese alcance.

Las cajas de persona se conservan. Las regiones anotadas como casco/sin casco y chaleco/sin chaleco se asocian geomÃ©tricamente con la persona. Solo se aceptan imÃ¡genes en las que todas las personas tienen ambos estados explÃ­citos y sin conflictos. Una etiqueta ausente no demuestra ausencia de equipo.

Se generan cuatro estados mutuamente excluyentes por persona. Â«EPP completoÂ» significa Ãºnicamente casco y chaleco visibles. No se evalÃºan certificaciÃ³n, ajuste, otros equipos ni cumplimiento normativo.

Se agrupan duplicados exactos y similitudes conservadoras antes de dividir 70/15/15 para entrenamiento, validaciÃ³n y prueba. El conjunto original no tiene particiÃ³n de prueba: se construye una particiÃ³n interna nueva. La independencia entre escenas y sesiones no estÃ¡ garantizada. Las etiquetas derivadas heredan errores de la fuente y de la asociaciÃ³n; la revisiÃ³n visual por muestreo no equivale a una auditorÃ­a exhaustiva.

La procedencia, las asociaciones por persona, las exclusiones y los conteos se conservan en `data/ppe_four`. Al redistribuir el dataset deben incluirse `SOURCE_LICENSE.txt`, `SOURCE_README.md`, `provenance.json` y este documento.
