# Segmentación de carril con SegFormer: pistas y dataset de Webots

## Objetivo

Preparar imágenes y máscaras y ajustar SegFormer-B0 como segmentador de la
superficie transitable de una pista. El proyecto incluye escenarios, captura,
auditoría, entrenamiento, análisis, exportación a ONNX y seguimiento de un
cinco circuitos fijos con TurtleBot3 Burger en Webots, mediante su controlador interno.

## Exportar y seguir una pista fija

Desde esta carpeta, con el ambiente de deep learning activado:

```powershell
python .\export_onnx.py
.\scripts\run_fixed_track.ps1 -Python python
```

El exportador utiliza `runs/segformer_b0/best_model/`, guarda la red en
`runs/segformer_b0/best_model.onnx` y escribe junto a ella `best_model.json`
con el preprocesamiento. Comprueba el modelo y compara logits y máscaras de
24 imágenes de prueba con PyTorch; guarda `onnx_validation.json`.

El lanzador configura los intérpretes locales de ambos controladores y abre
`worlds/turtlebot_lane_fixed.wbt`. Requiere Webots R2025a y NumPy, OpenCV y
ONNX Runtime en el intérprete seleccionado. La inferencia usa CPU y no importa
PyTorch dentro de Webots. Después de esta configuración también puede abrirse
directamente el mundo en Webots.

El circuito original conserva exactamente sus coordenadas. Las otras cuatro
pistas tienen geometría definida mediante óvalos, rectas, arcos y curvas suaves;
sus trazados no se generan aleatoriamente. `fixed_tracks.py` contiene el catálogo.

| Número | Pista | Mundo de Webots |
|---:|---|---|
| 1 | Original | `worlds/turtlebot_lane_fixed.wbt` |
| 2 | Óvalo | `worlds/turtlebot_lane_fixed_02.wbt` |
| 3 | Estadio: dos rectas y dos curvas | `worlds/turtlebot_lane_fixed_03.wbt` |
| 4 | Rectángulo redondeado | `worlds/turtlebot_lane_fixed_04.wbt` |
| 5 | Curvas en S | `worlds/turtlebot_lane_fixed_05.wbt` |

Elegir una pista con `-Track`, de 1 a 5; el valor predeterminado es 1:

```powershell
.\scripts\run_fixed_track.ps1 -Python python -Track 3
```

Después de configurar los intérpretes, también puede abrirse directamente el
mundo correspondiente en Webots. Cada archivo selecciona su propia pista y
la repite en cada ejecución. Las cinco conservan ancho, cámara, colores e
iluminación para comparar el control. Reutilizan el mismo modelo ONNX; cambiar
de pista no requiere volver a entrenar ni exportar.

Para consultar el trazado de las cinco pistas:

```powershell
python .\scripts\preview_fixed_tracks.py
```

La figura se guarda en `runs/fixed_tracks_catalog.png`. El punto negro marca
el inicio y la flecha indica el sentido de circulación.

La cámara proporciona imágenes al segmentador; `lane_inference.py`
estima el centro del carril predicho mediante la calibración de la cámara.
`turtlebot_lane_follower.py` aplica seguimiento geométrico local (*pure pursuit*)
y convierte la velocidad lineal y angular en velocidades de las ruedas.
El recorrido termina tras una vuelta, una salida del carril, pérdida de la
referencia visual o tiempo máximo. Al terminar, el robot se detiene y la
simulación queda pausada.

### Vistas en Webots

Los cinco mundos incluyen el panel `vision dashboard`, con tres vistas:

- **Cámara RGB:** imagen que recibe la red.
- **Máscara predicha:** carril en blanco y fondo en negro.
- **Carril y punto objetivo:** segmentación en verde y objetivo de seguimiento
  en rojo, con una línea desde el centro inferior de la imagen.

El panel muestra también las velocidades lineal y angular, el tiempo de
inferencia y el estado del controlador. Se actualiza con cada inferencia,
aproximadamente cada 96 ms de simulación. `vision_dashboard.png` conserva una
captura periódica en el directorio de resultados.

El panel aparece sobre la vista 3D. Puede moverse y redimensionarse. Si se
cierra, seleccionar el robot y activarlo en **Overlays → Display Devices →
vision dashboard**. La cámara nativa de Webots puede ocultarse desde
**Overlays → Camera Devices** para ver únicamente el panel procesado.

### Relación con el ejercicio de comandos visuales

El flujo general se conserva: **cámara → preprocesamiento → ONNX → decisión
de control → velocidades de rueda**. El supervisor construye el escenario,
coordina inicio y parada y evalúa el resultado.

| Etapa | MobileNet: comandos visuales | SegFormer: seguimiento de carril |
|---|---|---|
| Predicción | Una clase: avanzar, izquierda, derecha o detenerse. | Una clase por píxel: fondo o carril. |
| Decisión | Confirmar la orden con varias predicciones estables. | Estimar un punto del centro del carril a partir de la máscara. |
| Movimiento | Ejecutar una distancia o un giro; verificar con encoders e IMU. | Ajustar continuamente avance y giro con seguimiento geométrico local. |
| Realimentación | Sensores de movimiento durante la maniobra. | Nuevas imágenes mientras el robot avanza. |

La red proporciona percepción; el controlador convierte esa percepción en
movimiento. En este ejercicio la red no predice directamente velocidades
ni utiliza las coordenadas reales de la pista para conducir.

`fixed_lane_supervisor.py` construye la pista y registra el progreso y la
desviación real. No proporciona al controlador coordenadas, mapa, orientación
ni máscaras de referencia. Sólo envía inicio, calibración de inclinación de
cámara y parada. El seguimiento supone suelo plano y cámara calibrada.

Cada ejecución del lanzador conserva resultados en un directorio con fecha
dentro de `runs/fixed_track/`: `result.json`, `trajectory.csv`,
`vision_control.csv`, `track.json`, vistas del mundo y superposición de la
predicción. La apertura directa utiliza `runs/fixed_track/latest_01/` a
`runs/fixed_track/latest_05/`, según la pista elegida. `result.json` identifica
el número y nombre del circuito.

Para una prueba automática que cierre Webots al terminar:

```powershell
.\scripts\run_fixed_track.ps1 -Python python -Track 3 -Batch
python .\scripts\analyze_following.py
```

Abrir **`supplement_segformer_lane_following.ipynb`** para revisar la equivalencia
ONNX, el punto objetivo, la trayectoria, la desviación lateral y la latencia.
Los modelos y resultados permanecen excluidos de Git.

## Entrenamiento y análisis

Seguir `GUIA_AJUSTE_FINO_SEGFORMER.md` para repetir el proceso de MobileNet con
las particularidades de la segmentación. En el ambiente de deep learning:

```powershell
python .\scripts\analyze_dataset.py
python .\train.py --epochs 15 --head-epochs 3 --batch-size 8
python .\scripts\analyze_results.py
```

Abrir `supplement_segformer_fine_tuning.ipynb` para analizar el entrenamiento.
La primera descarga utiliza `nvidia/mit-b0`. El script selecciona el checkpoint
con mayor IoU media de validación y luego evalúa la prueba. No reemplaza un
directorio de resultados existente; usar `--output` para otro experimento.

Si las máscaras originales parecen negras, consultar
`runs/dataset_analysis/mask_verification.png` y las copias de alto contraste en
`runs/dataset_analysis/mask_previews/`. Las etiquetas originales 0 y 1 se
conservan; la paleta de las copias sólo modifica su apariencia.

## Dataset completo

| Partición | Pistas independientes | Pares imagen–máscara |
|---|---:|---:|
| Entrenamiento (`train`) | 20 | 1 920 |
| Validación (`val`) | 6 | 576 |
| Prueba (`test`) | 6 | 576 |
| Total | 32 | 3 072 |

Cada muestra contiene una imagen RGB de 320 × 192 píxeles y una máscara PNG
monocanal del mismo tamaño con identificadores enteros:

- `0`: fondo, suelo fuera de la pista, cielo y elementos del entorno.
- `1`: superficie del carril, incluidas sus marcas de borde.

Las máscaras almacenan **0 y 1**, no 0 y 255; por ello pueden parecer negras en
un visor convencional. Para visualizarlas, multiplicar una copia por 255. No
modificar las etiquetas originales. Al configurar el procesador de SegFormer,
conservar `do_reduce_labels=False`: el fondo es una clase válida. Las máscaras
se redimensionan con interpolación de vecino más próximo, nunca bilineal.

```text
data/lane_segmentation_v1/
  train/images/    train/masks/
  val/images/      val/masks/
  test/images/     test/masks/
  tracks/                  Especificaciones y coordenadas de cada pista
  manifest.json            Clases, cámara, particiones y configuración
  metadata.csv             Pose, pista, rutas, hashes y proporción de carril
  capture_complete.json    Confirmación de captura completa
  audit_report.json        Resultado de auditoría
  tracks_overview.png      Vista superior de las 32 pistas
  dataset_preview.png      Imagen, máscara y superposición
```

La separación se realiza por pistas completas. Las imágenes de una pista no
se reparten entre entrenamiento, validación y prueba. También se comprueban
semillas y geometrías distintas y duplicados RGB exactos entre particiones.

## Escenarios y captura

Las pistas son circuitos cerrados, sin bifurcaciones ni cruces. El generador
comprueba que sus bordes no se intersecten y limita la curvatura. Varían el
ancho (0.48–0.68 m), el trazado, el color de superficie y suelo, la iluminación,
las marcas de borde, el entorno y la inclinación de la cámara.

Se capturan 96 poses por pista, en ambos sentidos, con variaciones de posición
lateral y orientación. Una cámara cinemática reproduce el montaje del Burger:
0.03 m hacia delante, 0.213 m de altura y campo horizontal de 1.1 radianes.
No se obtienen los datos mediante conducción autónoma de una red ya entrenada:
las poses son supervisadas. Este procedimiento evita que un controlador previo
limite la diversidad de las muestras.

Webots produce en el mismo instante la imagen RGB y una segmentación por
identidad geométrica, mediante `Recognition`. Cada tramo de carril tiene un
color de reconocimiento rojo; ese color se convierte al identificador `1`.
El color de reconocimiento no aparece en la imagen RGB. La etiqueta no se
deduce del color visible del suelo ni de la predicción de otra red.

La geometría se almacena en teselas pequeñas para evitar la limitación de
reconocimiento de superficies grandes alrededor de la cámara. Las máscaras
son de la superficie visible: los objetos que la ocluyen pertenecen al fondo.

## Abrir una pista

Abrir `worlds/lane_dataset.wbt` en Webots. El modo predeterminado muestra una
pista aleatoria y un TurtleBot estacionario. La semilla aparece en la consola.
El robot todavía no sigue el carril: esta vista no conecta un modelo a los motores.

Desde la carpeta de este proyecto, para repetir una pista de vista previa:

```powershell
.\scripts\open_track.ps1 -Seed 4012
```

Para abrir exactamente una pista del dataset:

```powershell
.\scripts\open_track.ps1 -TrackId track_020
```

Para un trazado nuevo, ejecutar `open_track.ps1` sin argumentos. La semilla
global del dataset y la semilla de una pista tienen funciones distintas; las
semillas individuales están en `manifest.json`.

## Regenerar los datos

Requisitos: Webots R2025a y un intérprete ligero de Python con NumPy y OpenCV.
Configurar `controllers/lane_dataset_generator/runtime.ini` a partir de
`runtime.ini.example` con la ruta del intérprete de cada computadora. El archivo
local `runtime.ini` no se versiona.

```powershell
.\scripts\generate_dataset.ps1 -Python "C:\ruta\a\python.exe"
```

La generación no reemplaza directorios existentes. Para otro experimento:

```powershell
.\scripts\generate_dataset.ps1 -Python "C:\ruta\a\python.exe" -Seed 2026 -Output .\data\lane_segmentation_v2
```

Prueba breve en una carpeta nueva:

```powershell
.\scripts\generate_dataset.ps1 -Python "C:\ruta\a\python.exe" -Smoke -SamplesPerTrack 8 -Output .\data\smoke
```

La captura requiere renderizado para la cámara; no desactivarlo. Un error de
captura deja un directorio incompleto para su diagnóstico; no usarlo para
entrenamiento. El conjunto válido debe contener `capture_complete.json` y un
`audit_report.json` con `status: PASS`.

## Analizar antes de entrenar

```powershell
python .\scripts\audit_dataset.py .\data\lane_segmentation_v1
```

Abrir **`supplement_segformer_lane_dataset.ipynb`** con el ambiente de deep
learning del curso. El cuaderno permite examinar pistas, máscaras, particiones,
balance de píxeles y variaciones de pose antes de configurar el entrenamiento.

La auditoría comprueba archivos, hashes, dimensiones, valores de etiqueta,
capturas por pista, geometría y ausencia de fuga por pista. También es necesaria
la inspección visual de las superposiciones; un reporte automático no demuestra
por sí solo la corrección de todos los bordes.

Para entrenar, utilizar las imágenes como entradas y las máscaras como objetivos.
No entregar coordenadas de pista ni segmentación de reconocimiento al modelo
durante la inferencia. En evaluación, medir IoU por clase y, posteriormente,
desviación del robot y recorridos completados. La exactitud por píxel puede ser
engañosa cuando predomina el fondo.

Este es un conjunto sintético inicial, con superficies planas y condiciones
controladas. No demuestra transferencia a cámara real, condiciones climáticas,
obstáculos sobre la ruta o intersecciones. Los fondos y marcas siguen siendo
señales visuales fuertes: evaluar después cambios de apariencia y un conjunto
externo si se pretende medir robustez fuera del generador.

## Distribución

`data/`, modelos entrenados, cachés y rutas locales están excluidos de Git.
El código permite reproducir las pistas; el dataset se distribuye como archivo
comprimido independiente. No es necesario agregar las imágenes al historial
del repositorio.

```powershell
python .\scripts\package_dataset.py
```

El archivo `data/lane_segmentation_v1.zip` conserva la carpeta del dataset.
Descomprimirlo dentro de `data/` en otra computadora antes de abrir el cuaderno.
