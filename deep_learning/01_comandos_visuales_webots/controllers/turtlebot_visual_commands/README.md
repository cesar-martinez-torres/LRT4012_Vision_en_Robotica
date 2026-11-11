# Control de movimiento del TurtleBot3 Burger

La red MobileNetV3-Small selecciona `forward`, `left`, `right` o `stop` a partir
de la cámara. El control de movimiento utiliza realimentación de sensores.

- `forward`: avanza 0.42 m, una celda de la pista. Los encoders miden la distancia
  y el controlador corrige la diferencia entre la orientación actual y la referencia.
- `left` y `right`: modifican la referencia de orientación en 90 grados. La unidad
  inercial mide el ángulo y las ruedas reducen su velocidad al acercarse al objetivo.
- `stop`: mantiene ambas ruedas detenidas.

La tolerancia interna es de 0.5 mm para la distancia estimada por encoders y
0.5 grados para la orientación. La distancia física puede diferir por deslizamiento.
El supervisor verifica la posición real después de cada maniobra: permite como
máximo 4 cm de error de posición y 2 grados de orientación. No cambia la posición
del robot durante el recorrido.

Una maniobra se confirma después de seis pasos estables con motores detenidos.
El supervisor presenta el siguiente letrero y autoriza una nueva lectura. Al
terminar la ruta, el robot permanece detenido. Los tiempos máximos son 15 s para
un movimiento y 30 s para reconocer un letrero; un fallo detiene la prueba.

La pista aleatoria incluye `navigation imu` y `command receiver`. En los mundos
anteriores sin unidad inercial, el controlador estima el ángulo con los encoders.

## Ejecución

Abrir `ABRIR_TURTLEBOT_PISTA_ALEATORIA.cmd`. Para repetir una pista concreta:

```powershell
.\ABRIR_TURTLEBOT_PISTA_ALEATORIA.cmd 4012
```

Prueba automática desde la carpeta del proyecto:

```powershell
.\scripts\run_random_track.ps1 -Seed 4012 -Batch
```

El resumen de la prueba se guarda en
`data/turtlebot_route/motion_summary_4012.json`. Contiene los comandos y los
errores físicos de posición y orientación medidos después de cada movimiento.

Los lanzadores establecen `WEBOTS_FORWARD_DISTANCE=0.42` y
`WEBOTS_TURN_ANGLE=90`. La distancia debe coincidir con el tamaño de celda
`CELL_SIZE` del supervisor. Los antiguos tiempos `WEBOTS_FORWARD_STEPS` y
`WEBOTS_TURN_STEPS` ya no determinan la finalización del movimiento.
