# Visión en Robótica — Otoño 2026

## 1. Instalar o verificar WinGet

Abre **Windows PowerShell** desde el menú Inicio y ejecuta:

```powershell
winget --version
```

Si aparece un número de versión, continúa con el siguiente paso. Si Windows indica que no reconoce `winget`, instala o actualiza **App Installer** desde Microsoft Store. Cierra PowerShell, ábrelo nuevamente y repite el comando.

## 2. Instalar o verificar Visual Studio Code

En PowerShell, ejecuta:

```powershell
code --version
```

Si aparece un número de versión, Visual Studio Code ya está instalado. De lo contrario, instálalo con:

```powershell
winget install -e --id Microsoft.VisualStudioCode --accept-package-agreements --accept-source-agreements
```

Abre Visual Studio Code e instala estas dos extensiones desde el panel **Extensiones**:

- **Python**, publicada por Microsoft.
- **Jupyter**, publicada por Microsoft.

## 3. Instalar o verificar Git

Git es necesario para descargar el repositorio y obtener las actualizaciones del curso. En PowerShell, ejecuta:

```powershell
git --version
```

Si aparece un número de versión, Git ya está instalado. De lo contrario, instálalo con:

```powershell
winget install -e --id Git.Git --accept-package-agreements --accept-source-agreements
```

## 4. Instalar o verificar Python 3.13

En PowerShell, ejecuta:

```powershell
py -3.13 --version
```

Si la versión mostrada comienza con `Python 3.13`, continúa con el siguiente paso. De lo contrario, instálalo con:

```powershell
winget install -e --id Python.Python.3.13 --accept-package-agreements --accept-source-agreements
```

No instales la variante *free-threaded* de Python.

Después de instalar Visual Studio Code, Git y Python, **reinicia la computadora** antes de continuar. Esto garantiza que todos los comandos y las rutas del sistema estén disponibles.

## 5. Clonar el repositorio del curso

Se recomienda utilizar Git porque permite actualizar fácilmente el material del curso antes de cada clase. **No se requiere una cuenta de GitHub** para clonar o actualizar este repositorio público. Sin embargo, se recomienda que los estudiantes creen una cuenta gratuita de GitHub para familiarizarse con el control de versiones y almacenar sus propios proyectos y trabajos futuros.

1. Abre Visual Studio Code.
2. Presiona `Ctrl+Shift+P` para abrir la paleta de comandos.
3. Escribe `Git: Clone` y selecciona **Git: Clone**.
4. Pega la siguiente dirección del repositorio:

   ```text
   https://github.com/cesar-martinez-torres/Vision_en_Robotica.git
   ```

5. Selecciona una ubicación local permanente, como una carpeta dentro de `Documentos`, para guardar el repositorio. **No coloques el repositorio dentro de OneDrive, Google Drive, Dropbox ni otra carpeta sincronizada con la nube.** La sincronización puede bloquear o modificar archivos dentro del entorno virtual y provocar errores de instalación o ejecución.
6. Cuando termine la descarga, selecciona **Abrir**.
7. Si Visual Studio Code pregunta si confías en los autores de los archivos, selecciona **Sí, confío en los autores**.

### Alternativa mediante ZIP

Si no es posible clonar con Git, abre el [repositorio del curso](https://github.com/cesar-martinez-torres/Vision_en_Robotica) y selecciona **Code > Download ZIP**. Extrae el archivo ZIP en una ubicación permanente; no trabajes dentro del archivo comprimido.

La alternativa mediante ZIP no permite usar `git pull`. Para obtener nuevo material del curso, descarga y extrae una nueva copia del repositorio.

## 6. Utilizar la carpeta de trabajo personal

El repositorio incluye la siguiente estructura, destinada exclusivamente a los archivos creados por el estudiante, como tareas, exámenes, ejercicios, código personal y experimentos:

```text
student_work/
└── imagenes/
```

Guarda todo el código y los cuadernos creados por el estudiante dentro de `student_work`. Guarda las imágenes creadas o utilizadas específicamente para esos trabajos dentro de `student_work/imagenes`. No muevas los materiales oficiales del curso a estas carpetas. Esta separación mantiene organizado el repositorio del curso y evita que los cambios personales interfieran con `git pull`.

Git ignora los archivos añadidos dentro de `student_work`. Estos permanecen en la computadora del estudiante, no son visibles para el profesor y no se suben a GitHub.

## 7. Crear el entorno del curso

En Visual Studio Code, selecciona **Terminal > New Terminal**. Confirma que la terminal se encuentre en la carpeta `Vision_en_Robotica` y ejecuta:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
```

El script realizará las siguientes acciones:

1. Localizar Python 3.13.
2. Crear el entorno virtual `.venv`.
3. Instalar los paquetes del curso, incluidos OpenCV y Jupyter.
4. Registrar el kernel **Python (Robot Vision 2026)** para los cuadernos.
5. Comprobar la importación de los paquetes principales.

La primera instalación puede tardar varios minutos. Mantén abierta la terminal hasta que aparezca `Setup completed successfully`.

## 8. Verificar el entorno

Abre `00_check_env.ipynb` en Visual Studio Code. Haz clic en el nombre del kernel, en la esquina superior derecha, y selecciona:

```text
Python (Robot Vision 2026)
```

Selecciona **Ejecutar todo (Run All)**. El cuaderno deberá mostrar las versiones de los paquetes y una imagen con el mensaje `Environment OK`.

## Antes de cada clase

Abre la carpeta del repositorio en Visual Studio Code y selecciona **Terminal > New Terminal**. Descarga el material más reciente del curso con:

```powershell
git pull
```

Cuando termine la actualización, utiliza los materiales oficiales del curso como referencia. Guarda las tareas, exámenes, ejercicios y cualquier otro código creado por el estudiante dentro de `student_work`, con sus imágenes personales correspondientes dentro de `student_work/imagenes`.

Después de actualizar el repositorio, abre el cuaderno requerido y confirma que **Python (Robot Vision 2026)** esté seleccionado como kernel. No es necesario ejecutar nuevamente el script de instalación, a menos que se elimine el entorno o cambien los requisitos del curso.

## Solución de problemas

- **No se reconoce `winget`:** instala o actualiza App Installer desde Microsoft Store y reinicia la computadora.
- **No se reconoce `code` después de instalar Visual Studio Code:** reinicia la computadora y repite `code --version`.
- **No se reconoce `git`:** reinicia la computadora y repite `git --version`.
- **No se encontró Python 3.13:** reinicia la computadora y ejecuta `py -3.13 --version`.
- **El entorno `.venv` existente utiliza otra versión de Python:** elimina únicamente la carpeta `.venv` y ejecuta nuevamente `setup.ps1`.
- **El cuaderno utiliza el kernel incorrecto:** utiliza el selector de kernel en la esquina superior derecha y selecciona **Python (Robot Vision 2026)**.
- **Falla la instalación de paquetes:** verifica la conexión a Internet y ejecuta nuevamente `setup.ps1`.
- **`git pull` informa cambios locales o un conflicto:** conserva una copia de tu trabajo con un nombre de archivo diferente y consulta al profesor antes de eliminar o sobrescribir archivos.
