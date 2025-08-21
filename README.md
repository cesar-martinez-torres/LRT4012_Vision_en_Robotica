# 🖥️ CV Environment Starter Kit (Windows) — v3

**Cambios:** ahora el script separa **comando** y **argumentos** (ej. `py` y `-3.12`), evitando el error *'py -3.12' no reconocido*.

## Pasos rápidos
1) Instala Python 3.12 si no lo tienes:
```powershell
winget install -e --id Python.Python.3.12
```
2) Permite scripts (una sola vez):
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
3) Abre PowerShell **dentro de esta carpeta** y ejecuta:
```powershell
.\setup.ps1
```
4) Activa y usa Jupyter:
```powershell
.\.venv\Scripts\Activate.ps1
jupyter lab
```
