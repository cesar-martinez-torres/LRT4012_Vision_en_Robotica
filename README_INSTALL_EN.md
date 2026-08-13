# Robot Vision — Fall 2026

## 1. Install or verify WinGet

Open **Windows PowerShell** from the Start menu and run:

```powershell
winget --version
```

If a version number appears, continue to the next step. If Windows reports that `winget` is not recognized, install or update **App Installer** from the Microsoft Store. Close PowerShell, open it again, and repeat the command.

## 2. Install or verify Visual Studio Code

In PowerShell, run:

```powershell
code --version
```

If a version number appears, Visual Studio Code is already installed. Otherwise, install it with:

```powershell
winget install -e --id Microsoft.VisualStudioCode --accept-package-agreements --accept-source-agreements
```

Open Visual Studio Code and install these two extensions from the **Extensions** panel:

- **Python**, published by Microsoft.
- **Jupyter**, published by Microsoft.

## 3. Install or verify Git

Git is required to download the repository and obtain course updates. In PowerShell, run:

```powershell
git --version
```

If a version number appears, Git is already installed. Otherwise, install it with:

```powershell
winget install -e --id Git.Git --accept-package-agreements --accept-source-agreements
```

## 4. Install or verify Python 3.13

In PowerShell, run:

```powershell
py -3.13 --version
```

If the displayed version begins with `Python 3.13`, continue to the next step. Otherwise, install it with:

```powershell
winget install -e --id Python.Python.3.13 --accept-package-agreements --accept-source-agreements
```

Do not install the free-threaded Python variant.

After installing Visual Studio Code, Git, and Python, **restart the computer** before continuing. This ensures that all commands and system paths are available.

## 5. Clone the course repository

Using Git is the recommended method because it allows the course material to be updated easily before each class. A GitHub account is **not required** to clone or update this public repository. However, students are encouraged to create a free GitHub account to become familiar with version control and to store their own projects and future coursework.

1. Open Visual Studio Code.
2. Press `Ctrl+Shift+P` to open the Command Palette.
3. Enter `Git: Clone` and select **Git: Clone**.
4. Paste the following repository address:

   ```text
   https://github.com/cesar-martinez-torres/Vision_en_Robotica.git
   ```

5. Select a permanent local location, such as a folder inside `Documents`, for the repository. **Do not place the repository inside OneDrive, Google Drive, Dropbox, or another cloud-synchronized folder.** File synchronization may lock or modify files inside the virtual environment and cause installation or execution errors.
6. When the download finishes, select **Open**.
7. If Visual Studio Code asks whether you trust the authors of the files, select **Yes, I trust the authors**.

### ZIP alternative

If cloning with Git is not possible, open the [course repository](https://github.com/cesar-martinez-torres/Vision_en_Robotica) and select **Code > Download ZIP**. Extract the ZIP file to a permanent location; do not work inside the compressed file.

The ZIP alternative does not support `git pull`. To obtain new course material, download and extract a new copy of the repository.

## 6. Use the personal work folder

The repository includes the following structure exclusively for files created by the student, such as assignments, exams, exercises, personal code, and experiments:

```text
student_work/
└── imagenes/
```

Save all student-created code and notebooks inside `student_work`. Save images created or used specifically for that work inside `student_work/imagenes`. Do not move the official course materials into these folders. This separation keeps the course repository organized and prevents personal changes from interfering with `git pull`.

Files added inside `student_work` are ignored by Git. They remain on the student's computer, are not visible to the instructor, and are not uploaded to GitHub.

## 7. Create the course environment

In Visual Studio Code, select **Terminal > New Terminal**. Confirm that the terminal is located in the `Vision_en_Robotica` folder, and run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
```

The script will:

1. Locate Python 3.13.
2. Create the `.venv` virtual environment.
3. Install the course packages, including OpenCV and Jupyter.
4. Register the **Python (Robot Vision 2026)** notebook kernel.
5. Check the main package imports.

The first installation may take several minutes. Keep the terminal open until `Setup completed successfully` appears.

## 8. Verify the environment

Open `00_check_env.ipynb` in Visual Studio Code. Click the kernel name in the upper-right corner and select:

```text
Python (Robot Vision 2026)
```

Select **Run All**. The notebook should print the package versions and display an image containing the message `Environment OK`.

## Before each class

Open the repository folder in Visual Studio Code and select **Terminal > New Terminal**. Download the latest course material with:

```powershell
git pull
```

After the update finishes, use the official course materials as reference. Save assignments, exams, exercises, and any other student-created code inside `student_work`, with their corresponding personal images inside `student_work/imagenes`.

After updating the repository, open the required notebook and confirm that **Python (Robot Vision 2026)** is selected as its kernel. The setup script does not need to be run again unless the environment is deleted or the course requirements change.

## Troubleshooting

- **`winget` is not recognized:** install or update App Installer from the Microsoft Store and restart the computer.
- **`code` is not recognized after installing Visual Studio Code:** restart the computer and repeat `code --version`.
- **`git` is not recognized:** restart the computer and repeat `git --version`.
- **Python 3.13 was not found:** restart the computer and run `py -3.13 --version`.
- **The existing `.venv` uses another Python version:** delete only the `.venv` folder and run `setup.ps1` again.
- **The notebook uses the wrong kernel:** use the kernel selector in the upper-right corner and choose **Python (Robot Vision 2026)**.
- **Package installation fails:** verify the Internet connection and rerun `setup.ps1`.
- **`git pull` reports local changes or a conflict:** keep a copy of your work under a different filename and ask the instructor before deleting or overwriting files.
