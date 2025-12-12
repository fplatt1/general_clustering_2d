# Bachelorarbeit

This project uses `uv` for dependency management.

## Setup

1. **Install `uv`**

   If you don't have `uv` installed, you can install it with:

   ```sh
   pip install uv
   ```

2. **Create a virtual environment**

   Create a virtual environment in the project root:

   ```sh
   uv venv
   ```

3. **Activate the virtual environment**

   Activate the virtual environment:

   - On macOS and Linux:
     ```sh
     source .venv/bin/activate
     ```
   - On Windows:
     ```sh
     .venv\Scripts\activate
     ```

4. **Install dependencies**

   Install the project dependencies with `uv`:

   ```sh
   uv sync
   ```

## Running the application

To run the Streamlit application, use the following command:

```sh
uv run streamlit run app.py
```

**Building A Windows EXE**

- **Goal:** Create a single executable so non-technical users can run the app without Python.
- **Script:** Use the PowerShell helper `build_exe.ps1` included in the repository.
- **How to build (Windows, PowerShell):**

```powershell
# Run once to build (creates/uses .venv, installs deps and PyInstaller)
.
\build_exe.ps1

# Clean build artifacts and rebuild:
.
\build_exe.ps1 -Clean
```

- **Result:** The produced EXE will be placed in the `dist` folder (e.g. `dist\run_app.exe`).
- **Notes:**
   - The EXE bundles the Python runtime and required packages via PyInstaller; the build can take several minutes.
   - You can still run the app from VS Code / development environment with:

```powershell
uv run streamlit run app.py
```

If you need an automated CI build (GitHub Actions) for Windows, I can add an example workflow.
