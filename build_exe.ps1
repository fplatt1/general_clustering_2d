param(
    [string]$VenvName = ".venv",
    [switch]$Clean
)

Set-StrictMode -Version Latest
Write-Host "== Build EXE for Streamlit app =="

if ($Clean) {
    Write-Host "Cleaning previous build artifacts..."
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue .\dist, .\build, .\__pycache__, .\run_app.spec
}

if (-Not (Test-Path $VenvName)) {
    Write-Host "Creating virtual environment: $VenvName"
    python -m venv $VenvName
}

Write-Host "Activating virtual environment..."
& "$VenvName\Scripts\Activate.ps1"

Write-Host "Ensuring pip is available in the venv (will run ensurepip if necessary)..."
& "$VenvName\Scripts\python.exe" -m ensurepip --upgrade | Out-Null

Write-Host "Upgrading pip and installing build dependencies..."
& "$VenvName\Scripts\python.exe" -m pip install --upgrade pip
& "$VenvName\Scripts\python.exe" -m pip install -r requirements.txt pyinstaller

# Files and folders to include inside the frozen app under sys._MEIPASS/app_content
$addData = @(
    "app.py;app_content",
    "funktionen_streamlit.py;app_content",
    "run_app.py;app_content",
    "pages;app_content/pages",
    "bachelorarbeit;app_content/bachelorarbeit",
    "funktionen;app_content/funktionen",
    "data;app_content/data",
    "README.md;app_content/README.md"
)

$addArgs = $addData | ForEach-Object { "--add-data `"$_`"" }
$addArgs = $addArgs -join " "

Write-Host "Running PyInstaller (this can take several minutes)..."

# Prefer a .spec if present (better control over hiddenimports & datas)
if (Test-Path run_app.spec) {
    Write-Host "Found run_app.spec - using spec for build"
    # When passing a .spec file, don't include makespec-only options like --onefile
    $pyinstallerCmd = "pyinstaller --noconfirm --clean run_app.spec"
} else {
    # Use --onefile to create a single EXE. We keep console so Streamlit output is visible.
    $pyinstallerCmd = "pyinstaller --noconfirm --clean --onefile $addArgs run_app.py"
}
Write-Host $pyinstallerCmd

Invoke-Expression $pyinstallerCmd

if ($LASTEXITCODE -ne 0) {
    Write-Host "PyInstaller failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Build finished. The EXE is in the 'dist' folder." -ForegroundColor Green
