param(
    [switch]$SkipDownload,
    [switch]$SkipConvert,
    [string]$PythonVersion = "3.11"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $Root ".venv-vla\Scripts\python.exe"
$SourceModel = Join-Path $Root "models\qwen3-vl-4b-source"
$OutputModel = Join-Path $Root "models\qwen3-vl-4b-int4-ov"

if (-not (Test-Path $VenvPython)) {
    $ManagedPython = Join-Path $env:APPDATA "uv\python\cpython-3.11.15-windows-x86_64-none\python.exe"
    if (Test-Path $ManagedPython) {
        & $ManagedPython -m venv (Join-Path $Root ".venv-vla")
    } else {
        py "-$PythonVersion" -m venv (Join-Path $Root ".venv-vla")
    }
}

& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
& $VenvPython -m pip install -r (Join-Path $Root "requirements-vla.txt")
if ($LASTEXITCODE -ne 0) { throw "runtime dependency installation failed" }
& $VenvPython -m pip install --upgrade-strategy eager `
    "optimum-intel[openvino]>=1.27" `
    "transformers>=4.57,<5" `
    "torch>=2.7" `
    "torchvision>=0.22" `
    "einops>=0.8" `
    "timm>=1.0"
if ($LASTEXITCODE -ne 0) { throw "OpenVINO export dependency installation failed" }
& $VenvPython -m pip install --no-build-isolation --no-deps --force-reinstall `
    "https://github.com/huggingface/optimum-intel/archive/a8c4734741e766ef95d7f1a7d1e29a1d4ba2ab8f.tar.gz#egg=optimum-intel"
if ($LASTEXITCODE -ne 0) { throw "Qwen3-VL export plugin installation failed" }

if (-not $SkipDownload) {
    hf download Qwen/Qwen3-VL-4B-Instruct --local-dir $SourceModel
    if ($LASTEXITCODE -ne 0) { throw "Hugging Face model download failed" }
}

if (-not $SkipConvert) {
    $env:PYTHONIOENCODING = "utf8"
    & (Join-Path $Root ".venv-vla\Scripts\optimum-cli.exe") export openvino `
        --model $SourceModel `
        --task image-text-to-text `
        --weight-format int4 `
        --trust-remote-code `
        $OutputModel
    if ($LASTEXITCODE -ne 0) { throw "OpenVINO INT4 conversion failed" }
}

Write-Output "VLA environment ready: $VenvPython"
Write-Output "OpenVINO model: $OutputModel"
