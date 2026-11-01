# One-time setup on Windows: virtual environment + all packages + tests.
#   powershell -ExecutionPolicy Bypass -File setup.ps1            (or double-click setup.bat)
#   powershell -ExecutionPolicy Bypass -File setup.ps1 -Mirror    (pip mirror in mainland China)
#   powershell -ExecutionPolicy Bypass -File setup.ps1 -Quick     (also run a 5-minute end-to-end check)
param([switch]$Mirror, [switch]$Quick)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Find-Python {
    foreach ($v in "3.12", "3.13", "3.11", "3.10") {
        try { $p = & py "-$v" -c "import sys; print(sys.executable)" 2>$null; if ($LASTEXITCODE -eq 0 -and $p) { return $p.Trim() } } catch {}
    }
    foreach ($c in "python", "python3") {
        try { $p = & $c -c "import sys; print(sys.executable)" 2>$null; if ($LASTEXITCODE -eq 0 -and $p) { return $p.Trim() } } catch {}
    }
    throw "Python 3.10-3.13 not found. Install it from https://www.python.org/downloads/ (tick 'Add to PATH') and run again."
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    $py = Find-Python
    Write-Host "creating .venv with $py"
    & $py -m venv .venv
}
$vpy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$idx = @()
if ($Mirror) { $idx = @("-i", "https://pypi.tuna.tsinghua.edu.cn/simple") }

& $vpy -m pip install --upgrade pip @idx
& $vpy -m pip install -r requirements.txt @idx
& $vpy -m pip install -e . --no-deps
& $vpy -c "import torch; print('torch', torch.__version__, '| threads', torch.get_num_threads(), '| cuda', torch.cuda.is_available())"
& $vpy -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "tests failed" }

$found = Get-ChildItem -Path data -Recurse -Include *_4d.nii, *_4d.nii.gz -ErrorAction SilentlyContinue
if (-not $found) {
    Write-Host "`nNo patient data yet. Put an ACDC patient folder into data\acdc\ (see data\README.md)." -ForegroundColor Yellow
} elseif ($Quick) {
    & $vpy scripts\pipeline.py --preset quick
}

Write-Host "`nReady. Next:" -ForegroundColor Green
Write-Host "  .venv\Scripts\activate"
Write-Host "  python scripts\pipeline.py                 # whole study for the first patient in data\ (~1 h)"
Write-Host "  python scripts\pipeline.py --preset quick  # 5-minute check"
