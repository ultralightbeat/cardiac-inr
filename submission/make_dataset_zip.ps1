# Builds the dataset archive for the final submission:
#   submission\Group4_Dataset_ACDC_patient051_patient015.zip
# Run from anywhere:  powershell -ExecutionPolicy Bypass -File submission\make_dataset_zip.ps1
$ErrorActionPreference = "Stop"
$sub  = $PSScriptRoot
$proj = Split-Path -Parent $sub
$tmp  = Join-Path $sub "dataset_tmp"
if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
New-Item -ItemType Directory $tmp | Out-Null
foreach ($p in "patient051", "patient015") {
    $src = Join-Path $proj "data\acdc\$p"
    if (-not (Test-Path (Join-Path $src "${p}_4d.nii"))) { throw "missing $src\${p}_4d.nii - copy the patient folder into data\acdc first" }
    Copy-Item $src $tmp -Recurse
}
Copy-Item (Join-Path $sub "DATASET_README.md") (Join-Path $tmp "README.md")
$zip = Join-Path $sub "Group4_Dataset_ACDC_patient051_patient015.zip"
Compress-Archive -Path (Join-Path $tmp "*") -DestinationPath $zip -Force
Remove-Item $tmp -Recurse -Force
Write-Host "done: $zip"
