$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

$env:PYTHONUNBUFFERED = "1"
$env:RES_SAM_SKIP_FAISS_PREFLIGHT = $env:RES_SAM_SKIP_FAISS_PREFLIGHT
$env:RES_SAM_ALLOW_SKLEARN_KNN = $env:RES_SAM_ALLOW_SKLEARN_KNN

if (-not $env:BANK_SUFFIX) {
    $env:BANK_SUFFIX = "_fullset_20260508"
}

if (-not $env:RUN_PARALLEL) {
    $env:RUN_PARALLEL = "1"
}

if (-not $env:PARALLEL_WORKERS) {
    $env:PARALLEL_WORKERS = "2"
}

if (-not $env:SKIP_COMPLETED_CASES) {
    $env:SKIP_COMPLETED_CASES = "0"
}

if (-not $env:BUILD_CURRENT_BANK) {
    $env:BUILD_CURRENT_BANK = "1"
}

if (-not $env:BUILD_V7_BANK) {
    $env:BUILD_V7_BANK = "1"
}

if (-not $env:BUILD_V18_BANK) {
    $env:BUILD_V18_BANK = "1"
}

if (-not $env:WORKER_RESTART_DELAY_SEC) {
    $env:WORKER_RESTART_DELAY_SEC = "5"
}

if (-not $env:WORKER_MAX_RESTARTS) {
    $env:WORKER_MAX_RESTARTS = "20"
}

Write-Host "RepoRoot              = $RepoRoot"
Write-Host "BANK_SUFFIX           = $env:BANK_SUFFIX"
Write-Host "RUN_PARALLEL          = $env:RUN_PARALLEL"
Write-Host "PARALLEL_WORKERS      = $env:PARALLEL_WORKERS"
Write-Host "SKIP_COMPLETED_CASES  = $env:SKIP_COMPLETED_CASES"
Write-Host "BUILD_CURRENT_BANK    = $env:BUILD_CURRENT_BANK"
Write-Host "BUILD_V7_BANK         = $env:BUILD_V7_BANK"
Write-Host "BUILD_V18_BANK        = $env:BUILD_V18_BANK"
Write-Host ""
Write-Host "Running full ablation suite on the full dataset..."

conda run -n pfs-res-sam python .\experiments\run_ablation_main.py

Write-Host ""
Write-Host "Done. Check these outputs:"
Write-Host "  .\outputs\ablation_suite\ablation_summary.csv"
Write-Host "  .\outputs\ablation_suite\ablation_summary.md"
Write-Host "  .\outputs\ablation_suite\case_manifest.json"
Write-Host "  .\outputs\ablation_suite\logs\"
