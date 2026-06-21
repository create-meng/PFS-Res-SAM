#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export BANK_SUFFIX="${BANK_SUFFIX:-_fullset_20260508}"
export RUN_PARALLEL="${RUN_PARALLEL:-1}"
export PARALLEL_WORKERS="${PARALLEL_WORKERS:-2}"
export SKIP_COMPLETED_CASES="${SKIP_COMPLETED_CASES:-1}"
export BUILD_CURRENT_BANK="${BUILD_CURRENT_BANK:-auto}"
export BUILD_V7_BANK="${BUILD_V7_BANK:-auto}"
export BUILD_V18_BANK="${BUILD_V18_BANK:-auto}"
export WORKER_RESTART_DELAY_SEC="${WORKER_RESTART_DELAY_SEC:-5}"
export WORKER_MAX_RESTARTS="${WORKER_MAX_RESTARTS:-20}"
export SUPERVISOR_HEARTBEAT_SEC="${SUPERVISOR_HEARTBEAT_SEC:-30}"
export SUPERVISOR_POLL_SEC="${SUPERVISOR_POLL_SEC:-2}"
export PARALLEL_SNAPSHOT_SEC="${PARALLEL_SNAPSHOT_SEC:-30}"
export WORKER_DYNAMIC="${WORKER_DYNAMIC:-1}"

echo "RepoRoot              = ${REPO_ROOT}"
echo "PYTHON_BIN            = ${PYTHON_BIN}"
echo "BANK_SUFFIX           = ${BANK_SUFFIX}"
echo "RUN_PARALLEL          = ${RUN_PARALLEL}"
echo "PARALLEL_WORKERS      = ${PARALLEL_WORKERS}"
echo "SKIP_COMPLETED_CASES  = ${SKIP_COMPLETED_CASES}"
echo "BUILD_CURRENT_BANK    = ${BUILD_CURRENT_BANK}"
echo "BUILD_V7_BANK         = ${BUILD_V7_BANK}"
echo "BUILD_V18_BANK        = ${BUILD_V18_BANK}"
echo "SUPERVISOR_HEARTBEAT_SEC = ${SUPERVISOR_HEARTBEAT_SEC}"
echo "PARALLEL_SNAPSHOT_SEC = ${PARALLEL_SNAPSHOT_SEC}"
echo
echo "Running full ablation suite on the full dataset..."

"${PYTHON_BIN}" ./experiments/run_ablation_main.py

echo
echo "Done. Check these outputs:"
echo "  ./outputs/ablation_suite/ablation_summary.csv"
echo "  ./outputs/ablation_suite/ablation_summary.md"
echo "  ./outputs/ablation_suite/case_manifest.json"
echo "  ./outputs/ablation_suite/logs/"
