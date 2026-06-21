#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export BANK_SUFFIX="${BANK_SUFFIX:-_fullset_20260508}"
export PARALLEL_WORKERS="${PARALLEL_WORKERS:-2}"
export WORKER_RESTART_DELAY_SEC="${WORKER_RESTART_DELAY_SEC:-5}"
export WORKER_MAX_RESTARTS="${WORKER_MAX_RESTARTS:-20}"
export RUN_PARALLEL=1
export SKIP_COMPLETED_CASES="${SKIP_COMPLETED_CASES:-1}"

CURRENT_BANK="./outputs/feature_banks_v30${BANK_SUFFIX}/feature_bank_v30${BANK_SUFFIX}.pth"
V7_BANK="./outputs/feature_banks_v7/feature_bank_v7.pth"
V18_BANK="./outputs/feature_banks_v18/feature_bank_v18.pth"

CURRENT_CASES="a0_final,a1_wo_m5_secondary_filter,a2_wo_m4_strict_filter,a5_wo_m1_beta_calibration,c1_patchcore_auc,c2_single_scale_auc_control"
V7_CASES="s0_v7_baseline"
V18_CASES="s1_v18_transition"

echo "RepoRoot              = ${REPO_ROOT}"
echo "PYTHON_BIN            = ${PYTHON_BIN}"
echo "BANK_SUFFIX           = ${BANK_SUFFIX}"
echo "PARALLEL_WORKERS      = ${PARALLEL_WORKERS}"
echo "SKIP_COMPLETED_CASES  = ${SKIP_COMPLETED_CASES}"
echo "Current bank          = ${CURRENT_BANK}"
echo "V7 bank               = ${V7_BANK}"
echo "V18 bank              = ${V18_BANK}"
echo

bank_exists() {
    local bank_path="$1"
    [[ -f "${bank_path}" ]]
}

wait_for_bank() {
    local bank_path="$1"
    local bank_name="$2"
    echo "[wait] ${bank_name}: ${bank_path}"
    until bank_exists "${bank_path}"; do
        sleep 5
    done
    echo "[ready] ${bank_name}"
}

build_bank() {
    local version="$1"
    local bank_path=""
    case "${version}" in
        current) bank_path="${CURRENT_BANK}" ;;
        v7) bank_path="${V7_BANK}" ;;
        v18) bank_path="${V18_BANK}" ;;
    esac

    if bank_exists "${bank_path}"; then
        echo "[build] ${version} feature bank reuse: ${bank_path}"
        return 0
    fi

    echo "[build] ${version} feature bank start"
    case "${version}" in
        current)
            "${PYTHON_BIN}" ./experiments/_feature_bank.py
            ;;
        v7)
            git show bc76702:experiments/01_build_feature_bank_v7.py > ./experiments/_materialized_build_v7.py
            "${PYTHON_BIN}" ./experiments/_materialized_build_v7.py
            ;;
        v18)
            git show 2dc7338:experiments/01_build_feature_bank_v18.py > ./experiments/_materialized_build_v18.py
            "${PYTHON_BIN}" ./experiments/_materialized_build_v18.py
            ;;
        *)
            echo "Unknown bank version: ${version}" >&2
            return 1
            ;;
    esac
    echo "[build] ${version} feature bank done"
}

run_cases() {
    local cases="$1"
    local require_bank="$2"
    local bank_name="$3"
    wait_for_bank "${require_bank}" "${bank_name}"
    echo "[run] cases=${cases}"
    BUILD_CURRENT_BANK=0 BUILD_V7_BANK=0 BUILD_V18_BANK=0 \
    ABLATION_CASES="${cases}" "${PYTHON_BIN}" ./experiments/run_ablation_main.py
    echo "[done] cases=${cases}"
}

echo "[stage-1] build current bank first"
build_bank current

echo "[stage-2] run current cases while building v7 bank"
build_bank v7 &
V7_BUILD_PID=$!
run_cases "${CURRENT_CASES}" "${CURRENT_BANK}" "current bank" &
CURRENT_RUN_PID=$!

wait "${CURRENT_RUN_PID}"
wait "${V7_BUILD_PID}"

echo "[stage-3] run v7 case while building v18 bank"
build_bank v18 &
V18_BUILD_PID=$!
run_cases "${V7_CASES}" "${V7_BANK}" "v7 bank" &
V7_RUN_PID=$!

wait "${V7_RUN_PID}"
wait "${V18_BUILD_PID}"

echo "[stage-4] run v18 case"
run_cases "${V18_CASES}" "${V18_BANK}" "v18 bank"

echo
echo "Done. Check these outputs:"
echo "  ./outputs/ablation_suite/ablation_summary.csv"
echo "  ./outputs/ablation_suite/ablation_summary.md"
echo "  ./outputs/ablation_suite/case_manifest.json"
echo "  ./outputs/ablation_suite/logs/"
