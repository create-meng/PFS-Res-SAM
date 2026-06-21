#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${1:-$(cd "$SCRIPT_DIR/.." && pwd)}"
STATE_DIR="$REPO_DIR/.setup_state"
STATE_FILE="$STATE_DIR/github_after_clone.done"

cd "$REPO_DIR"
mkdir -p "$STATE_DIR"

mark_done() {
  local step="$1"
  printf '%s\n' "$step" >> "$STATE_FILE"
}

is_done() {
  local step="$1"
  test -f "$STATE_FILE" && grep -Fxq "$step" "$STATE_FILE"
}

echo "[1/5] Installing CUDA 11.8 torch wheels..."
if is_done "torch"; then
  echo "  - already done, skipping"
else
  python -m pip install \
    "torch==2.3.1+cu118" \
    "torchvision==0.18.1+cu118" \
    "torchaudio==2.3.1+cu118" \
    --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu118 \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --timeout 120
  mark_done "torch"
fi

echo "[2/5] Installing remaining Python dependencies..."
if is_done "deps"; then
  echo "  - already done, skipping"
else
  grep -vi "segment-anything" requirements-gpu.txt \
    | grep -Ev "^(--index-url|--extra-index-url|torch|torchvision|torchaudio)" \
    | python -m pip install -r /dev/stdin \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        --extra-index-url https://pypi.org/simple \
        --timeout 120
  mark_done "deps"
fi

echo "[3/5] Installing segment-anything..."
if is_done "segment_anything"; then
  echo "  - already done, skipping"
else
  python -m pip install "segment-anything==1.0" \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://pypi.org/simple \
    --timeout 120
  mark_done "segment_anything"
fi

echo "[4/5] Ensuring git-lfs is available..."
if is_done "git_lfs"; then
  echo "  - already done, skipping"
else
  if ! git lfs version >/dev/null 2>&1; then
    if command -v conda >/dev/null 2>&1; then
      conda install -c conda-forge git-lfs -y
    else
      echo "git-lfs is not available and conda was not found."
      echo "Please install git-lfs first, then rerun this script."
      exit 1
    fi
  fi
  mark_done "git_lfs"
fi

echo "[5/5] SAM checkpoint（GitHub：Git LFS tracks sam/sam_vit_l_0b3195.pth; can exclude and download from Meta URL)"
if is_done "lfs_pull"; then
  echo "  - already done, skipping"
else
  git lfs install
  # Default: ViT-L (~1.2GB). Skip this file and download manually from the Meta official URL
  git lfs pull --exclude="sam/sam_vit_l_0b3195.pth"
  if [ ! -s sam/sam_vit_l_0b3195.pth ]; then
    if [ ! -d sam ] || [ ! -f sam/sam.py ]; then
      echo "ERROR: expected repo-root sam/ with sam/sam.py (clone layout). cwd=$(pwd)" >&2
      exit 1
    fi
    apt-get update && apt-get install -y aria2
    aria2c -x 16 -s 16 -k 1M \
      -o sam_vit_l_0b3195.pth \
      -d sam \
      "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
  fi
  mark_done "lfs_pull"
fi

echo
echo "GitHub post-clone setup completed."
