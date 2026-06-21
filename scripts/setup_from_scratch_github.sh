#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${1:-https://github.com/create-meng/PFS-Res-SAM-clear.git}"
TARGET_DIR="${2:-$PWD/PFS-Res-SAM}"

echo "[1/2] Shallow-cloning repo from GitHub with LFS smudge disabled..."
export GIT_LFS_SKIP_SMUDGE=1
if [ -d "$TARGET_DIR/.git" ]; then
  echo "Repo already exists at $TARGET_DIR, skipping clone."
  echo "Updating existing repo (fetch depth=1 on current branch)..."
  BRANCH="$(git -C "$TARGET_DIR" rev-parse --abbrev-ref HEAD)"
  git -C "$TARGET_DIR" fetch --depth 1 origin "$BRANCH"
  git -C "$TARGET_DIR" reset --hard "origin/$BRANCH"
else
  git clone --depth 1 "$REPO_URL" "$TARGET_DIR"
fi

echo "[2/2] Running GitHub post-clone setup..."
bash "$TARGET_DIR/scripts/setup_after_clone_github.sh" "$TARGET_DIR"

echo
echo "GitHub full setup completed."
