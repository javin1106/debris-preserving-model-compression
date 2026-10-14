#!/usr/bin/env bash
# Build env-orig inside WSL2 Ubuntu, following third_party/mados/README.md "Installation" exactly.
# Only addition: Miniforge provides conda (none was installed). Run from the repo root inside WSL.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_DIR="$HOME/miniforge3"

if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$CONDA_DIR"
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"

# --- README commands, verbatim apart from -y and env name ---
conda create -y -n mados python=3.8.12
set +u; conda activate mados; set -u  # gdal activate.d script reads unset vars
conda install -y -c conda-forge gdal==3.3.2
cd "$REPO/third_party/mados"
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 \
    -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html
conda install -y pytables==3.7.0
# --- end README commands ---

python -c "import torch, mmcv; print(torch.__version__, torch.cuda.is_available(), mmcv.__version__)"
conda env export > "$REPO/docs/environment-orig.yml"
