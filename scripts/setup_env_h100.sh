#!/usr/bin/env bash
# Build env-h100: PyTorch 2.x + CUDA 12 build, mmcv 1.7.x lite (no compiled ops), timm 0.4.12, GDAL.
# Used on the H100 server and (as a dry run) on the laptop. Run from the repo root.
#   bash scripts/setup_env_h100.sh [ENV_NAME]      (default: mados-h100)
# Versions are pinned here; the resolved set is exported to docs/environment-h100-<host>.txt.
set -eo pipefail
ENV="${1:-mados-h100}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_DIR="${CONDA_DIR:-$HOME/miniforge3}"

if ! command -v conda >/dev/null 2>&1 && [ ! -x "$CONDA_DIR/bin/conda" ]; then
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$CONDA_DIR"
fi
[ -x "$CONDA_DIR/bin/conda" ] && source "$CONDA_DIR/etc/profile.d/conda.sh"

conda create -y -n "$ENV" -c conda-forge python=3.10 gdal=3.6.4
conda activate "$ENV"

# Large wheels first, resumable (slow/flaky networks), then pip resolves the rest from the local copies.
mkdir -p ~/wheels
pip download --no-deps --timeout 120 --retries 20 -d ~/wheels \
    torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install --timeout 120 --retries 20 -f ~/wheels \
    torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.org/simple
pip install --timeout 120 --retries 20 \
    numpy==1.26.4 mmcv==1.7.2 yapf==0.40.1 timm==0.4.12 \
    opencv-python-headless==4.8.1.78 rasterio==1.3.9 pandas==2.0.3 scikit-learn==1.3.2 scipy==1.11.4 \
    scikit-image==0.22.0 h5py==3.10.0 tqdm==4.66.2 tensorboard==2.14.0 pyyaml==6.0.1 pytest==7.4.4 \
    matplotlib==3.8.4 fvcore==0.1.5.post20221221

bash "$REPO/scripts/apply_patches.sh"
python - <<'EOF'
import torch, mmcv, timm
from osgeo import gdal
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(),
      "| mmcv", mmcv.__version__, "| timm", timm.__version__, "| gdal", gdal.__version__)
try:
    import mmcv.ops  # noqa: F401
    print("WARNING: mmcv.ops importable -> this is not the lite package")
except Exception as e:
    print("mmcv.ops not available (expected for lite):", type(e).__name__)
EOF
cd "$REPO"
python -c "
from src.debriskd.official import MariNext
import torch
m = MariNext(11, 15)
print('MariNeXt params', sum(p.numel() for p in m.parameters()))
"
pip freeze > "$REPO/docs/environment-h100-$(hostname).txt"
echo "SETUP_H100_DONE"
