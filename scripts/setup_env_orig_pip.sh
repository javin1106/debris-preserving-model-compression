#!/usr/bin/env bash
# Second half of the env-orig install (README pip step), made robust to a slow / dropping network:
# the 1.9 GB torch wheel is fetched with resumable curl first, then pip installs requirements.txt
# with that local wheel available (-f ~/wheels). Package versions are exactly those in requirements.txt.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p ~/wheels && cd ~/wheels
U="https://download.pytorch.org/whl/cu113/torch-1.11.0%2Bcu113-cp38-cp38-linux_x86_64.whl"
F="torch-1.11.0+cu113-cp38-cp38-linux_x86_64.whl"
for i in $(seq 1 200); do
    curl -sfL -C - --max-time 900 -o "$F" "$U" && break
    sleep 5
done
ls -la ~/wheels
python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).testzip(); print('wheel ok')" "$F" || exit 1

source ~/miniforge3/etc/profile.d/conda.sh
set +u; conda activate mados; set -u  # gdal activate.d script reads unset vars
cd "$REPO/third_party/mados"
rc=1
for i in 1 2 3 4 5; do
    pip install --progress-bar off --timeout 120 --retries 10 -f ~/wheels -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu113 \
        -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html && { rc=0; break; }
done
echo "PIP_EXIT=$rc"
[ $rc -eq 0 ] || exit 1
conda install -y pytables==3.7.0
python -c "import torch, mmcv; print(torch.__version__, torch.cuda.is_available(), mmcv.__version__)"
conda env export > "$REPO/docs/environment-orig.yml"
echo "SETUP_DONE"
