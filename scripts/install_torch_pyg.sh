#!/usr/bin/env bash
# Run after conda env create -f environment.yml and conda activate toxfm38.
# cu113 reproduces the verified server build; CPU wheels are also available.
set -euo pipefail
variant="${1:-cu113}"
if [[ "$variant" != "cu113" && "$variant" != "cpu" ]]; then
  echo 'Usage: bash scripts/install_torch_pyg.sh [cu113|cpu]' >&2
  exit 2
fi
python -m pip install "torch==1.11.0+$variant" --extra-index-url https://download.pytorch.org/whl/"$variant"
python -m pip install --only-binary=:all: torch-scatter==2.0.9 torch-sparse==0.6.14 torch-cluster==1.6.0 torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.11.0+"$variant".html
python -m pip install torch-geometric==2.0.4
