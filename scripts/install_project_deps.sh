#!/usr/bin/env bash
set -euo pipefail

source configs/hpc_env.sh

echo "Python path: $(which python)"
python --version
if ! python -c "import torch; print('torch version:', torch.__version__)" 2>/dev/null; then
  echo "torch is not installed in the active environment."
fi

python -m pip install --upgrade-strategy only-if-needed -r requirements.txt
bash testing/run_hpc_tests.sh
