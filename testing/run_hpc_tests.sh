#!/usr/bin/env bash
set -euo pipefail

echo "Running HPC test suite..."
python testing/test_configs.py
python testing/test_scripts_compile.py
python testing/test_hpc_env_file.py
python testing/test_data_utils.py
python testing/test_validate_processed_data_fake.py
python -c "import datasets; print('datasets ok')"
python -c "import sentence_transformers; print('sentence-transformers ok')"
python -c "import yaml; print('yaml ok')"
python testing/test_gpu.py
echo "HPC tests passed."
