#!/usr/bin/env bash
set -euo pipefail

echo "Running Codex-safe offline tests..."
python testing/test_configs.py
python testing/test_scripts_compile.py
python testing/test_hpc_env_file.py
python testing/test_data_utils.py
python testing/test_validate_processed_data_fake.py
python testing/test_beir_config_loading.py
echo "Codex-safe tests passed. GPU/HPC tests were not run."
