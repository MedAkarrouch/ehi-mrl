#!/usr/bin/env bash
set -euo pipefail

echo "Running HPC test suite..."
python testing/test_configs.py
python testing/test_scripts_compile.py
python testing/test_hpc_env_file.py
python testing/test_data_utils.py
python testing/test_validate_processed_data_fake.py
python testing/test_beir_config_loading.py
python testing/test_analyze_qrels_fake.py
python testing/test_phase2_configs.py
python testing/test_retrieval_metrics_fake.py
python testing/test_exact_search_fake.py
python -c "import faiss; print('faiss ok:', faiss.__version__ if hasattr(faiss, '__version__') else 'installed')"
python testing/test_phase3_configs.py
python testing/test_faiss_ivf_fake.py
python testing/test_faiss_flat_check_fake.py
python -c "import matplotlib; print('matplotlib ok:', matplotlib.__version__)"
python testing/test_plot_faiss_ivf_fake.py
python testing/test_plot_faiss_ivf_column_normalization.py
python testing/test_plot_labels.py
python -c "import datasets; print('datasets ok')"
python -c "import sentence_transformers; print('sentence-transformers ok')"
python -c "import yaml; print('yaml ok')"
python testing/test_gpu.py
echo "HPC tests passed."
