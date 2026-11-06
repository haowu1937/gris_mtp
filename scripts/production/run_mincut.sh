#!/bin/bash

set -euo pipefail

# =========================
# CONFIG
# =========================
DATASET="Instruments"

INTERACTIONS_PATH="data/${DATASET}/${DATASET}.inter.json"
EMB_PATH="data/${DATASET}/${DATASET}.emb-qwen3-td.npy"

OUT_DIR="./mincut_runs/${DATASET}_$(date +%F_%H-%M-%S)"
GRAPH_CACHE="./graph_cache/${DATASET}_adjacent"

DEVICE="cuda:0"

# =========================
# RUN
# =========================
python main_gensid.py \
  --config config/stacked_dmon_prod.yaml \
  --dataset "${DATASET}" \
  --interactions_path "${INTERACTIONS_PATH}" \
  --embeddings_path "${EMB_PATH}" \
  --saved_model_dir "mincut" \
  --device "${DEVICE}" \
  --output_file "${OUT_DIR}/${DATASET}.index.mincut.parquet"

echo "Done. Results in ${OUT_DIR}"
