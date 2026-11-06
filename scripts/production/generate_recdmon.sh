#!/bin/bash

set -euo pipefail

# NOTE: superseded by the folded recdmon parquet export inside main_gensid.py
# (RecursiveClusteringTrainer writes the OOV-aware dmon_semantic_ids.parquet by
# default). Kept only as a reference for the old standalone contract.


DATASET="/home/ma-user/work/dataset"
labels_path="mincut_runs/dataset_2026-08-25_19-27-45/graph_hierarchy_labels.json"
EMB_PATH="${DATASET}/semantic_data/20260301/embeddings/20260301_qwen3_emb_06B_name_categories_desc_sd_20260728_2.parquet"

python RQ-VAE/generate_recdmon_indices.py \
  --labels_path $labels_path \
  --embedding_table $EMB_PATH \
  --all_items_index "trash/all_items.parquet" \
  --output_file dmon_semantic_ids.parquet # has to be parquet file
