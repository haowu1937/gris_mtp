#!/bin/bash

set -euo pipefail

# =========================
# CONFIG
# =========================
DATASET="service_dis_GR_recall_ulan_preprocess/20260728/tran_orc_to_parquet_irc_preprocessed_2_only_train"
DATASET_DIR="/home/ma-user/work/dataset"

INTERACTIONS_PATH="${DATASET_DIR}/${DATASET}"
# EMB_PATH="${DATASET}/semantic_data/20260301/embeddings/20260301_qwen3_emb_06B_name_categories_desc_sd_20260728_2.parquet"
EMB_PATH="${DATASET_DIR}/semantic_data/20260301/embeddings/20260301_qwen3_emb_06B_name_categories_desc_sd_20260728_subset.parquet"


OUT_DIR="./results/mincut_runs/RecDMoN/${DATASET}_$(date +%F_%H-%M-%S)"
GRAPH_CACHE="./graph_cache/pyg_data.pt"

DEVICE="npu"
SEED=42

# OLD (run_letter.sh:116-121)
SKIP_STEPS=()
for arg in "$@"; do
  case "${arg}" in
    --skip=*) IFS=',' read -r -a SKIP_STEPS <<< "${arg#--skip=}" ;;
  esac
done
# NEW
SKIP_STEPS=()
PDB=0; PYINSTRUMENT=0; PASSTHRU=()
for arg in "$@"; do
  case "${arg}" in
    --skip=*)       IFS=',' read -r -a SKIP_STEPS <<< "${arg#--skip=}" ;;
    --pdb)          PDB=1 ;;
    --pyinstrument) PYINSTRUMENT=1 ;;
    --pdb=*|--pyinstrument=*) ;;
    *)              PASSTHRU+=("${arg}") ;;
  esac
done
if   [[ "$PDB" -eq 1 ]];          then RUNNER=(python -m pdb -c continue)
elif [[ "$PYINSTRUMENT" -eq 1 ]]; then RUNNER=(pyinstrument)
else                               RUNNER=(python); fi
RUNNER_STR="${RUNNER[*]}"
PASSTHRU_STR=""; for a in "${PASSTHRU[@]}"; do PASSTHRU_STR+="${a}"; done

pause_on_exit() {
  status=$?

  echo
  if [ "$status" -eq 0 ]; then
    echo "Done. Results in ${OUT_DIR}"
  else
    echo "FAILED with exit code ${status}"
  fi

  echo
  read -r -p "Press Enter to close terminal..."

  exit "$status"
}

trap pause_on_exit EXIT

# =========================
# RUN
# =========================
"${RUNNER[@]}" main_gensid.py \
  --config config/recdmon_prod.yaml \
  --dataset "${DATASET}" \
  --interactions_path "${INTERACTIONS_PATH}" \
  --embeddings_path "${EMB_PATH}" \
  --parquet_folder "${INTERACTIONS_PATH}" \
  --saved_model_dir "recdmon" \
  --device "${DEVICE}" \
  --output_file "${OUT_DIR}/dmon_semantic_ids.parquet" \
  --seed "${SEED}" \
  "${PASSTHRU[@]}"
