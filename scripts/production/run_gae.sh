set -euo pipefail
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=6000
export HCCL_EXEC_TIMEOUT=6000

# log
export ASCEND_SLOG_PRINT_TO_STDOUT=0   # 日志打屏, 可选
export ASCEND_GLOBAL_LOG_LEVEL=3       # 日志级别常用 1 INFO级别; 3 ERROR级别
export ASCEND_GLOBAL_EVENT_ENABLE=0    # 默认不使能event日志信息
export ASCEND_LAUNCH_BLOCKING=0        # 默认不开启算子下发同步，影响训练性能; 开启后每执行完一个算子会做一次流同步

NNODES="1"
NODE_RANK="0"
NGPUS_PER_NODE="1"
NUM_PROCESSES=$(($NGPUS_PER_NODE * $NNODES))

MASTER_PORT="1235"
MASTER_ADDR="127.0.0.1"

PIPELINE_FAILED=0

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

run_step () {
  local step_name="$1"
  shift

  # After first failure, skip all remaining steps.
  [[ "${PIPELINE_FAILED}" -ne 0 ]] && return 0

  echo -e "\n=== ${step_name} ==="

  if "$@"; then
    echo "✅  Step OK: ${step_name}"
  else
    local status=$?
    echo "❌  Step FAILED (exit code ${status}): ${step_name}"
    PIPELINE_FAILED=1
    exit 1
  fi
}

METHOD_NAME="GAE_qwen"
SEED=42

DATASET="service_dis_GR_recall_ulan_preprocess/20260728/tran_orc_to_parquet_irc_preprocessed_2_only_train"
DATASET_DIR="/home/ma-user/work/dataset"

INTERACTIONS_PATH="${DATASET_DIR}/${DATASET}"
EMB_PATH="${DATASET_DIR}/semantic_data/20260301/embeddings/20260301_qwen3_emb_06B_name_categories_desc_sd_20260728_subset.parquet"
SAVE_MODEL_DIR=${METHOD_NAME}
# Flat YAML config (new structure); rename per your naming conventions.
CONFIG="config/rqvae_prod.yaml"

run_step "Train RQ-VAE + generate indices (parquet)" \
  "${RUNNER[@]}" main_gensid.py \
    --config="${CONFIG}" \
    --dataset="${DATASET}" \
    --embeddings_path="${EMB_PATH}" \
    --interactions_path="${INTERACTIONS_PATH}" \
    --parquet_folder="${INTERACTIONS_PATH}" \
    --saved_model_dir="${SAVE_MODEL_DIR}" \
    --seed="${SEED}" \
    "${PASSTHRU[@]}"

# Indices written to cache/gensid/${ALIAS}/${ALIAS}.index.${SAVE_MODEL_DIR}.parquet
# (default output; override with --output_file).
