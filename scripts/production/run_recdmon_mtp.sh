# !/bin/bash
echo -e "\n--> Generative Recommenders on MTP production environment\n"

echo -e "Period: ${period}"
cat train.config

if [ "$step" == "" ]; then
    echo "[ERROR] Invalid step"
    exit 1
fi

echo "output dir: $output_dir"

if [ "$period" == "" ]; then
    date=`ls ${data_dir} | grep '^[0-9]\{8\}$' | awk 'BEGIN {max=0} {if($1>max) max=$1} END {print max}'`
    period=$(printf "%s-000000" ${date})
else
    date=${period%-*}
fi
echo "period: ${period}"

sudo cat /etc/Ascend/ascend_cann_install.info
CANN_INSTALL_PATH="/usr/local/Ascend/"

if [ -d ${CANN_INSTALL_PATH}/ascend-toolkit/latest ]; then
  source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
else
  source ${CANN_INSTALL_PATH}/nnae/set_env.sh
fi

echo -e "\n====== installing python packages ======"
start=`date +%s`

if [ -d "$output_dir/python_package" ]; then
    echo "python package dir: $output_dir/python_package"
    pip install $output_dir/python_package/*whl
fi

pip install onnx onnxruntime
pip install gin-config absl-py scikit-learn==1.5.1 scipy==1.10.1 matplotlib==3.7.3 apex hypothesis iopath pyarrow==14.0.1
pip install accelerate==0.34.1
end=`date +%s`
echo -e "\nTime taken to install python packages: $((end-start)) seconds"
echo "====== installed python packages ======"

# 若需使用mxrec加速算子，在MTP平台任务编页面，"算法与训练数据"->"输出引用"中，引用'mxrec_acceleration_ops'，"输出目录"同名
# 版本说明："ulan-torch2.1"为torch 2.1.0配套加速算子，"ulan-torch2.6"为torch 2.6配套加速算子
# 当前为临时方案，待包含加速算子的镜像发布后，该段代码可移除
# echo "mxrec_acceleration_ops: $mxrec_acceleration_ops"
# if [ -n "$mxrec_acceleration_ops" -a -d "$mxrec_acceleration_ops" ]; then
#   echo "====== Install mxrec acceleration ops ======"
#   pip install $mxrec_acceleration_ops/torch_plugin/*.whl
#   for ops_file in `ls -1 $mxrec_acceleration_ops/mxrec_ops/*.run`
#   do
#     bash $ops_file
#   done
#   sitepkgs_dir=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")
#   cp -v $mxrec_acceleration_ops/torch_library/*.so $sitepkgs_dir
#   echo "====== Install mxrec ops finished ======"
# fi

pip install torch_geometric polars==1.43.2
pip install numpy==1.26.4

export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=6000
export HCCL_EXEC_TIMEOUT=6000
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=0
export ASCEND_LAUNCH_BLOCKING=0

echo "------> system config <------"
echo "VC_WORKER_HOSTS: ${VC_WORKER_HOSTS}"
echo "MASTER_ADDR: ${VC_WORKER_HOSTS%%,*}"
echo "MASTER_PORT: 12345"
echo "NNODES: ${MA_NUM_HOSTS}"
echo "NODE_RANK: ${VC_TASK_INDEX}"
echo "NGPUS_PER_NODE: ${MA_NUM_GPUS}"
echo "${MA_JOB_DIR}"
echo "------>  <------"

run_step () {
  local step_name="$1"
  shift

  echo -e "\n=== ${step_name} ==="

  if "$@"; then
    echo "Step OK: ${step_name}"
  else
    echo "Step FAILED (exit code $?): ${step_name}"
    exit 1
  fi
}

# =========================
# DELETE
# train_config_file="configs/recdmon_config_ref.yaml"

# semantic_data="/home/ma-user/work/dataset/semantic_data/"
# semantic_path="20260301/embeddings/20260301_qwen3_emb_06B_name_categories_desc_sd_20260728_subset.parquet"

# data_dir="/home/ma-user/work/dataset/service_dis_GR_recall_ulan_preprocess/"
# date="20260728"
# path="tran_orc_to_parquet_irc_preprocessed_2_only_train"
# version="test_run"
# =========================

# =========================
# CONFIG
# =========================
ALIAS="DMON"
RQVAE_DIR="./RQ-VAE"
DATASET="${data_dir}/${date}/${path}"
all_items_index_path="${data_dir}/${date}/all_items.parquet"
INTERACTIONS_PATH="${DATASET}"
EMB_PATH="${semantic_data}/${semantic_path}"

OUT_DIR="${semantic_data}/${ALIAS}/new_GrIS_recdmon_${version}"
GRAPH_CACHE="${OUT_DIR}/pyg_data.pt"

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
# RUN (clustering + parquet index generation are folded into main_gensid.py)
# =========================
python main_gensid.py \
  --config="${train_config_file}" \
  --dataset="${ALIAS}" \
  --interactions_path="${INTERACTIONS_PATH}" \
  --embeddings_path="${EMB_PATH}" \
  --parquet_folder="${INTERACTIONS_PATH}" \
  --all_items_index="${all_items_index_path}" \
  --saved_model_dir="${version}" \
  --device="npu" \
  --output_file="${OUT_DIR}/dmon_semantic_ids.parquet"
