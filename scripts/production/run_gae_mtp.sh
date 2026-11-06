# !/bin/bash
echo -e "\n--> Generative Recommenders on MTP production environment\n"

echo -e "Period: ${period}"
cat train.config

if [ "$step" == "" ]; then
    echo "[ERROR] Invalid step"
    exit 1
fi
steps=$step

output_dir=$output_dir
model_save_dir=$model_save_dir
echo "output dir：$output_dir"
cur_dir=$(dirname $0)

train_workdir=$(cd $(dirname $0); pwd)
echo "train_workdir: ${train_workdir}"

if [ "$period" == "" ]; then
    # 获取最近数据日期
    date=`ls ${data_dir} | grep '^[0-9]\{8\}$' | awk 'BEGIN {max=0} {if($1>max) max=$1} END {print max}'`
    # 必选参数，用以区分测试日期
    period=$(printf "%s-000000" ${date})
else
    date=${period%-*}
fi
echo "period: ${period}"

version=$version

# 配置CANN相关环境变量
echo -e "\n====== 配置CANN相关环境变量 ======"
CANN_INSTALL_PATH_CONF='/etc/Ascend/ascend_cann_install.info'
sudo cat /etc/Ascend/ascend_cann_install.info
DEFAULT_CANN_INSTALL_PATH="/usr/local/Ascend/"
CANN_INSTALL_PATH="/usr/local/Ascend/"

if [ -d ${CANN_INSTALL_PATH}/ascend-toolkit/latest ];then
  cat ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
  source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
else
  cat ${CANN_INSTALL_PATH}/nnae/set_env.sh
  source ${CANN_INSTALL_PATH}/nnae/set_env.sh
fi
echo -e "\n====== CANN相关环境变量配置完成 ======"

echo -e "\n====== installing python packages ======"
start=`date +%s`


# 指定要检查的文件夹
directory="$output_dir/python_package"

# 检查文件夹是否存在
if [ -d "$directory" ]; then
    echo "python package dir：$directory"
    # 在这里可以执行其他操作
    pip install $output_dir/python_package/*whl
else
    echo "python package dir：$directory"
fi

# 安装 ONNX 导出所需包
pip install onnx
pip install onnxruntime

pip install gin-config absl-py scikit-learn scipy matplotlib numpy apex hypothesis iopath pyarrow
pip install accelerate
end=`date +%s`
runtime=$((end-start))
echo -e "\nTime taken to install python packages: ${runtime} seconds"
echo "====== installed python packages ======"

# 若需使用mxrec加速算子，在MTP平台任务编辑页面，"算法与训练数据"->"输出引用"中，引用'mxrec_acceleration_ops'，"输出目录"同名
# 版本说明："ulan-torch2.1"为torch 2.1.0配套加速算子，"ulan-torch2.6"为torch 2.6配套加速算子
# 当前为临时方案，待包含加速算子的镜像发布后，该段代码可移除
mxrec_acceleration_ops=$mxrec_acceleration_ops
echo "mxrec_acceleration_ops: $mxrec_acceleration_ops"
if [ -n "$mxrec_acceleration_ops" -a -d "$mxrec_acceleration_ops" ]; then
  echo "====== Install mxrec acclelration ops ======"
  pip install $mxrec_acceleration_ops/torch_plugin/*.whl
  for ops_file in `ls -1 $mxrec_acceleration_ops/mxrec_ops/*.run`
  do
    bash $ops_file
  done
  sitepkgs_dir=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")
  cp -v $mxrec_acceleration_ops/torch_library/*.so $sitepkgs_dir
  echo "====== Install mxrec ops finished ======"
fi

pip install torch_geometric polars

# 配置自定义环境变量
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=6000
export HCCL_EXEC_TIMEOUT=6000

# log
export ASCEND_SLOG_PRINT_TO_STDOUT=0   # 日志打屏, 可选
export ASCEND_GLOBAL_LOG_LEVEL=3       # 日志级别常用 1 INFO级别; 3 ERROR级别
export ASCEND_GLOBAL_EVENT_ENABLE=0    # 默认不使能event日志信息
export ASCEND_LAUNCH_BLOCKING=0        # 默认不开启算子下发同步，影响训练性能; 开启后每执行完一个算子会做一次流同步

# 系统默认环境变量，不建议修改
MASTER_HOST="$VC_WORKER_HOSTS"
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
NNODES="$MA_NUM_HOSTS"
NODE_RANK="$VC_TASK_INDEX"
NGPUS_PER_NODE="$MA_NUM_GPUS"
NUM_PROCESSES=$(($NGPUS_PER_NODE * $NNODES))

MASTER_PORT="12345"
JOB_ID="1234"

echo "------> system config <------"
echo "VC_WORKER_HOSTS: ${VC_WORKER_HOSTS}"
echo "MASTER_HOST: ${MASTER_HOST}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NNODES: ${NNODES}"
echo "NODE_RANK: ${NODE_RANK}"
echo "NGPUS_PER_NODE: ${NGPUS_PER_NODE}"
echo "NUM_PROCESSES: ${NUM_PROCESSES}"
echo "${MA_JOB_DIR}"
echo "------>  <------"

# https://www.hiascend.com/document/detail/zh/canncommercial/63RC2/modeldevpt/ptmigr/ptmigr_0022.html
export HCCL_WHITELIST_DISABLE=1

if [[ $NODE_RANK == 0 ]]; then
    EXT_ARGS="--rdzv_conf=is_host=1"
else
    EXT_ARGS=""
fi

# set npu plog env, https://3ms.huawei.com/hi/group/3225441/wiki_6402466.html
ma_vj_name=`echo ${MA_VJ_NAME} | sed 's:ma-job:modelarts-job:g'`
task_name="worker-${VC_TASK_INDEX}"
task_plog_path=${MA_LOG_DIR}/${ma_vj_name}/${task_name}

PIPELINE_FAILED=0

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

if [[ "$version" == "not_use" ]]; then
    model_save_dir=${data_dir}${date}/model
else
    model_save_dir=${data_dir}${date}/model/${version}
    mkdir -p ${model_save_dir}

fi
echo "model_save_dir: ${model_save_dir}"

semantic_map_dir=${semantic_data}/${METHOD_NAME}
echo "sid_map_dir: ${semantic_map_dir}/${version}/${SAVE_MODEL_DIR}"

ALIAS=${version}
SEED=42

DATASET="${data_dir}${date}/${path}"

INTERACTIONS_PATH="${DATASET}"
EMB_PATH="${semantic_data}/${semantic_path}" #/20260301/embeddings/20260301_qwen3_emb_06B_name_categories_desc_ag-rank_20260309_subset_normalized_id.parquet
SAVE_MODEL_DIR="${model_sub_folder:-model}"

run_step "Train RQ-VAE + generate indices (parquet)" \
  python main_gensid.py \
    --config="${train_config_file}" \
    --dataset="${ALIAS}" \
    --embeddings_path="${EMB_PATH}" \
    --interactions_path="${INTERACTIONS_PATH}" \
    --parquet_folder="${INTERACTIONS_PATH}" \
    --saved_model_dir="${SAVE_MODEL_DIR}" \
    --seed="${SEED}" \
    --output_file="${semantic_map_dir}/${version}/${SAVE_MODEL_DIR}/${ALIAS}.index.${SAVE_MODEL_DIR}.parquet"
