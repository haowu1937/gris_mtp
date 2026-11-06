sid_train_conf:
  num_workers: 24
  batch_size: 16_284
  prefetch_factor:  4
  device: "npu:0"
  quant_loss_weight: 0.25
  epochs: 10_000
  e_dim: 64
  alpha: 0.02
  beta: 0.0
  eval_step: 100
  num_prop: 1
  appnp_alpha: 0.1
  lognorm_weights: true
  disable_sk: true
  disable_vq_init: true
  persistent_workers: true
  use_edge_weight: true
  graph_signal: true
  disable_vq_init: true
  cf_loss_use_codebook_rep: true
  use_edge_reconstruction_loss: true
  layers: [512,256,128]
sid_gen_conf:
  checkpoint: "latest_model.pth"
  graph_signal: true

ASCEND_PROCESS_LOG true
ASCEND_PROCESS_LOG_PATH /opt/huawei/schedule-train/log/ascend-log
model_sub_folder full_run_latest_model_sd_recall
path tran_orc_to_parquet_irc_preprocessed_2_only_train
period 20260728-000000
semantic_path 20260301/embeddings/20260301_qwen3_emb_06B_name_categories_desc_sd_20260728_subset.parquet
step train
sys_hang_retry false
train_config_file /opt/huawei/schedule-train/algorithm/train.config
version rqGAE_adj_sd-recall_1prop_0.9appnp_undir_4cb_3Kepochs

path:date_level_train_eval_out_sharded_processed/train