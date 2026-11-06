#!/usr/bin/env bash
# =============================================================================
# scripts/benchmark_prod.sh
#
# A FAIR, production-data benchmark over every GensID branch exercised by
# scripts/smoke_test.sh, all of them sharing the SAME inputs:
#
#   * embeddings    : qwen item embeddings (Parquet, production layout)
#   * interactions  : parquet folder
#                     service_dis_GR_recall_ulan_preprocess/20260728/...
#   * graph building: adjacent_cooc_prod  (the map-reduce prod graph builder)
#
# Fairness rules
#   * --recursive_epochs == --epochs for recdmon / spectral.
#   * num_emb_list (codebook sizes) is shared by every method. recdmon and
#     spectral do not read num_emb_list, so for them it is translated into:
#         recursive_levels = len(num_emb_list)
#         recursive_factor = num_emb_list[0]
#   * The codebook-size grid defaults to "[64, 64, 64]" and "[128, 128]" and is
#     applied to EVERY branch so the comparison shows how each method scales.
#
# CLUSTER_LABELS  (consumed by rqvae_graph_signal_cluster_labels) is derived
# from the RecDMoN run of the same (num_emb_list, seed) group, so recdmon is
# executed BEFORE that variant (its trainer writes
# <outdir>/mincut_runs/recdmon_seed<seed>/graph_hierarchy_labels.json).
#
# Device handling
#   * default device is NPU; on NPU the windowed / rw branches are SKIPPED
#     (those graph builders need the NON-prod .inter.json format).
#   * on cpu (or any non-npu device) they are executed against the same
#     production interaction folder and simply recorded PASS/FAIL.
#
# Output
#   * CSV appended to <script_dir>/benchmark_prod_results.csv (override with
#     BENCH_OUT): num_emb_list, seed, device, variant, sid_model, graph_type,
#     epochs, status, elapsed_s, collision_rate.
#   * A human-readable aligned table is printed on the shell as runs happen.
#
# Usage
#   bash scripts/benchmark_prod.sh                      # NPU, seed 42, epochs 10
#   bash scripts/benchmark_prod.sh --seed=7 --epochs=30
#   bash scripts/benchmark_prod.sh --device=cpu
#   bash scripts/benchmark_prod.sh --num-emb-lists='64 64 64|128 128'
#
# Env
#   DATA_DIR       base dir containing the dataset + semantic_data
#                  (default /home/ma-user/work/dataset)
#   DATASET_REL    interaction folder relative to DATA_DIR
#   EMB_REL        qwen embeddings parquet relative to DATA_DIR
#   PYTHON         python interpreter (default $ROOT/.venv/bin/python)
#   BENCH_OUT      output CSV path (default next to this script)
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# --- defaults -------------------------------------------------------------
DEVICE="${DEVICE:-npu}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-10}"
NEMB_SPEC="64 64 64|128 128"

DATA_DIR="${DATA_DIR:-/home/ma-user/work/dataset}"
DATASET_REL="${DATASET_REL:-service_dis_GR_recall_ulan_preprocess/20260728/tran_orc_to_parquet_irc_preprocessed_2_only_train}"
EMB_REL="${EMB_REL:-semantic_data/20260301/embeddings/20260301_qwen3_emb_06B_name_categories_desc_sd_20260728_subset.parquet}"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
CSV="${BENCH_OUT:-$SCRIPT_DIR/benchmark_prod_results.csv}"
OUT_BASE="${BENCH_RESULTS_DIR:-$ROOT/results/benchmark_prod}"
LOG_BASE="${BENCH_LOGS_DIR:-$ROOT/logs/benchmark_prod}"

# --- CLI parsing (leading --key=value) -----------------------------------
for arg in "$@"; do
  case "$arg" in
    --seed=*)          SEED="${arg#*=}" ;;
    --device=*)        DEVICE="${arg#*=}" ;;
    --epochs=*)        EPOCHS="${arg#*=}" ;;
    --num-emb-lists=*) NEMB_SPEC="${arg#*=}" ;;
    --help|-h)
      sed -n '2,60p' "$0" | grep '^# ' | sed 's/^# \{0,1\}//' | grep -v '^=\{20\}'
      exit 0 ;;
  esac
done

case "$DEVICE" in
  cpu|cuda|npu|auto) ;;
  *) echo "Unknown device '$DEVICE'. Use cpu, cuda, npu, or auto." >&2 && exit 2 ;;
esac

INTERACTIONS_PATH="$DATA_DIR/$DATASET_REL"
EMB_PATH="$DATA_DIR/$EMB_REL"

if [[ ! -d "$INTERACTIONS_PATH" ]]; then
  echo "[warn] interaction folder not found: $INTERACTIONS_PATH" >&2
fi
if [[ ! -f "$EMB_PATH" ]]; then
  echo "[warn] embeddings parquet not found: $EMB_PATH" >&2
fi

CMD=("$PY" main_gensid.py)

# Variants: name|sid_model|graph_type|extra args (extra, if any, overrides BASE).
# rqvae_graph_signal_cluster_labels must stay AFTER recdmon: its @@LABELS@@
# placeholder is filled with the RecDMoN-derived graph_hierarchy_labels.json.
VARIANTS=(
  "rqvae_plain|rqvae|adjacent_cooc_prod|"
  "rqvae_graph_signal|rqvae|adjacent_cooc_prod|--graph_signal --n_propagations 10 --teleport_prob 0.1 --use_edge_weight"
  "rqvae_graph_signal_edge_recon|rqvae|adjacent_cooc_prod|--graph_signal --n_propagations 10 --teleport_prob 0.1 --use_edge_weight --use_edge_reconstruction_loss --num_anchors 16 --lognorm_weights"
  "rqvae_graph_signal_spectral|rqvae|adjacent_cooc_prod|--graph_signal --n_propagations 10 --teleport_prob 0.1 --use_edge_weight --use_spectral_vectors"
  "rqvae_graph_signal_windowed|rqvae|windowed|--graph_signal --graph_type=windowed"
  "rqvae_graph_signal_rw|rqvae|rw|--graph_signal --graph_type=rw --walk_length 2 --walks_per_item 2"
  "rqvae_grqvae|grqvae|adjacent_cooc_prod|--sid_model grqvae --gnn_hidden_dim 512"
  "rqvae_graph_augmentation|rqvae|adjacent_cooc_prod|--graph_augmentation --hops 0 1"
  "rqvae_graph_signal_s2gr|rqvae|adjacent_cooc_prod|--graph_signal --convolution_type S2GR --n_propagations 10 --teleport_prob 0.1 --use_edge_weight"
  "rkmeans|rkmeans|adjacent_cooc_prod|--sid_model rkmeans"
  "rkmeans_hierarchical|hierarchical_rkmeans|adjacent_cooc_prod|--sid_model hierarchical_rkmeans"
  # pq / opq use a fixed n_subquantizers x 2**n_bits codebook structure
  # (--n_subquantizers 4 --n_bits 4, matching the smoke-test defaults) rather than
  # num_emb_list: 4 sub-quantizers divides the production embedding dims
  # (768/1024/2048/4096), and the faiss variants exercise the faiss-cpu backend.
  "pq|pq|adjacent_cooc_prod|--sid_model pq --n_subquantizers 4 --n_bits 4"
  "pq_faiss|pq|adjacent_cooc_prod|--sid_model pq --n_subquantizers 4 --n_bits 4 --quantizer_backend faiss"
  "opq|opq|adjacent_cooc_prod|--sid_model opq --n_subquantizers 4 --n_bits 4 --opq_niter 2"
  "opq_faiss|opq|adjacent_cooc_prod|--sid_model opq --n_subquantizers 4 --n_bits 4 --opq_niter 2 --quantizer_backend faiss"
  "recdmon|recdmon|adjacent_cooc_prod|--sid_model recdmon --max_dense_nodes 100000"
  "rqvae_graph_signal_cluster_labels|rqvae|adjacent_cooc_prod|--graph_signal --n_propagations 10 --teleport_prob 0.1 --use_edge_weight --cluster_labels_path @@LABELS@@ --alpha_graph_cluster 0.1"
  "spectral|spectral|adjacent_cooc_prod|--sid_model spectral --max_dense_nodes 100000"
  "stacked_dmon|stacked_dmon|adjacent_cooc_prod|--sid_model stacked_dmon --max_dense_nodes 100000"
)

# Shared setup for every variant.
BASE=(
  --dataset "$DATASET_REL"
  --embedder qwen3
  --device "$DEVICE"
  --seed "$SEED"
  --epochs "$EPOCHS"
  --eval_step "$EPOCHS"
  --train_batch_size 512
  --eval_batch_size 128
  --overwrite
  --disable_vq_init
  # --fix_collisions  # RQ/GRQ-VAE only; no-op for clustering methods
  --max_sk_fix_collision_iters 5
  --graph_type adjacent_cooc_prod
  --parquet_folder "$INTERACTIONS_PATH"
  --embeddings_path "$EMB_PATH"
  --interactions_path "$INTERACTIONS_PATH"
  --skip_n_last 0
)

# Map a sid_model to its model-sizing args so every method gets the SAME
# codebook structure. recdmon/spectral translate num_emb_list into their
# recursion knobs; recursive_epochs is forced to equal --epochs (fairness).
model_sizing_args() {
  local sid="$1" spec="$2" depth="$3" factor="$4"
  case "$sid" in
    rqvae|grqvae)
      echo --num_emb_list $spec --e_dim 32 --layers 512 256
      ;;
    rkmeans|hierarchical_rkmeans|stacked_dmon)
      echo --num_emb_list $spec
      ;;
    recdmon|spectral)
      echo --recursive_levels "$depth" --recursive_factor "$factor" --recursive_epochs "$EPOCHS"
      ;;
  esac
}

# Read the best collision rate from a run's checkpoint (the *.json sidecar is
# overwritten by main_gensid.py with {"fully_trained": true}, so the value is
# scraped from the *.pth state dict, falling back to the sidecar when present).
collision_from_ckpt() {
  "$PY" - "$1" <<'PY'
import json, os, sys
ckpt_dir = sys.argv[1]
candidates = [
    os.path.join(ckpt_dir, "best_collision_model.pth"),
    os.path.join(ckpt_dir, "latest_model.pth"),
    os.path.join(ckpt_dir, "best_collision_model.json"),
]
for path in candidates:
    if not os.path.exists(path):
        continue
    try:
        if path.endswith(".pth"):
            import torch
            state = torch.load(path, map_location="cpu", weights_only=False)
            value = state.get("best_collision_rate")
        else:
            with open(path) as f:
                value = json.load(f).get("best_collision_rate")
        if isinstance(value, (int, float)) and float(value) != float("inf"):
            print(f"{float(value):.6f}")
            break
    except Exception:
        continue
print("")
PY
}

mkdir -p "$(dirname "$CSV")" "$OUT_BASE" "$LOG_BASE"
printf 'num_emb_list,seed,device,variant,sid_model,graph_type,epochs,status,elapsed_s,collision_rate\n' > "$CSV"

echo "================================================================="
echo "Benchmark config: device=$DEVICE seed=$SEED epochs=$EPOCHS"
echo "Interactions   : $INTERACTIONS_PATH"
echo "Embeddings     : $EMB_PATH"
echo "num_emb_list grid: $NEMB_SPEC"
echo "Graph type     : adjacent_cooc_prod (windowed/rw only on non-NPU)"
echo "CSV            : $CSV"
echo "================================================================="
printf '\n%-12s %-4s %-6s %-38s %-22s %-16s %-6s %-7s %-10s %-10s\n' \
  CONFIG SEED DEVICE VARIANT SID_MODEL GRAPH_TYPE EPOCHS STATUS ELAPSED_S COLLISION

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

# name|sid_model|graph_type|extra
run_variant() {
  local name="$1" sid_model="$2" graph_type="$3" extra="$4"
  local spec="$5" tag="$6" depth="$7" factor="$8"

  local save_dir="benchmark_prod/$tag/${name}_seed${SEED}"
  local out_file="$OUT_BASE/$tag/${name}_seed${SEED}.parquet"
  local log_file="$LOG_BASE/$tag/${name}_seed${SEED}.log"

  # Fairness: never let one variant reuse another's fused-embedding cache.
  rm -f ckpt/gensid/quantizer/graph.npy
  rm -rf ckpt/gensid/quantizer/multihop

  local status="PASS" reason=""

  # NPU skips windowed / rw: their graph builders need the .inter.json format.
  if [[ "$DEVICE" == "npu" && ( "$graph_type" == "windowed" || "$graph_type" == "rw" ) ]]; then
    status="SKIP"
    reason="windowed/rw require non-NPU (JSON interactions format)"
  fi

  # Resolve the cluster-labels path (from the RecDMoN run of this group).
  local extra_resolved="$extra"
  if [[ "$extra" == *"@@LABELS@@"* ]]; then
    local labels="$OUT_BASE/$tag/mincut_runs/recdmon_seed${SEED}/graph_hierarchy_labels.json"
    extra_resolved="${extra//@@LABELS@@/$labels}"
    if [[ ! -f "$labels" ]] && [[ "$status" != "SKIP" ]]; then
      status="FAIL"
      reason="labels missing (recdmon did not finish): $labels"
    fi
  fi

  # Model-sizing args (same num_emb_list structure for every method).
  local -a sizing
  read -r -a sizing <<< "$(model_sizing_args "$sid_model" "$spec" "$depth" "$factor")"
  local -a extra_arr
  read -r -a extra_arr <<< "$extra_resolved"

  local elapsed_ms=0
  if [[ "$status" != "SKIP" && "$status" != "FAIL" ]]; then
    mkdir -p "$LOG_BASE/$tag" "$OUT_BASE/$tag"
    local start_s end_s
    start_s=$(date +%s%N)
    if "${CMD[@]}" "${BASE[@]}" "${sizing[@]}" "${extra_arr[@]}" \
        --saved_model_dir="$save_dir" \
        --output_file="$out_file" \
        >>"$log_file" 2>&1; then
      status="PASS"
    else
      status="FAIL"
      reason="exit code != 0 (see $log_file)"
    fi
    end_s=$(date +%s%N)
    elapsed_ms=$(( (end_s - start_s) / 1000000 ))
  fi

  local collision=""
  if [[ "$status" == "PASS" ]]; then
    collision="$(collision_from_ckpt "ckpt/gensid/quantizer/$save_dir")"
  fi

  local elapsed_sec
  elapsed_sec=$(awk -v ms="$elapsed_ms" 'BEGIN{printf "%.3f", ms/1000}')

  printf '"%s",%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$spec" "$SEED" "$DEVICE" "$name" "$sid_model" "$graph_type" "$EPOCHS" \
    "$status" "$elapsed_sec" "$collision" >> "$CSV"

  printf '%-12s %-4s %-6s %-38s %-22s %-16s %-6s %-7s %-10s %-10s %s\n' \
    "$spec" "$SEED" "$DEVICE" "$name" "$sid_model" "$graph_type" "$EPOCHS" \
    "$status" "$elapsed_sec" "$collision" "$reason"

  case "$status" in
    PASS) PASS_COUNT=$((PASS_COUNT + 1)) ;;
    FAIL) FAIL_COUNT=$((FAIL_COUNT + 1)) ;;
    SKIP) SKIP_COUNT=$((SKIP_COUNT + 1)) ;;
  esac
}

IFS='|' read -r -a NEMB_SPECS <<< "$NEMB_SPEC"
for spec in "${NEMB_SPECS[@]}"; do
  read -r -a nemb_arr <<< "$spec"
  local_tag="n$(printf '%s' "$spec" | tr ' ' 'x')"
  depth=${#nemb_arr[@]}
  factor=${nemb_arr[0]}
  mkdir -p "$OUT_BASE/$local_tag" "$LOG_BASE/$local_tag"
  echo
  echo "--------------- codebook config: $spec (depth=$depth, factor=$factor) ---------------"
  for entry in "${VARIANTS[@]}"; do
    IFS='|' read -r vname vsid vgraph vextra <<< "$entry"
    run_variant "$vname" "$vsid" "$vgraph" "$vextra" "$spec" "$local_tag" "$depth" "$factor"
  done
done

echo
echo "================================================================="
echo "SUMMARY  PASS=$PASS_COUNT FAIL=$FAIL_COUNT SKIP=$SKIP_COUNT (device=$DEVICE seed=$SEED epochs=$EPOCHS)"
echo "CSV saved to: $CSV"
echo "Artifacts   : $OUT_BASE"
echo "Logs        : $LOG_BASE"
echo "================================================================="
if [[ "$FAIL_COUNT" -gt 0 ]]; then
  echo "There were FAILures - inspect the per-variant logs above."
  exit 1
fi

# Keep this line last: satisfies shellcheck/SC2317 complaints about unused
# helper functions is not an issue here, but keeps `bash -n` deterministic.
true
