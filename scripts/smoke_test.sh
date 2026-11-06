#!/usr/bin/env bash
# Smoke test: run every GensID code path end-to-end via `main_gensid.py` on the
# smallest available cached dataset (Beauty), with minimal epochs, so each
# dataset/builder/graph/encoder branch is exercised quickly.
#
# Every variant keeps its own `saved_model_dir` (=> distinct index JSON), and the
# shared graph caches under ckpt/gensid/quantizer are cleared before each run so
# one variant cannot reuse another's fused-embedding cache.
#
# Usage:
#   bash scripts/smoke_test.sh                      # run all variants on cpu
#   bash scripts/smoke_test.sh rqvae_plain          # run only some variants
#   bash scripts/smoke_test.sh --device=cuda        # run all variants on cuda
#   bash scripts/smoke_test.sh --device=cuda rqvae_plain rkmeans
#   SMOKE_DEVICE=cuda bash scripts/smoke_test.sh    # device via env (fallback)
#
# Exit code is 0 only if every requested variant PASSes.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEVICE="${SMOKE_DEVICE:-cpu}"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
CMD=("$PY" main_gensid.py)

# Parse leading --device=<value>; everything else is a variant name.
requested=()
for arg in "$@"; do
  if [[ "$arg" == --device=* ]]; then
    DEVICE="${arg#*=}"
  else
    requested+=("$arg")
  fi

done
case "$DEVICE" in
  cpu|cuda|npu|auto) ;;
  *) echo "Unknown device '$DEVICE'. Use cpu, cuda, npu, or auto." >&2 && exit 2 ;;
esac
echo "Smoke test device: $DEVICE"

# Shared minimal setup so runs stay fast on CPU.
BASE=(
  --dataset=Beauty
  --embedder=sentence-t5-base
  --device="$DEVICE"
  --seed=42
  --epochs=2
  --eval_step=1
  --train_batch_size=1024
  --eval_batch_size=128
  --overwrite
  --disable_vq_init
  --max_sk_fix_collision_iters=5
)

# Small but viable RQ-VAE model: 256-slot codebooks (matches the k-means init
# hardcoded n_clusters=256), tiny MLP.
RQVAE_MODEL=(--layers 32 16 --e_dim 16 --num_emb_list 256 256 256)

# RecDMoN hierarchical cluster labels produced by the mincut pipeline (cached).
CLUSTER_LABELS=cache/gensid/mincut_runs/RecDMoN_t5_Beauty_seed42/graph_hierarchy_labels.json

# variant_name | extra args (empty => default EmbDataset / rqvae path)
VARIANTS=(
  "rqvae_plain|"
  "rqvae_fix_collisions|--fix_collisions"
  "rqvae_graph_signal|--graph_signal"
  "rqvae_graph_signal_edge_recon|--graph_signal --use_edge_reconstruction_loss --num_anchors 16"
  "rqvae_graph_signal_spectral|--graph_signal --use_spectral_vectors"
  "rqvae_graph_signal_cluster_labels|--graph_signal --cluster_labels_path $CLUSTER_LABELS --alpha_graph_cluster 0.1"
  "rqvae_graph_signal_windowed|--graph_signal --graph_type=windowed"
  "rqvae_graph_signal_rw|--graph_signal --graph_type=rw --walk_length 2 --walks_per_item 2"
  "rqvae_grqvae|--sid_model grqvae --gnn_hidden_dim 4096"
  "rqvae_graph_augmentation|--graph_augmentation --hops 0 1"
  "rqvae_graph_signal_s2gr|--graph_signal --convolution_type S2GR"
  "rkmeans|--sid_model rkmeans"
  "rkmeans_hierarchical|--sid_model hierarchical_rkmeans"
  "pq|--sid_model pq --n_subquantizers 4 --n_bits 4"
  "pq_faiss|--sid_model pq --n_subquantizers 4 --n_bits 4 --quantizer_backend faiss"
  "opq|--sid_model opq --n_subquantizers 4 --n_bits 4 --opq_niter 2"
  "opq_faiss|--sid_model opq --n_subquantizers 4 --n_bits 4 --opq_niter 2 --quantizer_backend faiss"
  "recdmon|--sid_model recdmon --max_dense_nodes 100000"
  "spectral|--sid_model spectral --max_dense_nodes 100000"
  "stacked_dmon|--sid_model stacked_dmon --max_dense_nodes 100000"
)

# Allow selecting variants from the command line (e.g. `bash scripts/smoke_test.sh rqvae_plain`)
if [[ ${#requested[@]} -eq 0 ]]; then
  for v in "${VARIANTS[@]}"; do requested+=("${v%%|*}"); done
fi

PASS=()
FAIL=()

run_variant() {
  local name="$1" extra="$2"
  echo
  echo "==================== SMOKE: $name ===================="
  # Clear shared graph caches so variants don't leak fused embeddings into each other.
  rm -f ckpt/gensid/quantizer/graph.npy
  rm -rf ckpt/gensid/quantizer/multihop

  local args=()
  if [[ "$extra" != *"--sid_model"* ]]; then
    args+=(--sid_model=rqvae "${RQVAE_MODEL[@]}")
  fi
  args+=($extra)  # intentional word splitting of the extra-args string

  if "${CMD[@]}" "${BASE[@]}" "${args[@]}" \
      --saved_model_dir="smoke/$name" --output_file="smoke_$name" 2>&1; then
    PASS+=("$name")
    echo "---- PASS: $name ----"
  else
    FAIL+=("$name")
    echo "---- FAIL: $name ----"
  fi
}

for name in "${requested[@]}"; do
  matched=0
  for v in "${VARIANTS[@]}"; do
    if [[ "${v%%|*}" == "$name" ]]; then
      run_variant "$name" "${v#*|}"
      matched=1
      break
    fi
  done
  if [[ $matched -eq 0 ]]; then
    echo "Unknown variant '$name'. Available: ${VARIANTS[*]%%|*}"
    exit 2
  fi
done

echo
echo "==================== SUMMARY ===================="
echo "PASS (${#PASS[@]}): ${PASS[*]:-}"
echo "FAIL (${#FAIL[@]}): ${FAIL[*]:-}"
if [[ ${#FAIL[@]} -gt 0 ]]; then
  exit 1
fi
