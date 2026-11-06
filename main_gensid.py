import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from gensid.pipeline import SIDPipeline
from gensid.utils import (
    ensure_dir,
    get_device,
    get_trained_flag,
    init_dist_state,
    init_seed,
    is_distributed,
    yaml_to_argv,
)

logger = logging.getLogger(__name__)

AVAILABLE_SID_MODELS = [
    "rqvae",
    "rkmeans",
    "hierarchical_rkmeans",
    "grqvae",
    "recdmon",
    "spectral",
    "stacked_dmon",
    "pq",
    "opq",
]


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="SID Builder")

    # --- Config Args ---
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file whose keys are translated into CLI args (real CLI args win)")

    # --- General Args ---
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--epochs", type=int, default=20000, help="number of epochs")
    parser.add_argument("--train_batch_size", type=int, default=1024, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Evaluation batch size")
    parser.add_argument("--skip_n_last", type=int, default=None, help="Number of items used for leave-n-out evaluation")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--eval_step", type=int, default=2000, help="eval step")
    parser.add_argument("--learner", type=str, default="AdamW", help="optimizer")
    parser.add_argument("--root_path", type=str, default="./", help="Input data path.")
    parser.add_argument("--saved_model_dir", type=str, default=None, help="Directory to save the model.")
    parser.add_argument("--checkpoint", type=str, default="best_collision_model.pth", help="Checkpoint file name used for generation (e.g. best_collision_model.pth or latest_model.pth)")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name (Beauty, Yelp, MIND, etc).")
    parser.add_argument("--sid_model", type=str, default="rqvae", choices=AVAILABLE_SID_MODELS, help="Model to generate SIDs")
    parser.add_argument("--embedder", type=str, default=None, help="Sentence Embedding model used.")
    parser.add_argument("--output_file", type=str, default=None, help="Output File Name (default: cache/gensid/<dataset>/<dataset>.index.<ckpt_id>.parquet, or .json with --save_json)")
    parser.add_argument("--save_json", action="store_true", help="Write the SID mapping as JSON instead of the default parquet artifact")
    parser.add_argument("--all_items_index", type=str, default=None, help="Parquet file with item_id/raw_id for every dataset item (extends clustering SIDs to OOV items)")
    parser.add_argument("--prefixes", type=str, nargs="+", default=None, help="Prefix format string per SID level (default: <a_{}>, <b_{}>, ...)")
    parser.add_argument("--unique_prefix", type=str, default="<z_{}>", help="Leaf prefix format string used to disambiguate colliding SIDs (clustering methods)")
    parser.add_argument("--unclustered_strategy", type=str, choices=["special", "unique", "error"], default="special", help="How to handle unclustered (-1) labels: special, unique, or error")
    parser.add_argument("--device", type=str, default="npu", help="gpu or cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--logging_level", default=logging.DEBUG)

    # --- Model Specific Args ---
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="l2 regularization weight")
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", action="store_true", help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--no_kmeans_init", action="store_false", dest="kmeans_init", help="do not use kmeans_init")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument("--distance_function", default="euclidean", choices=["euclidean", "cosine"], help="distance function for KMeans")
    parser.add_argument("--mean_function", default="mean", choices=["mean", "normalized_mean"], help="function to compute KMeans centroids")
    parser.add_argument("--kmeans_backend", default="torch", choices=["torch", "faiss"], help="k-means backend for rkmeans / hierarchical_rkmeans (faiss runs on CPU)")
    parser.add_argument("--quantizer_backend", default="torch", choices=["torch", "faiss"], help="quantizer backend for pq / opq SID methods (faiss runs on CPU)")
    parser.add_argument("--n_subquantizers", type=int, default=4, help="number of sub-quantizers / codebooks for pq / opq (in_dim must be divisible by it)")
    parser.add_argument("--n_bits", type=int, default=8, help="bits per sub-code for pq / opq (codebook size = 2**n_bits)")
    parser.add_argument("--opq_niter", type=int, default=25, help="outer rotation-refinement iterations for opq")
    parser.add_argument("--kmeanspp_exact", action="store_true", help="use the exact (bit-identical) full-matrix k-means++ init instead of the fast incremental variant")
    parser.add_argument("--disable_vq_init", action="store_true", help="run one-pass codebook initialization before training")
    parser.add_argument("--norm_kmeans_input", action="store_true")
    parser.add_argument("--sk_epsilons", type=float, nargs="+", default=[0.0, 0.0, 0.0, 0.003], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")
    parser.add_argument("--max_sk_fix_collision_iters", type=int, default=20, help="max sinkhorn iterations post-training for SID generation to reduce collisions")
    parser.add_argument("--fix_collisions", action="store_true", help="run the Sinkhorn 2nd pass to resolve residual SID collisions (off by default: single nearest-neighbour pass, matching GrIS_v1)")
    parser.add_argument("--disable_cf_loss", action="store_true")
    parser.add_argument("--disable_sk", action="store_true")
    parser.add_argument("--num_emb_list", type=int, nargs="+", default=[256, 256, 256, 256], help="emb num of every vq/cluster")
    parser.add_argument("--e_dim", type=int, default=32, help="vq codebook embedding size")
    parser.add_argument("--quant_loss_weight", type=float, default=1.0, help="vq quantion loss weight")
    parser.add_argument("--alpha", type=float, default=0.1, help="cf loss weight (encoder-directed)")
    parser.add_argument("--beta", type=float, default=0.1, help="diversity loss weight")
    parser.add_argument("--n_clusters", type=int, default=10, help="n_clusters")
    parser.add_argument("--sample_strategy", type=str, default="all", help="sample_strategy")
    parser.add_argument("--cf_emb", type=str, default=None, help="cf emb; leave unset to disable CF loss")
    parser.add_argument("--cf_loss_use_codebook_rep", action="store_true", help="compute CF loss on reconstructed codebook representations instead of dense_out")
    parser.add_argument("--cf_loss_last_m_codebooks", type=int, default=0, help="if >0, compute CF loss using only the last M codebooks")
    parser.add_argument("--layers", type=int, nargs="+", default=[2048, 1024, 512, 256, 128, 64], help="hidden sizes of every layer")

    # --- Graph Args ---
    graph_group = parser.add_mutually_exclusive_group()
    graph_group.add_argument("--graph_signal", action="store_true")
    graph_group.add_argument("--graph_augmentation", action="store_true")

    parser.add_argument("--convolution_type", type=str, default="APPNP", help="otherwise time based")
    parser.add_argument("--graph_type", choices=["adjacent_cooc", "adjacent_cooc_prod", "adjacent_prod", "rw", "windowed"], default="adjacent_cooc")
    parser.add_argument("--make_directed", action="store_true", help="build the graph directed")
    parser.add_argument("--add_identity_features", action="store_true", help="attach identity features to graph nodes")

    # Args for the production-scale adjacent co-occurrence graph (adjacent_cooc_prod)
    parser.add_argument("--parquet_folder", type=str, default="", help="Folder containing interaction Parquet files")
    parser.add_argument("--seq_col", type=str, default="history_item_id", help="Column holding one item sequence per row")
    parser.add_argument("--filter_col", type=str, default=None, help="Optional filter column aligned element-wise with seq_col")
    parser.add_argument("--allowed_filter_values", type=int, nargs="+", default=None, help="Keep sequence elements whose filter value is in this set")
    parser.add_argument("--no_directed", action="store_false", dest="directed", default=True, help="Build the graph undirected (directed defaults to True)")
    parser.add_argument("--no_chronological", action="store_false", dest="chronological", default=True, help="Ignore chronological order when building edges (chronological defaults to True)")
    parser.add_argument("--time_col", type=str, default=None, help="Timestamp column (used by time-aware graph types such as session_prod)")
    parser.add_argument("--embeddings_path", type=str, default=None, help="Parquet file with item_id/normalized_id columns (also the dataset embedding source for prod flows)")
    parser.add_argument("--interactions_path", type=str, default=None, help="Interaction data: gensid .inter.json or a parquet folder for prod graph types (default: cache/gensid/<dataset>/<dataset>.inter.json)")
    parser.add_argument("--files_per_chunk", type=int, default=16, help="Max interaction files processed per map chunk")
    parser.add_argument("--max_files", type=int, default=None, help="Max total files to process (None = all)")
    parser.add_argument("--tmp_dir", type=str, default=None, help="Parent dir for temporary partial edge files")

    # Common args for random-walk (rw, default = 2) and windowed (default = 3) graph types
    parser.add_argument("--window_size", type=int, default=None, help="Size of the sliding window for windowed or of the context window for co-occurrence")

    # Args for random-walk (rw) graph type
    parser.add_argument("--walk_length", type=int, default=10, help="Length of each random walk")
    parser.add_argument("--walks_per_item", type=int, default=10, help="Number of walks per item")
    parser.add_argument("--self_loops", action="store_true", help="Allow self-loops in random walks")
    parser.add_argument("--dedup_per_walk", action="store_true", help="Deduplicate pairs within each walk")

    # Args for windowed graph type
    parser.add_argument("--bidirectional", action="store_true", help="Make graph bidirectional")
    parser.add_argument("--dedup_within_window", action="store_true", help="Deduplicate pairs within window")
    parser.add_argument("--decay", type=str, choices=["inverse_distance"], default=None, help="Edge weighting scheme or None")

    ### GNN param block
    parser.add_argument("--gnn_conv_type", type=str, default="gat", help="Either: gat, gcn, gin, gated, or sage")
    parser.add_argument("--gnn_hidden_dim", type=int, default=None, help="GNN hidden dimension")
    parser.add_argument("--gnn_dropout", type=float, default=0.0, help="GNN dropout")
    parser.add_argument("--gnn_heads", type=int, default=4, help="GNN heads (only used by GAT)")
    parser.add_argument("--gnn_layers", type=int, default=2, help="GNN layers")

    ### APPNP param block
    parser.add_argument("--n_propagations", type=int, default=10, help="num propagations")
    parser.add_argument("--teleport_prob", type=float, default=0.1, help="self-loop prob")
    parser.add_argument("--lognorm_weights", action="store_true", help="apply log1p to the edge-reconstruction loss weights")
    parser.add_argument("--use_edge_weight", action="store_true", help="whether to use edge weights")

    # APPNP as augmentation block
    parser.add_argument("--hops", type=int, nargs="+", default=None, help="hops")
    parser.add_argument("--probs", type=int, nargs="+", default=None, help="hops")

    # edge reconstruction block
    parser.add_argument("--plain_reconstruction", action="store_true", help="whether to use edge weights")
    parser.add_argument("--use_edge_reconstruction_loss", action="store_true", help="whether to use reconstruct edges data on quantizer level")
    parser.add_argument("--num_anchors", type=int, default=1024)

    # spectral clustering block
    parser.add_argument("--use_spectral_vectors", action="store_true", help="use spectral vectors with inner quantizer dim")
    parser.add_argument("--cluster_labels_path", type=str, default=None)
    parser.add_argument("--alpha_graph_cluster", type=float, default=0.0, help="graph_cluster loss coef")
    parser.add_argument("--max_cluster_level", type=int, default=None, help="get cluster up to this level")

    # recursive_clustering
    # parser.add_argument("--graph_builder_dump_path", type=str, default=None)

    # Recursive hierarchy.
    parser.add_argument("--recursive_factor", type=int, default=4)
    parser.add_argument("--recursive_levels", type=int, default=2)
    parser.add_argument("--recursive_epochs", type=int, default=100)
    parser.add_argument("--dmon_balance_weight", type=float, default=1.0)
    parser.add_argument("--dmon_entropy_weight", type=float, default=0.01)
    parser.add_argument("--dmon_hidden_dim", type=int, default=256)
    parser.add_argument("--assign_hidden_dim", type=int, default=128)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--max_dense_nodes", type=int, default=8000)
    # fmt: on

    config_args, _ = parser.parse_known_args()
    yaml_argv = []
    if config_args.config:
        with open(config_args.config, encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        yaml_argv = yaml_to_argv(config, parser)

    # Later arguments win, so real CLI arguments override YAML.
    args = parser.parse_args(yaml_argv + sys.argv[1:])
    if args.window_size is None:
        args.window_size = 3 if args.graph_type == "windowed" else 2
    args.spectral_dim = args.e_dim if args.use_spectral_vectors else 0
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    # Setup Paths
    if not args.embeddings_path:
        args.embeddings_path = str(
            Path(args.root_path) / f"cache/gensid/{args.dataset}/{args.dataset}.emb-{args.embedder}-td.npy"
        )
    if not args.interactions_path:
        args.interactions_path = str(Path(args.root_path) / f"cache/gensid/{args.dataset}/{args.dataset}.inter.json")
    args.ckpt_dir = Path(args.root_path) / "ckpt/gensid/quantizer"

    init_seed(args.seed)
    if is_distributed():
        dist_state = init_dist_state(args.device)
        args.device = torch.device(dist_state.device)
    else:
        dist_state = None
        args.device = get_device(args.device)

    assert args.saved_model_dir is not None, "saved_model_dir is required"

    pipeline = SIDPipeline(args, args.device, dist_state=dist_state)

    metrics_path = os.path.join(args.ckpt_dir, args.saved_model_dir or "", "best_collision_model.json")
    trained = get_trained_flag(metrics_path) and not args.overwrite
    if trained:
        if dist_state is None or dist_state.rank == 0:
            logger.info("Model already trained at %s. Skipping training. Use --overwrite to retrain.", metrics_path)
    else:
        logger.info("Starting Training...")
        pipeline.train()
        if dist_state is None or dist_state.rank == 0:
            # Mark as trained
            ensure_dir(os.path.dirname(metrics_path))
            with open(metrics_path, "w") as f:
                json.dump({"fully_trained": True}, f)

    if dist_state is None or dist_state.rank == 0:
        # Generate Indices
        logger.info("Starting Index Generation...")
        ckpt_id = str(args.saved_model_dir).replace("..", "").replace("/", "_")
        if args.output_file:
            output_file = args.output_file
            if not Path(output_file).suffix:
                output_file = str(Path(output_file)) + (".json" if args.save_json else ".parquet")
        else:
            output_dir = Path(args.root_path) / f"cache/gensid/{args.dataset}"
            ext = ".json" if args.save_json else ".parquet"
            output_file = os.path.join(output_dir, f"{args.dataset}.index.{ckpt_id}{ext}")
        ensure_dir(os.path.dirname(output_file) or ".")

        ckpt_path = args.ckpt_dir / args.saved_model_dir / args.checkpoint

        pipeline.generate(str(ckpt_path), output_file)

        logger.info("Indices saved to %s", output_file)

    if dist_state is not None:
        # Keep every rank alive until the leader finishes generation so the group
        # tears down together (avoids "Connection closed by peer" on exit).
        dist.barrier()


if __name__ == "__main__":
    main()
