# AGENTS.md — GrIS (Graph-Informed Semantic IDs)

## What this repo is

GrIS is a research codebase for generating **Semantic IDs (SIDs)** — hierarchical, discrete item identifiers used to train generative recommendation models (TIGER / MemGen-GR lineage). It is the official implementation of *"Neither Black nor White: Balancing Semantic and Collaborative Signals with Graph-Informed Semantic IDs (GrIS)"*.

Its mission, refactored explicitly:

> A **single, shareable codebase for SID generation**: system-design-oriented, with full tests on all main functionality, and easy to extend by plugging in a new SID method (with its trainer/pipeline).

Long-term (see `docs/TODO.md` #9): this repo will be unified with the **`gr_unified`** generative-recommendation repo (in codehub) so the two behave like **two submodules of a larger org repo**: this repo **produces** SID indices; `gr_unified` **consumes** them to train recommendation models.

## Status & important context for agents

- This repo started as an **old, ad-hoc research codebase** and is now mid-way through a structured restructure. The refactoring roadmap lives in `docs/TODO.md` and the **restructure plan/status lives in `docs/restructure.md`** — **read both before doing significant work**; `docs/TODO.md` is the road map (ordered items #1–#10), `docs/restructure.md` is where code physically lives today.
- The restructure's **Phases 1–4 have largely landed** (see `docs/restructure.md` §5): `SIDBuilder` is gone, replaced by `SIDPipeline` + `SIDMethod`; config is a typed `config/` dataclass package; a `Registry` exists; a `tests/` suite is green (≈60% coverage). What remains: folding the bespoke monofiles into registered methods, deriving the CLI choices from `Registry.keys()`, config JSON sidecars, and closing coverage gaps.
- **`genrec/` is NOT part of this project.** It was dropped in *only* to share the entire codebase for research purposes and **will be removed in the future**. Do not treat it as the thing to develop. It is still valuable as:
  - a reference for the **intended goal** of the whole pipeline (SID → generative recommendation),
  - documentation of the **consumer contract** (`.sem_ids`, `rq_n_codebooks`, `rq_codebook_size`),
  - a model for the clean architecture that `gensid` should mirror (`Pipeline`/`AbstractModel`/`AbstractTokenizer` + registries + YAML/dict config).
- The end-to-end intent is best understood from `scripts/research/*.sh` (hand-rolled orchestration) and `scripts/smoke_test.sh` (in-repo end-to-end smoke over every `main_gensid.py` branch).

## Repository layout

```
main_gensid.py            CLI for gensid: flat argparse -> SIDPipeline(args).train()/.generate() (index JSON)
main_genrec.py            CLI for genrec (accelerate-based generative rec training)
prepare_dataset.py        GR dataset -> gensid inputs (.npy embeddings + .inter.json) via genrec infra
map_gensid_to_genrec.py   SID JSON -> GR .sem_ids + shell env handoff file
map_cf_embeddings.py      extract SASRec CF embeddings from a genrec checkpoint -> gensid ckpt
pyproject.toml            deps, uv config, ruff/pytest config (ruff: line-length 120, double quotes)
requirements.txt          legacy pip pin-list (conda README route)

gensid/                   THE core package (the real product) — restructured, see docs/restructure.md
  registry.py             Registry + REGISTRY hub; package self-discovery (iter_subclasses) for: sid_method,
                          trainer, encoder_config, graph_building_config, dataset_config, trainer_config, dataset.
                          get_<group>[_config]() / keys(group) accessors.
  pipeline.py             SIDPipeline: the one orchestrator. _build_model/_build_trainer/_build_dataset_and_loader,
                          train() and generate(); replaces the old SIDBuilder.
  config/                 typed dataclass package (was a single config.py)
    graph_build_config.py   GraphBuildingConfig (graph_type discriminant) + AdjacentCooc/Prod/RandomWalk(rw)/Windowed
    encoder_config.py       EncoderConfig + GraphEncoderConfig + APPNP/S2GR/GNNFusion/GraphSage (name discriminant)
    dataset_config.py       DatasetConfig + GraphDatasetConfig + Plain/GraphFusion/MixedDepth/GRQVAEFusion
    model_config.py         ModelConfig + RQVAE/GRQVAE/RKMeans/HierarchicalRKMeans
    trainer_config.py       TrainerConfig + RQ/RKMeans/HierarchicalRKMeans
    docstring.py            auto_docstring: splices config Args: blocks onto model class docstrings
    __init__.py             re-exports the 5 base configs + _CONFIGS
  eval.py                 SID-quality metrics: utilization, perplexity, collision rate, Gini (pure)
  utils.py                collision helpers, device selection (cuda/npu/cpu/auto), seeding, get_trained_flag
  data/
    base.py               dataset resolution (TODO #2): get_dataset_config(args) -> DatasetConfig,
                          build_dataset(), build_data_loader() (NeighborLoader for GRQVAE train;
                          EdgeWeightCollator when --use_edge_reconstruction_loss). Keyed-dict batches.
    dataset.py            EmbDataset, GraphAwareEmbDataset, GNNFusionEmbDataset, GRQVAEFusionEmbDataset,
                          MixedFusionDepthDataset (graph-aware ones take graph_building_config + encoder_config)
    graph.py              graph builder fns (adjacent_cooc/prod, rw, windowed; config-based) + get_graph_builder
                          + polars_graph_to_pyg + compute_graph_spectral_vectors + InteractionGraphBuilder
                          (the old graph_aware_sid/ package is fully folded in here)
    utils.py              EdgeWeightCollator + load_train_item_mask + load_graph_cluster_labels
  generation/
    base.py               encode driver: collect_indices / report_and_save / save_generation_mapping
    utils.py              format_prefix (single formatter) + cluster-label -> codes helpers (build_codes, ...)
  method/                 SID method models only (flat for now; target: methods/<name>/ dirs)
    base.py               SIDMethod (nn.Module): codebook_sizes/n_codebooks/generate/compute_loss +
                          name/trainer class attrs; registry-discovered
    rqvae.py rkmeans.py grqvae.py   named SIDMethod subclasses (GRQVAE = RQVAE + nn/gnn.GNNFusionEncoder;
                          rkmeans.py also holds HierarchicalRKMeans)
    __init__.py           re-exports; utils.py empty
  nn/                     SHARED non-method building blocks
    gnn.py                GNNFusionEncoder, GatedGraphNet, APPNPConv, S2GRConv (config-based nn layers)
    layers.py             ResidualVectorQuantizer, VectorQuantizer, TorchKMeans, MLPLayers
    utils.py              DISTANCE_/MEAN_FUNCTIONS + constrained_km + sinkhorn_algorithm + activation_layer
    __init__.py           get_encoder(name -> conv class: none/appnp/s2gr)  [interim]
  trainer/
    base.py               Trainer (typed TrainerConfig: epoch loop, collision-based ckpt selection,
                          metrics JSON sidecar)
    rq_trainer.py         RQTrainer (VQ init pass, two-pass Sinkhorn collision generation, fused-cache save)
    rkmeans_trainer.py    RKMeansTrainer / HierarchicalRKMeansTrainer (fit + generate_indices)
    __init__.py           re-exports
  hierarchical_mincut_clustering.py   coarse-to-fine cluster SIDs: stacked_dmon / recursive_spectral /
                                      recursive_dmon (bespoke CLI, not on the SIDMethod interface yet)
  mmgrec_gnn_sid_adapter.py           GNN (SAGE + BPR) -> toy RQ-VAE -> SIDs (bespoke CLI; toy RQVAE :157
                                      and own DEFAULT_PREFIXES / load_interactions still to fold)
  generate_graph_cluster_indices.py   cluster labels -> SID mapping JSON (thin CLI; pure logic in generation/utils.py)

tests/                    pytest suite — config/dataset/graph/nn/method/trainer/utils (≈60% coverage);
                          pyproject.toml addopts run pytest with --cov

scripts/
  research/               end-to-end experiment pipelines: run_letter.sh, run_gae_components_ablation.sh,
                          run_mmgrec.sh, run_recdmon.sh, run_s2gr.sh, run_sasrec.sh, run_hstu.sh
  smoke_test.sh           smoke test over main_gensid.py: exercises every graph/encoder/dataset branch +
                          all 4 sid_models on CPU (SMOKE_DEVICE override)
  production/             TEMPORARY NPU/Ascend (MTP platform) scripts — not part of the dev workflow, keep out of scope

docs/TODO.md              technical roadmap (authoritative plan; supersedes any assumption below)
docs/restructure.md       live restructure status — where code physically lives, what remains
aux_configs/mmgrec.yaml   GR config override used by research pipelines

cache/ ckpt/ logs/ results/ tensorboard/ htmlcov/   runtime artifacts, all gitignored
```

## Core architecture

### SID pipeline (conceptual flow)

```
prepare data -> train SID method -> generate SID indices (JSON) -> map to consumer format -> train generative rec model
prepare_dataset.py   main_gensid.py      main_gensid.py           map_gensid_to_genrec.py            main_genrec.py
```

### `gensid` (the product)

- **`SIDPipeline`** (`pipeline.py`) is the orchestrator — data building, model instantiation, trainer construction,
  training and generation are distinct steps driven by `train()` / `generate()`. It replaces the old `SIDBuilder`.
- **Distributed (DDP) training** is auto-detected from the `torchrun` env (`RANK`/`WORLD_SIZE`/`LOCAL_RANK`/`MASTER_ADDR`/
  `MASTER_PORT`) via `gensid/utils.py::is_distributed`/`init_dist_state`/`wrap_model_for_ddp`. Backend: `nccl` (cuda),
  `hccl` (npu), `gloo` (cpu). The trainer keeps `self.model` as the raw module (attribute access/inference) and uses
  `self.ddp_model` only for the training forward; checkpoints save the raw state dict on rank 0 only; train loaders
  shard with `DistributedSampler`; generation runs on rank 0 (never DDP-wrapped) and `main_gensid.py` barriers at exit.
- **`SIDMethod`** (`method/base.py`) is the extension interface: a `nn.Module` subclass with class attrs `name`
  (registry key) and `trainer` (trainer-registry key) and an `__init__` typed with its `ModelConfig`
  (used to bind config↔method in the registry). Concrete today: `RQVAE`, `RKMeans`, `HierarchicalRKMeans`,
  `GRQVAE` (subclass of `RQVAE` prepending `nn/gnn.GNNFusionEncoder`).
- **Config is typed** (`gensid/config/`): `GraphBuildingConfig` (how the graph is built), `EncoderConfig`
  (refinement/dataset-stage knobs — **not** an `encoders/` package; APPNP/S2GR are plain `nn/gnn.py` layers, the
  learned GNN path ships inside the `GRQVAE` model), `DatasetConfig`, `ModelConfig`, `TrainerConfig` — all registry-
  discovered, each with `from_args(args)` to read the flat CLI namespace.
- **Registry** (`registry.py`): `REGISTRY.get_<group>(name)` / `REGISTRY.get_<group>_config(name)` / `REGISTRY.keys(group)`.
  Registration is **automatic** (package introspection over concrete subclasses); no decorators needed. Only the CLI
  `--sid_model`/`--graph_type` choices in `main_gensid.py` are still hand-synced (TODO #3 acceptance: derive from
  `REGISTRY.keys()`).
- **"Encoders" are just `nn` layers or offline dataset preprocessing** (APPNP = torch_geometric `APPNP` conv,
  S2GR = `MessagePassing`, GraphSAGE+BPR = pretrained item-repr refiner). The library ingests *pretrained*
  embeddings; there is no `Encoder` ABC or `encoders/` subsystem.
- Non-registry SID paths are still **bespoke entrypoints** (clustering-based: mincut/DMoN/spectral; the GNN
  adapter). They are not on the `SIDMethod`/`SIDPipeline` interface yet — folding them in is the remaining
  Phase 3 work (`docs/restructure.md` §5).

### `genrec` (temporary consumer reference)

- Config merge precedence: `config_dict` > `config_file` > model `config.yaml` > dataset `config.yaml` > `default.yaml` (`genrec/utils.py:get_config`).
- Registries via `get_model` / `get_tokenizer` / `get_trainer` / `get_pipeline`; models/datasets are per-directory plugins. The default `Pipeline`/`Trainer` are used when a model dir has no custom class.
- Evaluator is **collision-aware** (`genrec/evaluator.py:33` `calculate_pos_index_with_collisions`).

## How to add a new SID method (current + target)

**Today:**
1. Subclass `SIDMethod` (`gensid/method/base.py`) in a new module under `gensid/method/`, setting the `name` (and
   `trainer` if custom) class attributes and typing `__init__` with your `ModelConfig` subclass — the registry picks
   it up automatically via package discovery.
2. Define your `ModelConfig` (and `TrainerConfig` if needed) in `gensid/config/model_config.py` /
   `trainer_config.py`; if you need a custom training loop, subclass `gensid/trainer/base.py::Trainer` and put it in
   `gensid/trainer/`.
3. Add it to the `--sid_model` choices in `main_gensid.py` (still hand-synced today).
4. Metrics: `gensid/eval.check_id_distribution` (collision/utilization/perplexity/gini) + collision rate saved to the
   checkpoint JSON sidecar.
5. Cover it: unit tests in `tests/`; exercise the branch in `scripts/smoke_test.sh`.

**Planned (docs/TODO.md #3 + docs/restructure.md §5) — remaining target:**
- Derived CLI: `--sid_model` (and `--graph_type`) choices come from `Registry.keys()`, so a new method needs **no
  edit to `main_gensid.py`** (one new module = one new method).
- Fold the bespoke methods onto the interface: `hierarchical_mincut_clustering.py` → registered
  `{recdmon,spectral,recursive}` methods; `mmgrec_gnn_sid_adapter.py` → `SIDPipeline(sid_model="rqvae")` + a
  registered refiner (its toy `RQVAE` deleted).
- New methods expected eventually: balanced SID (#6), faiss PQ/OPQ/RQ (#7), literature methods (#8).

## Data & artifact contracts

- **gensid inputs** (produced by `prepare_dataset.py`), layout `cache/gensid/<dataset>/`:
  - `<dataset>.emb-<embedder>-td.npy` — `[num_items, dim]` float32, **row index == 0-indexed item_id**.
  - `<dataset>.inter.json` — `{user_id: [item_id, ...]}` (0-indexed, JSON keys as strings).
- **SID mapping JSON** (LETTER-style, gensid output): `{item_id: ["<a_3>", "<b_17>", ...]}`. Written to `cache/gensid/<dataset>/<dataset>.index.<ckpt_id>.json` (ckpt_id = `saved_model_dir` with `/` and `..` replaced).
- **GR `.sem_ids`** (output of `map_gensid_to_genrec.py`): `{raw_ASIN: [3, 17, ...]}`. The mapper also writes a shell **env handoff** file exporting `GR_SEM_IDS_PATH`, `GR_SEM_IDS_DEPTH`, `GR_SEM_IDS_CODEBOOK_SIZES`, `GR_DATASET` — sourced by the research scripts to launch `main_genrec.py`.
- **CF embeddings** (for LETTER-style CFLoss): `ckpt/gensid/<dataset>-32d-SASRec.pt`, produced by `map_cf_embeddings.py` from a pretrained SASRec checkpoint.
- Note: `map_gensid_to_genrec.py:9` hardcodes `DATASET_N_ITEMS` item counts — a known duplicated source of truth (roadmap #9).

## Workflows

### Setup

```shell
uv sync                  # or: conda/pip per README (requirements.txt)
pip install -e ".[dev]"  # dev extras: pytest, ruff
ruff check . && ruff format --check .   # line-length 120, double quotes
python -m pytest         # tests/ suite (runs with --cov per pyproject addopts)
```

### Reference end-to-end run (Amazon Beauty)

```shell
# 1. Prepare data: sentence embeddings + gensid inputs
python prepare_dataset.py --category=Beauty \
  --sent_emb_model=sentence-transformers/sentence-t5-base --sent_emb_batch_size=64

# 2. Train SID + generate indices (main_gensid.py)
python main_gensid.py --dataset=Beauty --embedder=sentence-t5-base \
  --saved_model_dir=<alias>/full_run_model --device=cuda:0 --seed=42 \
  --quant_loss_weight 0.25 --epochs 20000 --alpha 0.02 --beta 0.0 \
  --layers 512 256 128 64 --eval_step 100 --n_propagations 1 --teleport_prob 0.5 \
  --use_edge_weight --graph_signal --disable_vq_init --cf_loss_use_codebook_rep \
  --use_edge_reconstruction_loss

# 3. Map SID mapping -> GR .sem_ids + env handoff
python map_gensid_to_genrec.py \
  --sid_mapping_path cache/gensid/Beauty/Beauty.index.<alias>_full_run_model.json \
  --gr_path cache/AmazonReviews2014/Beauty/processed \
  --dataset Beauty --env_file logs/env/<alias>_sid_mapping.env

# 4. Finetune a generative rec model with the SIDs (main_genrec.py)
CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision bf16 main_genrec.py \
  --final_metrics_file=results/<alias>.json --model=TIGER \
  --dataset=AmazonReviews2014 --category=Beauty --max_item_seq_len=20 \
  --train_batch_size=512 --eval_batch_size=32 --rand_seed=42 \
  --sem_ids_path=<GR_SEM_IDS_PATH> --rq_n_codebooks=<GR_SEM_IDS_DEPTH> --rq_codebook_size=<GR_SEM_IDS_CODEBOOK_SIZES> \
  --config=aux_configs/mmgrec.yaml
```

The research scripts encode steps 1–4 as reusable functions with preset flag sets (`letter()`, `rqvae_args_full()`, `rkmeans()`, ...) and support partial runs via `--skip=gensid,copy,remap,finetune,pretrain_cf,copy_cf`. `scripts/smoke_test.sh` is the fast in-repo way to verify every `main_gensid.py` branch on CPU.

## Conventions & tooling

- **Python 3.11** (`>=3.11,<3.12`), **torch 2.11 + CUDA 13**, `torch_geometric`, `sentence-transformers`, `accelerate`; deps under **uv** (`uv.lock`).
- **Ruff** is the formatter/linter (config in `pyproject.toml`: line-length 120, double quotes, LF; isort is not explicitly configured — keep imports grouped with `gensid`/`genrec`/`scripts` first-party).
- **Tests**: `pytest` with pytest-cov (configured in `pyproject.toml`); suite lives in `tests/`. Keep tests CPU-only and deterministic.
  - **Test policy (project rule — preserve for future interactions):** tests must target **actual, fundamental functional logic** — the behavior that defines what the library does (index generation, collision handling, training/checkpointing, config↔registry dispatch, graph-building correctness, dataset semantics, loss decomposition). **Do NOT** write tests for trivial or obvious things just to pad coverage; the suite must stay lean, not dense and heavy. Concretely, avoid: cosmetic/formatting helpers (e.g. `set_color`, `get_local_time`), pure `NotImplementedError` checks on abstract methods, and smoke tests that assert only shapes/`isfinite` without verifying the real behavior. Assert **behavior/values**, not just shapes.
- **Devices**: always go through `gensid.utils.get_device` (accepts `cuda`/`npu`/`cpu`/`auto`; NPU = Huawei Ascend). Do not hardcode `cuda`.
- **Reproducibility**: seed via `init_seed` / `rand_seed`; results go to `results/`, logs to `logs/`, tensorboard to `tensorboard/` (all gitignored).
- Checkpoints are `torch.save` with `pickle_protocol=4`; each saves the typed config dataclass under the `"args"` key plus a metrics JSON sidecar (`<ckpt>.json`) tracking epoch/best_loss/best_collision_rate/ckpt_path.

## Known state & gotchas

- **Tests exist** (`tests/`, 14 files, green, ≈60% coverage) — `pytest` is configured with `--cov` in `pyproject.toml`; the thinnest-covered files are `pipeline.py` and the trainers (validated end-to-end via `scripts/smoke_test.sh`).
- `scripts/production/*` are temporary NPU/MTP-platform leftovers — **out of scope**, do not refactor or run them as part of dev workflows.
- The bespoke monofiles (`hierarchical_mincut_clustering.py`, `mmgrec_gnn_sid_adapter.py`, and the thin `generate_graph_cluster_indices.py` wrapper) are **not yet on the `SIDMethod`/`SIDPipeline` interface** — don't treat their entry APIs as stable. `mmgrec_gnn_sid_adapter.py:157` still has its own toy `RQVAE` and its own `DEFAULT_PREFIXES`/`load_interactions` to fold (`docs/restructure.md` §4 U5/U6).
- **Ignore, do not develop or delete** legacy/untracked bespoke files: `generate_graph_cluster_indices.py` is **deprecated** (the clustering methods `recdmon`/`spectral`/`stacked_dmon` already produce the SID mapping in a single pipeline via `main_gensid.py`) but is **kept as a reference** to verify functionality is preserved and consistent — do NOT treat it as a target to develop or delete. General rule: during development, leave untracked legacy files untouched; they may serve as reference for behavior-preservation checks.
- `--sid_model` and `--graph_type` **choices in `main_gensid.py` are still hand-synced** with the registry — keep them in sync when adding a method (until the CLI derives from `Registry.keys()`).
- Two ad-hoc name→fn tables remain (interim scaffolding, do NOT grow them): `data/graph.py::get_graph_builder` and `nn/__init__.py::get_encoder` (see `docs/restructure.md` §1.7).
- `MixedFusionDepthDataset` samples its hop per `__getitem__` (non-deterministic even with a seeded `rng`); flagged for an indexed/seedable rewrite (`docs/restructure.md` §8).
- `GRQVAE` is wired (pipeline + smoke test) but still research-grade: the `GNNFusionEncoder` is not pretrained and its training is noted unstable in the code TODOs.
- The `S2GRConv` `config.K` bug (`docs/restructure.md` §6.1) is **fixed** — `nn/gnn.py::S2GRConv.__init__` reads `config.n_propagations`.
- `method/utils.py` is an empty placeholder file; `method/base.py::SIDMethod` is the real base.
- `docs/` (both `TODO.md` and `restructure.md`) is tracked/committed; `cache/`, `ckpt/`, `logs/`, `results/`, `tensorboard/`, `htmlcov/`, `__pycache__/` are gitignored. `docs/TODO.md` is the roadmap source of truth; `docs/restructure.md` is the live status doc.

The authoritative plan for the restructuring (config dataclasses → dataset resolver → registry → tests → coverage → fold monofiles → new SID methods → `gr_unified` coupling) is **`docs/TODO.md`**, with live structure status in **`docs/restructure.md`**; follow their ordering when implementing refactors.
