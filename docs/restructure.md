# GrIS / gensid — Restructuring Plan

> Living design document for the `gensid/` restructure. Companion to `docs/TODO.md` (the technical
> roadmap); this file focuses on *where code physically goes* and *in what order it moves*.
> It first re-defines the current structure (what exists today and why it is a problem), then defines
> the target structure (folders + responsibilities), then the per-phase move/unify steps, and finally
> how config / registry / dataset-resolver cross-cutting concerns slot in.
>
> Status: **Phases 1–4 have largely landed.** The layout below in §2 is the *current committed state*, not
> a future proposal: typed config dataclasses (`config/`), the `Registry`, the `SIDPipeline`, the
> `SIDMethod` bundle, per-method trainers and a `tests/` suite all now exist. What remains is the "finish
> line" work captured in §4/§5/§8: fold the bespoke monofiles into registered methods, derive the CLI
> choices from the registry, persist config as JSON sidecars, and close the coverage gaps.

---

## 1. Design principles

1. **The library is a library of SID methods.** A *SID method* is a unit that turns
   `(item embeddings [, interactions])` into a SID mapping `{item_id: ["<a_i>", ...]}`.
2. **"Encoder" is NOT a first-class subsystem.** It is too hard to draw a boundary between an "encoder"
   and a normal `nn` layer: APPNP refinement *is* torch_geometric's `APPNP` conv, S2GR is a plain
   `MessagePassing` layer, and GNN items are *pretrained* encoders whose outputs are just embeddings.
   The library's goal is to ingest **pretrained** embeddings, so a whole "encode" logic/folder is
   overkill. Refinements are therefore either
   - ordinary `nn` layers (`nn/gnn.py`), or
   - **offline data preprocessing** computed inside the dataset / graph step (`data/graph.py`,
     graph-aware datasets).

   The only remnant of the encoder idea is **config**: `EncoderConfig` (and subclasses) lives in
   `config/encoder_config.py` *as data* — the hyperparameters of the refinement/`dataset` stage — but it
   does **not** imply an `encoder/` component package or an `Encoder` ABC (that placeholder was deleted
   in Phase 4).
3. **A method is a *pipeline of separate calls*, not a god object.** Data building, model
   instantiation, trainer construction, training, generation and artifact saving are distinct steps
   orchestrated by one `SIDPipeline` object (`pipeline.py`, mirroring `genrec/pipeline.py`). Method
   folders hold only *components* + registration — no orchestration code inside them.
4. **Shared, reusable logic lives OUTSIDE the method folders.** Anything two or more methods would
   duplicate is hoisted into shared modules (`generation/`, `trainer/`, `nn/`, `data/`, ...).
5. **Few files, decoupled internals.** One file may host several classes that do not depend on each
   other (all concrete datasets in one `dataset.py`, pure metrics in one `eval.py`).
6. **Mirror `genrec`** (per TODO #9/#10): registry + dataclass/YAML config + thin CLI, so a new
   method = one new module plus (eventually) one registration call.
7. **The dynamic dispatch helpers were interim scaffolding and are mostly gone.** `get_graph_building_conf`
   and `get_encoder_conf` (`config.py` of old) were *deleted* — call sites now resolve typed configs through
   the `Registry` (`REGISTRY.get_graph_building_config(...)` / `REGISTRY.get_encoder_config(...)`). Two small
   ad-hoc name→fn dispatch tables survive and are the remaining scaffolding to fold into registry-driven
   flow (see §4 U10): `data/graph.py::get_graph_builder` (graph_type → builder fn) and `nn/__init__.py::get_encoder`
   (conv name → `APPNPConv`/`S2GRConv` class). Do not grow them.

---

## 2. Current structure (committed state) and why it still needs to change

### 2.1 Layout today

```
main_gensid.py          CLI for gensid — parse args -> SIDPipeline(args).train() / .generate()
                        (--sid_model choices are still hand-synced: rqvae/rkmeans/hierarchical_rkmeans/grqvae)
main_genrec.py          CLI for genrec (consumer, OUT OF SCOPE for this restructure)
prepare_dataset.py      GR dataset -> gensid inputs (.npy embeddings + .inter.json)
map_gensid_to_genrec.py SID JSON -> GR .sem_ids + env handoff (hardcodes DATASET_N_ITEMS)
map_cf_embeddings.py    extract SASRec CF embeddings -> gensid ckpt

gensid/
  registry.py                       Registry + REGISTRY hub. Self-discovery by package introspection:
                                    each group scans a package for concrete subclasses of a base class,
                                    keyed by a class attribute (name / graph_type). Groups: sid_method,
                                    trainer, encoder_config, graph_building_config, dataset_config,
                                    trainer_config, dataset. get_<group>[_config]() dynamic accessors.
  pipeline.py                       SIDPipeline — the ONE orchestrator (train() / generate()). Builds
                                    dataset+loader (data/base), model+config (registry), trainer (registry),
                                    then delegates to trainer.fit() / trainer.generate_indices().

  config/                           typed dataclass *package* (was a single config.py, split in Phase 4)
    __init__.py                     re-exports the 5 base configs + _CONFIGS tuple (for docstring tooling)
    graph_build_config.py           GraphBuildingConfig (graph_type discriminant + .to_kwargs()/.from_args)
                                    + AdjacentCooc / AdjacentCoocProd / RandomWalk(rw) / Windowed subclasses
    encoder_config.py               EncoderConfig + GraphEncoderConfig base + APPNP / S2GR / GNNFusion /
                                    GraphSage subclasses (name discriminant: "APPNP", "S2GR", ...)
    dataset_config.py               DatasetConfig (name) + GraphDatasetConfig base (graph_building,
                                    spectral_dim, cluster_labels_path, ...) + Plain /
                                    GraphFusion / MixedDepth / GRQVAEFusion subclasses
    model_config.py                 ModelConfig (name) + RQVAE / GRQVAE / RKMeans / HierarchicalRKMeans
                                    (GRQVAE extends RQVAE with gnn_* knobs)
    trainer_config.py               TrainerConfig (name, lr, epochs, eval_step, ...) + RQ / RKMeans /
                                    HierarchicalRKMeans subclasses
    docstring.py                    auto_docstring decorator: splices a config's Args: block onto the
                                    consuming model class docstring (HF-style)
  eval.py                           check_id_distribution + utilization/perplexity/collision/gini (pure)
  utils.py                          collision helpers (check_collision / get_collision_item), get_device
                                    (cuda/npu/cpu/auto), init_seed, get_trained_flag, ensure_dir, set_color

  data/
    base.py                         dataset resolution (TODO #2): get_dataset_config(args) -> DatasetConfig,
                                    build_dataset(config, paths, eval_mode), build_data_loader(...). Datasets
                                    yield keyed dict batches ({"emb", "index", "target", ...}) — no positional
                                    tuple cascade. GRQVAE trains on a NeighborLoader; eval always plain loader.
    dataset.py                      EmbDataset, GraphAwareEmbDataset (shared base), GNNFusionEmbDataset
                                    (offline --graph_signal), GRQVAEFusionEmbDataset (online GNN, saves
                                    post-train fused cache), MixedFusionDepthDataset (--graph_augmentation)
    graph.py                        graph builder fns (adjacent_cooc / prod / rw / windowed, config-based)
                                    + get_graph_builder(name -> fn) [interim] + polars_graph_to_pyg
                                    + compute_graph_spectral_vectors + InteractionGraphBuilder (from
                                    the old graph_aware_sid/ package — fully folded in here)
    utils.py                        EdgeWeightCollator + load_train_item_mask + load_graph_cluster_labels

  generation/
    base.py                         encode driver (was RQVAEBuilder's generate half): collect_indices /
                                    report_and_save / indices_to_token_lists / save_generation_mapping
    utils.py                        format_prefix (the ONE prefix formatter) + cluster-label -> codes helpers
                                    (load_cluster_label_levels / make_contiguous_per_level / build_codes /
                                    collision_stats, moved from generate_graph_cluster_indices.py)

  method/                           ONLY SID method models (flat, not per-method dirs yet)
    base.py                         SIDMethod (nn.Module): n_codebooks / codebook_sizes / generate() /
                                    compute_loss(); class attrs name + trainer (registry keys)
    rqvae.py  rkmeans.py  grqvae.py named SIDMethod subclasses; GRQVAE subclasses RQVAE and prepends
                                    nn/gnn.GNNFusionEncoder; rkmeans.py also holds HierarchicalRKMeans
    pq.py                           PQMethod / OPQMethod (product quantizers; torch/faiss backend knob)
    __init__.py                     re-exports; utils.py empty

  nn/                               SHARED non-method building blocks
    gnn.py                          GNNFusionEncoder, GatedGraphNet, APPNPConv, S2GRConv (config-based nn
                                    layers; S2GRConv reads config.n_propagations — the old config.K bug is fixed)
    layers.py                       ResidualVectorQuantizer, VectorQuantizer, TorchKMeans, MLPLayers
    pq.py                           TorchPQ / FaissPQ / TorchOPQ / FaissOPQ (+ pack/unpack, make_pq/make_opq)
    utils.py                        DISTANCE_FUNCTIONS / MEAN_FUNCTIONS + constrained_km (k-means-constrained)
                                    + sinkhorn_algorithm + activation_layer
    __init__.py                     get_encoder(name -> conv class: none/appnp/s2gr)  [interim, kept]

  trainer/
    base.py                         Trainer (typed TrainerConfig; epoch loop, VQ-less generic contract,
                                    collision-based ckpt selection, metrics JSON sidecar)
    rq_trainer.py                   RQTrainer (VQ init pass, single-pass generation by default;
                                    optional Sinkhorn collision-fix branch behind config.fix_collisions;
                                    post-train fused-cache save for GRQVAE)
    rkmeans_trainer.py              RKMeansTrainer / HierarchicalRKMeansTrainer (fit + generate_indices)
    __init__.py                     re-exports

  hierarchical_mincut_clustering.py  stacked_dmon / recursive_spectral / recursive_dmon (bespoke CLI)
  generate_graph_cluster_indices.py  cluster labels -> SID mapping JSON (bespoke CLI; now imports its
                                     pure helpers from generation/utils.py)
  mmgrec_gnn_sid_adapter.py          GNN (SAGE + BPR) -> toy RQ-VAE -> SIDs (bespoke CLI; toy RQVAE at :157
                                     and its own DEFAULT_PREFIXES / load_interactions still here)

tests/                               14 pytest files covering config/datasets/graph/nn/methods/trainers/utils
scripts/smoke_test.sh                end-to-end smoke test over main_gensid.py: doses every dataset/graph/
                                     encoder branch + all 4 sid_models on CPU
```

### 2.2 The monofiles, explained

| File | What it really is |
|---|---|
| `generate_graph_cluster_indices.py` | The **encode/generation phase** of the graph-cluster SID method. Its pure logic (`load_cluster_label_levels`, `make_contiguous_per_level`, `build_codes`, `collision_stats`) has already moved to `generation/utils.py` and is imported from there; only the thin CLI wrapper remains here. |
| `hierarchical_mincut_clustering.py` | The **fit phase** of that same method, holding **3 distinct SID methods**: `stacked_dmon` (recdmon), `recursive_spectral` (spectral), `recursive_dmon` (recursive). Still mixes neural modules (`SparseEncoder`, `HierarchicalDMoN`, `LocalAssignNet`), a loss, train loops (`train_stacked_dmon` + its own `TrainConfig`), graph preprocessing (via `InteractionGraphBuilder`), and a CLI. |
| `mmgrec_gnn_sid_adapter.py` | A **two-stage pipeline**: GraphSAGE + BPR encoder over `(interactions x embeddings)` -> item reprs -> **its own toy `RQVAE`** (`:157`) -> popularity collision token -> SID JSON. It still carries its own `DEFAULT_PREFIXES` (`:34`) and `load_interactions` (`:53`). The toy `RQVAE` re-implements residual VQ with a different signature; the whole thing is really *a pretrained-GNN refinement + method=rqvae + collision-token*, not a fundamentally new method. |

### 2.3 What is left to unify (the remaining justification)

| Item | Status |
|---|---|
| `constrained_km` | **single copy** in `nn/utils.py:31`; consumed by `RQTrainer.generate_indices`. ✅ |
| `sinkhorn_algorithm` | **single copy** in `nn/utils.py:158`; consumed inside `ResidualVectorQuantizer` (nn/layers.py). ✅ |
| Sinkhorn collision-fix loop | moved out of the old `RQVAEBuilder` into `RQTrainer.generate_indices` (rq_trainer.py) + `generation/base.collect_indices`; now an **opt-in branch** (`config.fix_collisions`/`--fix_collisions`), off by default to match GrIS_v1's single-pass generation. ✅ |
| prefix formatter | **single `format_prefix`** in `generation/utils.py`; prefix/unclustered/leaf formats are now argparse-controlled (`--prefixes` / `--unclustered_strategy` / `--unique_prefix`, threaded via `TrainerConfig`) and consumed by `generation/base.collect_indices` (RQ path) and `build_codes` (clustering path). One copy left: `mmgrec_gnn_sid_adapter.py:34 DEFAULT_PREFIXES`. |
| interactions / train-mask loading | `data/utils.py::load_train_item_mask` + `data/graph.py::InteractionGraphBuilder._load_interactions` + `mmgrec:53` bespoke loader — partial, 3 places. |
| RQ-VAE quantizer | `method/rqvae.py` (full) **and** `mmgrec_gnn_sid_adapter.py:157` (toy). Still duplicated (U5). |
| graph builder fns | all consolidated in `data/graph.py` (builders + `get_graph_builder` + `polars_graph_to_pyg` + `InteractionGraphBuilder`); `graph_aware_sid/` is **deleted**. ✅ |
| training loops | `trainer/base.py` (Trainer), `trainer/rq_trainer.py` (RQTrainer), `trainer/rkmeans_trainer.py` (k-means trainers) — the consumers are covered. Remaining variants: `train_*` inside the two monofiles. |
| dataset->trainer contract | datasets return **keyed dicts** (`{"emb","index","target",...}`); `EdgeWeightCollator` augments with `edge_weights`; trainers read by key. The positional tuple cascade (U8) is **gone**. ✅ |
| orchestration (train + generate) | `SIDPipeline` (`pipeline.py`) drives all registered methods; only the bespoke monofile CLIs still orchestrate themselves. |

### 2.4 Dead / orphaned code (fate decided in a later cleanup pass)

- `method/utils.py` — empty placeholder file.
- `mmgrec_gnn_sid_adapter.py::RQVAE` — the toy VQ re-implementation (`:157`), still to be deleted (U5).
- `GRQVAE` — no longer orphaned: it is registered and driven by the pipeline (`--sid_model grqvae`, exercised by
  `scripts/smoke_test.sh`), but it still carries research TODOs (the graph encoder is not pretrained yet).
- The bespoke monofiles (`hierarchical_mincut_clustering.py`, `mmgrec_gnn_sid_adapter.py`) are not on the
  `SIDMethod`/`SIDPipeline` interface yet — unifying them is the remaining Phase 3 work.

---

## 3. Target structure

### 3.1 Current vs target — the delta

Most of the target tree from earlier revisions has already landed:

- **`config/` package** (not a single `config.py`) — five config hierarchies, all registered: `graph_build_config.py`,
  `encoder_config.py`, `dataset_config.py`, `model_config.py`, `trainer_config.py`.
- **`registry.py`** — `Registry` + `REGISTRY` hub with package self-discovery; `get_<group>[_config]()`, `keys(group)`.
- **`pipeline.py`** — `SIDPipeline` implemented (train + generate).
- **`data/base.py`** — dataset resolution (`get_dataset_config` / `build_dataset` / `build_data_loader`);
  keyed-dict batches replace the tuple cascade.
- **`method/`** flat (not yet `methods/<name>/`); **`trainer/`** flat with per-method trainers.
- **`nn/pq.py` + `method/pq.py`** — product-quantizer SID methods added: `pq`/`opq` with a
  `--quantizer_backend torch|faiss` knob (`TorchPQ`/`FaissPQ`/`TorchOPQ`/`FaissOPQ`); torch backends are
  device-agnostic, faiss runs on CPU and is parity-tested bit-for-bit against the torch encode (see
  `tests/test_nn_pq.py`, `scripts/pq_opq_scaling.py`, TODO.md #7).
- **`generation/`** has both `base.py` (encode driver) and `utils.py` (prefix formatter + cluster helpers).
- **`encoder/`, `graph_aware_sid/`, `builders.py`, `old_main_gensid.py`** — all deleted.

Still open (the remaining target):
- Fold `hierarchical_mincut_clustering.py` into registered methods (`recdmon`/`spectral`/`recursive`) and
  `mmgrec_gnn_sid_adapter.py` into `SIDPipeline(sid_model="rqvae")` + a registered refiner (Phase 3).
- Thin `cli.py` with `--sid_model` choices derived from `Registry.keys()` (today: hand-synced in `main_gensid.py`).
- `config.to_json()` checkpoint sidecars (today the trainer stores the config dataclass under `"args"` + a metrics JSON).

### 3.2 The `SIDPipeline` (already implemented)

`builders.py` is gone. In its place: **a generic pipeline** (orchestration), a **component bundle per method**
(strategy), and **shared helpers**. The actual `pipeline.py`:

```python
class SIDPipeline:
    def __init__(self, args, device): ...

    def _build_model(self, dataset_dim):
        config = REGISTRY.get_sid_method_config(self.args.sid_model).from_args(self.args)
        # overrides in_dim from the dataset, materializes --cf_emb into a numpy array
        return REGISTRY.get_sid_method(self.args.sid_model)(config)

    def _build_trainer(self, model):
        name = model.trainer                              # trainer key lives on the method
        config = REGISTRY.get_trainer_config(name).from_args(self.args)
        return REGISTRY.get_trainer(name)(config, model, device=..., ckpt_dir=..., saved_model_dir=...)

    def _build_dataset_and_loader(self, eval_mode):
        config = get_dataset_config(self.args)            # data/base.py
        dataset = build_dataset(config, data_path, interactions_path, eval_mode=eval_mode)
        loader = build_data_loader(dataset, config, self.args, eval_mode=eval_mode)
        return dataset, loader

    def train(self):    # dataset+loader (train) -> model -> trainer.fit(dataset, loader)
    def generate(self, ckpt_path, output_file):  # eval dataset -> reload ckpt -> trainer.generate_indices(...)
```

- **The "encoder" is *not* a separate pipeline step.** Offline refinement happens inside `GNNFusionEmbDataset`
  (`data/dataset.py`) via `nn/__init__.py::get_encoder(...)` (APPNP/S2GR); the learned path is the `GRQVAE` model's
  own `GNNFusionEncoder` (mmgrec flow is still a bespoke CLI).
- `main_gensid.py` is thin (parse flat args → `SIDPipeline(args)`); it still hand-lists `--sid_model` choices.

### 3.3 Component mapping (current -> target)

| Current | Target | Status |
|---|---|---|
| `SIDBuilder` ABC + `RQVAEBuilder`/`RKMeansBuilder` (`builders.py`) | `SIDMethod` bundle (`method/base.py`) + `SIDPipeline` (`pipeline.py`) | **done** (builders.py deleted) |
| `SIDBuilder.format_prefix` / `evaluate_and_save_generated_indices` | `generation/base.py` + `generation/utils.py` | **done** |
| `setup_dataset_and_loader` (`builders.py`) | `data/base.py` `get_dataset_config`/`build_dataset`/`build_data_loader` | **done** |
| `get_graph_building_conf`/`get_encoder_conf` helpers | `config/` dataclasses + `REGISTRY.get_*_config()` | **done** (helpers deleted; see §1.7) |
| graph builder fns + `InteractionGraphBuilder` (graph_utils.py) | `data/graph.py` | **done** (`graph_aware_sid/` deleted) |
| `RQVAEBuilder.setup_model` / `RQVAEBuilder.train` | `method/rqvae.py` + `trainer/rq_trainer.py` (RQTrainer) | **done** |
| `_run_vae_generation` + `constrained_km` (`builders.py`) | `RQTrainer.generate_indices` + `nn/utils.constrained_km` + `generation/base.collect_indices` | **done** |
| `RKMeansBuilder.*` | `method/rkmeans.py` + `trainer/rkmeans_trainer.py` | **done** |
| `get_sid_builder` (`builders.py`) | `REGISTRY.get_sid_method(name)` | **done** |
| `trainer.py Trainer` / `trainer/vq.py VQTrainer` | `trainer/base.py` Trainer + `trainer/rq_trainer.py` RQTrainer + `trainer/rkmeans_trainer.py` | **done** (VQTrainer superseded by RQTrainer) |
| `datasets_module.py` | `data/dataset.py` + `data/base.py` + `data/utils.py` | **done** |
| `models/` package | `nn/` (shared) + `method/` (method models) | **done** |
| `GRQVAE` | `method/grqvae.py` (RQVAE subclass + `nn/gnn.GNNFusionEncoder`) | **done & wired** (pipeline + smoke test) |
| `APPNPConv`/`S2GRConv` (graph_utils.py) | `nn/gnn.py` (config-based) | **done** |
| `graph_aware_sid/graph_utils.py` | `data/graph.py` (+ `data/dataset.py`) | **done** |
| `generate_graph_cluster_indices.py` pure fns | `generation/utils.py` | **done** (bespoke CLI wrapper remains) |
| `hierarchical_mincut_clustering.py` | methods `{recdmon,spectral,recursive}` + shared trainer/generation | pending (Phase 3) |
| `mmgrec_gnn_sid_adapter.py` | `nn/gnn.py` (SAGE+BPR) + method `rqvae` composition; toy `RQVAE` -> **deleted** | pending (Phase 3, U5) |
| `encoder/` package | **deleted** (folded into `nn/` + config) | **done** |
| `old_main_gensid.py` / `builders.py` | **deleted** | **done** |
| `main_gensid.py` | thin `cli.py` + `main_gensid.py` shim; choices from `Registry.keys()` | partial (CLI thin; choices still hardcoded) |

---

## 4. What to unify (consolidation list, by when it can land)

| # | Unify THIS | From these places | Lands in | Status |
|---|---|---|---|---|
| U1 | `constrained_km` + Sinkhorn collision resolution | old `builders._run_vae_generation` | `nn/utils.py` (km/sinkhorn) + `generation/base.collect_indices` + `trainer/rq_trainer.py` (loop) | **done** |
| U2 | interaction / train-mask loading | `InteractionGraphBuilder._load_interactions`, `mmgrec:53`, (old `load_train_item_mask`) | `data/utils.py` (`load_train_item_mask`) + `data/graph.py` | partial (mask + graph unified; `mmgrec:53` remains) |
| U3 | `get_graph_building_conf` | old `builders.py` + `graph_utils.py` | `config/graph_build_config.py` + `REGISTRY.get_graph_building_config` | **done** |
| U4 | graph-type dispatch | `get_graph_builder` (data/graph.py) + `REGISTRY` + CLI `--graph_type` | registry-driven builder lookup; CLI choices from `keys()` | partial (`data/graph.get_graph_builder` dict remains; CLI hardcoded) |
| U5 | RQ-VAE quantizer | `method/rqvae.py` + `mmgrec` toy `RQVAE` | `method/rqvae.py` (+ plain-VQ mode); toy deleted | pending (`mmgrec:157`) |
| U6 | prefix templates | `generation/utils.py` (home), `generate_graph_cluster_indices` (imports it), `mmgrec:34` | `generation/utils.format_prefix` (single formatter) + `--prefixes`/`--unique_prefix`/`--unclustered_strategy` CLI (via `TrainerConfig`) | mostly done (all in-package trainers use config; `mmgrec:34 DEFAULT_PREFIXES` copy remains) |
| U7 | training loops | `trainer/{base,rq,rkmeans}_trainer.py` + monofile `train_*` | `trainer/` + (later) registered methods | partial (in-package loops done; monofiles remain) |
| U8 | dataset->trainer contract | datasets `__getitem__` + trainers + `EdgeWeightCollator` | keyed dict batches (`{"emb","index","target",...}`) | **done** (no positional tuple cascade) |
| U9 | orchestration glue | old `RQVAEBuilder` vs `RKMeansBuilder` | `pipeline.py` `SIDPipeline` | **done** |
| U10 | config dispatch helpers | old `config.py`/`data/utils.py`/`encoder/utils.py` helpers | `REGISTRY` + `data/graph.get_graph_builder` + `nn/__init__.get_encoder` | **done** (helpers deleted; the two fns remain — see §1.7) |

---

## 5. Execution order (phases)

> Phases 1–4 of the original plan are largely done; §4/§8 list what remains.

### Phase 1 — Structural scaffold  ✅ done
- [x] New package skeleton: `data/`, `nn/`, `method/`, `trainer/`, `generation/`, `config/`.
- [x] **U3**: `GraphBuildingConfig` (+ per-graph-type sub-configs) and the `EncoderConfig` hierarchy created.
- [x] Datasets moved into `data/` (`dataset.py`, `utils.py`, `graph.py`); `eval.py`/`utils.py` unchanged.
- [x] `models/` split: shared blocks → `nn/`, method models → `method/`; `GRQVAE` re-parented to subclass `RQVAE`.
- [x] **U9**: `SIDPipeline` implemented and `rqvae`/`rkmeans`/`grqvae` ported onto it; `builders.py` deleted.
- [x] **Registry** implemented (`registry.py`).

### Phase 2 — Extract shared logic (unify + keep folding)  ✅ mostly done
1. [x] **U1**: Sinkhorn 2-pass collision loop moved into `RQTrainer.generate_indices` next to `constrained_km`/`sinkhorn_algorithm`/`collect_indices`.
2. [~] **U2**: `data/utils.py::load_train_item_mask` + `InteractionGraphBuilder._load_interactions` (in `data/graph.py`) — `mmgrec:53` copy remains.
3. [x] **U7**: base `Trainer` reads typed `TrainerConfig` (the `args.batch_size`/`train_batch_size` bug is fixed); `RQTrainer`/`RKMeansTrainer`/`HierarchicalRKMeansTrainer` are the canonical loops.
4. [x] **U6/U8**: `generation/base.py` encode driver; datasets emit keyed dicts (the tuple cascade is gone).
5. [x] Fold `graph_aware_sid/graph_utils.py` into `data/graph.py` (`InteractionGraphBuilder`, `polars_graph_to_pyg`, cluster-label/mask helpers); `graph_aware_sid/` **deleted**.

### Phase 3 — Fold the monofiles into methods  ⌛ remaining
1. **`hierarchical_mincut_clustering.py` -> registered methods `{recdmon,spectral,recursive}`** (easiest, most logic already pure): shared preprocessing (`data/graph.py`) + shared encode (`generation/utils.py`/`base.py`). Note: `generate_graph_cluster_indices.py` is **deprecated and kept as reference only** (the clustering methods already produce the SID mapping in a single pipeline via `main_gensid.py`) — do not delete it, and do not treat it as a folding target.
2. **`mmgrec_gnn_sid_adapter.py` -> `nn/gnn.py` + pipeline composition**: GraphSAGE+BPR stay in `nn/gnn.py`; **U5**: the toy `RQVAE` is deleted, flow becomes `SIDPipeline(sid_model="rqvae")` + optional registered refiner; BPR dataset → `data/dataset.py`.
3. **`GRQVAE`** stays a subclass of `RQVAE` (already done); pretrain/validate the `GNNFusionEncoder` branch before relying on it in research.

### Phase 4 — Cross-cutting infra on top of the settled structure  ✅ mostly done
1. **#1 config**: `graph_build_config`/`encoder_config`/`dataset_config`/`model_config`/`trainer_config` all landed in `config/`. Checkpoints persist the config dataclass (under `"args"`) + a metrics JSON sidecar; `config.to_json()` sidecars are the remaining nicety. `Trainer` reads typed config only. |
2. **#2 resolver** (`data/base.py`): `get_dataset_config(args)`/`build_dataset`/`build_data_loader`; keyed-dict `Sample`; `setup_dataset_and_loader` **deleted**.
3. **#3 registry**: `sid_method`/`graph_type`(config)/`encoder_config`/`dataset`/`dataset_config`/`trainer`/`trainer_config` groups implemented (`registry.py`). **U4/U10**: the interim helpers are replaced by registry lookups; `data/graph.get_graph_builder` and `nn/__init__.get_encoder` remain as the last ad-hoc tables. CLI `--sid_model` choices still hand-synced.
4. [x] Delete `old_main_gensid.py`, `builders.py`, and the `encoder/` placeholder — **all done**.

### Phase 5 (later) — Future methods & coupling
New SID methods (`faiss`, `balanced`, `capsid`) = new thin modules reusing `trainer/` + `generation/*`. #10: formal SID artifact contract (`io/artifacts.py`), kill `map_gensid_to_genrec.py` `DATASET_N_ITEMS`, align with `gr_unified`.

---

## 6. Cross-cutting concerns (how config / registry / resolver behave)

### 6.1 Config dataclasses (`config/`) — current, real design

The single `config.py` became a `config/` package with five typed hierarchies, all registry-discovered:

- **`graph_build_config.py`** — `GraphBuildingConfig` (graph_type discriminant; `.to_kwargs()` strips `graph_type`)
  → `AdjacentCoocGraphConfig`, `AdjacentCoocProdGraphConfig` (parquet map-reduce), `RandomWalkGraphConfig` (`rw`),
  `WindowedGraphConfig`.
- **`encoder_config.py`** — `EncoderConfig` → `GraphEncoderConfig` base → `APPNPEncoderConfig`, `S2GREncoderConfig`;
  standalone `GNNFusionEncoderConfig`, `GraphSageEncoderConfig`. Registered by `name` (`"APPNP"`, `"S2GR"`, ...).
- **`dataset_config.py`** — `DatasetConfig` → `GraphDatasetConfig` base (graph_building,
  spectral_dim, cluster_labels_path, ...) → `PlainDatasetConfig` (`plain`), `GraphFusionDatasetConfig` (`graph_fusion`,
  offline `--graph_signal`; carries an `encoder`), `MixedDepthDatasetConfig` (`mixed_depth`, `--graph_augmentation`;
  hops/probs/cache_root), `GRQVAEFusionDatasetConfig` (`grqvae_fusion`, online GNN path).
- **`model_config.py`** — `ModelConfig` → `RQVAEModelConfig`, `GRQVAEModelConfig` (+gnn_*), `RKMeansModelConfig`,
  `HierarchicalRKMeansModelConfig`.
- **`trainer_config.py`** — `TrainerConfig` (lr/learner/epochs/eval_step/compile) → `RQTrainerConfig` (sk/vq-init knobs),
  `RKMeansTrainerConfig`/`HierarchicalRKMeansTrainerConfig` (train_batch_size).
- `docstring.py::auto_docstring` splices a config's `Args:` block onto the consuming model class.

**Intermediate layer**: every config has `from_args(args)`, which reads the flat `argparse.Namespace` into the typed
fields (argparse is still the source of flags `main_gensid.py`).

**Remaining (Phase 4/5)**: `config.to_json()` sidecars and CLI/YAML merge mirroring `genrec.utils.get_config` precedence.

Documented latent issues — status:
- `nn/gnn.py::S2GRConv.__init__` now reads `config.n_propagations` (**fixed**; no more `config.K` AttributeError).
- The old `get_encoder_conf`-keys-on-`convolution_type` inconsistency is gone: `data/base.py::get_dataset_config` resolves
  the encoder config via `REGISTRY.get_encoder_config(args.convolution_type)` and passes it into **both** the
  `graph_fusion` and `mixed_depth` branches.

### 6.2 Registry (`registry.py`) — implemented, the successor to the dispatch helpers
- `Registry` class (`add`/`get`/`get_config`/`keys`) + a `_Hub` (`REGISTRY`) that builds groups lazily via `discover()`:
  each group runs `pkgutil.iter_subclasses(pkg, base, name_attr)` over its package, so **registration is automatic** —
  a concrete subclass with the right class attribute and module location is found on import, no decorator needed.
- Groups (`GROUPS`): `sid_method` (`gensid.method`/`SIDMethod`/`name`), `trainer` (`gensid.trainer`/`Trainer`/`name`),
  `encoder_config`/`dataset_config`/`trainer_config` (`gensid.config`/respective base/`name`),
  `graph_building_config` (`gensid.config`/`GraphBuildingConfig`/`graph_type`), `dataset` (`gensid.data`/`TorchDataset`/`name`).
- **Config binding**: `config_from_signature(cls)` introspects `cls.__init__` type hints for a `_CONFIGS`-typed parameter
  and pairs it with the class, so `get_<group>_config(name)` returns the right config class.
- The stock `Trainer`/`TrainerConfig` are added under the conventional `"base"` key so a `SIDMethod` can default to them.
- **Subsumed helpers**: config lookups (`get_graph_building_conf`/`get_encoder_conf`) are now `REGISTRY.get_graph_building_config`/`get_encoder_config`. `get_graph_builder` (`data/graph.py`) and `get_encoder` (`nn/__init__.py`) remain as the last ad-hoc name→fn tables (see §1.7).
- CLI choices are **not** yet derived from `Registry.keys()` (still hand-synced in `main_gensid.py`).

### 6.3 Dataset resolver (`data/base.py`)
- `get_dataset_config(args)` is the resolver switch: `--sid_model grqvae` → `GRQVAEFusionDatasetConfig`;
  `--graph_augmentation` → `MixedDepthDatasetConfig`; `--graph_signal` → `GraphFusionDatasetConfig`; else
  `PlainDatasetConfig`. The graph-building config always comes from `REGISTRY.get_graph_building_config(args.graph_type)`.
- `build_dataset` looks up `REGISTRY.get_dataset(config.name)`; `build_data_loader` picks the loader:
  `NeighborLoader` for the online-GNN dataset in train mode, `EdgeWeightCollator` when `--use_edge_reconstruction_loss`,
  plain `DataLoader` otherwise.
- Batches are **keyed dicts** (`{"emb", "index", "target", ...}` plus optional `spectral_vectors`/`cluster_labels`/
  `edge_weights`); trainers consume by key, killing the hand-synced tuple cascade (U8).
- `MixedFusionDepthDataset` still samples its hop per `__getitem__` (non-deterministic; flagged for a later decision).

---

## 7. Extension points (why this shape is future-proof)

| New thing | Where it goes | What it reuses |
|---|---|---|
| New SID method (e.g. `capsid`, `balanced`, `faiss`) | a `SIDMethod` subclass in a module under `gensid/method/` (auto-registered by name) + its `ModelConfig` in `config/model_config.py` + a trainer in `trainer/` | `trainer/` loops, `generation/base.py`, `generation/utils.py`, config dataclasses |
| New representation refiner (APPNP-like conv, GNN) | `nn/gnn.py` (a torch_geometric `nn` layer) + an `EncoderConfig` subclass (auto-registered by name) | graph building in `data/graph.py`; offline path via `data/dataset.py` + `get_encoder` |
| New graph builder flavor | a `build_*` fn in `data/graph.py` (+ `get_graph_builder` entry) + a `GraphBuildingConfig` subclass | `InteractionGraphBuilder`, `.to_kwargs()` |
| New dataset family | `data/dataset.py` + a `DatasetConfig` subclass (auto-registered by name) + `data/base.py::get_dataset_config` switch | `data/graph.py` graphs, `data/utils.py` collate/mask helpers |
| New evaluation metric | `eval.py` | `generation/base.py` report writer |
| New quantizer layer | `nn/layers.py` | DISTANCE/MEAN + conv registries |
| New generation strategy | `generation/*` | shared formatter + collision utils |

---

## 8. Open decisions & deferred items

- **"Encoder" model — RESOLVED and applied**: no `encoders/` subsystem/ABC; encoders are `nn` layers (`nn/gnn.py`)
  and/or offline dataset preprocessing; only `EncoderConfig` (config) survives. The `encoder/` placeholder was
  **deleted**.
- **Interim dispatch helpers**: mostly deleted/replaced by `REGISTRY` (§6.2). `data/graph.py::get_graph_builder`
  and `nn/__init__.py::get_encoder` are the last two ad-hoc tables — fold them into registry-driven flow when the
  registry's groups absorb graph builders and conv encoders; do NOT grow them (U4/U10).
- **`S2GRConv` `config.K` vs `n_propagations`** — **fixed** (§6.1).
- **`Trainer` vs `VQTrainer`** — resolved: `Trainer` (base) + `RQTrainer` + `RKMeansTrainer`/`HierarchicalRKMeansTrainer`;
  `VQTrainer` is gone. The base `Trainer` and `RQTrainer` remain the thinnest-covered files (see next item).
- **Coverage**: `tests/` now exists (14 files, `pytest` green, ~60% coverage); `pipeline.py` and the trainers are the
  lowest-coverage targets (end-to-end via `scripts/smoke_test.sh`).
- **Dead code fate**: `GRQVAE` now wired and smoke-tested; the monofiles (mincut clustering, mmgrec adapter) and the
  mmgrec toy `RQVAE` remain bespoke — folded in Phase 3, fate decided in a later cleanup pass.
- **`MixedFusionDepthDataset` determinism**: still per-`__getitem__` random hop sampling; rewrite to indexed/seedable
  hop sampling, or drop (TODO #2.2). The per-sample `rng` is seeded from `--seed` already.
- **Config JSON sidecars**: `Trainer._save_checkpoint` saves the config dataclass (under `"args"`) plus a metrics JSON;
  `config.to_json()` round-trip for portable checkpoints still to land.
- **CLI choices from the registry**: `--sid_model` (and `--graph_type`) choices in `main_gensid.py` are still hand-synced;
  deriving them from `Registry.keys()` is TODO #3's acceptance criterion.
- **Dataset contract** vs `data/base.py`: keyed-dict batches are the agreed `Sample`; `Sample`/`DatasetSpec`-named
  dataclasses are not strictly needed anymore.
- **`map_gensid_to_genrec.py`/`DATASET_N_ITEMS`** — deferred to the #10 coupling work.
- **genrec/ and scripts/production/** — out of scope; only `gensid/` + root CLIs are restructured.
