# GrIS / gensid — Technical TODO & Refactoring Roadmap

> Living document. Every item below is grounded in the current `gensid/` (and, where relevant,
> `genrec/`) implementation so that any future session can pick it up, understand *why* it exists,
> and implement it without re-deriving the context from scratch.
>
> Repo layout quick map (as of this writing — see `docs/restructure.md` for full status):
> - `main_gensid.py` — thin CLI: parse flat `args` Namespace -> `SIDPipeline(args, device).train()/.generate()`.
>   `--sid_model` / `--graph_type` choices are **still hand-synced** with the registry (`main_gensid.py:30,72`).
> - `gensid/registry.py` — `Registry` + `REGISTRY` hub; **automatic** discovery of concrete subclasses via
>   package introspection (`iter_subclasses`) across 7 groups: `sid_method`, `trainer`, `trainer_config`,
>   `dataset`, `dataset_config`, `graph_building_config`, `encoder_config`. No decorators needed.
> - `gensid/pipeline.py` — `SIDPipeline`, the one orchestrator (build dataset/model/trainer; `train()` / `generate()`);
>   replaces the old `SIDBuilder`.
> - `gensid/config/` — typed dataclass package (was a single `config.py`): `graph_build_config.py`,
>   `encoder_config.py`, `dataset_config.py`, `model_config.py`, `trainer_config.py`, `docstring.py`.
> - `gensid/method/` — SID models on the `SIDMethod` interface: `base.py`, `rqvae.py`, `rkmeans.py`
>   (also `HierarchicalRKMeans`), `grqvae.py` (= RQVAE + `nn/gnn.GNNFusionEncoder`).
> - `gensid/trainer/` — `base.py` (shared epoch loop, typed `TrainerConfig`, collision-based ckpt selection,
>   metrics JSON sidecar), `rq_trainer.py` (`RQTrainer`: VQ init pass, single-pass generation by default with
>   an optional Sinkhorn collision-fix branch behind `--fix_collisions`), `rkmeans_trainer.py`
>   (`RKMeansTrainer`/`HierarchicalRKMeansTrainer`).
> - `gensid/data/` — `base.py` (dataset resolution + loaders, keyed-dict batches), `dataset.py`, `graph.py`
>   (graph builders + `InteractionGraphBuilder`; old `graph_aware_sid/` folded in), `utils.py`.
> - `gensid/nn/` — shared non-method building blocks: `gnn.py` (`GNNFusionEncoder`, `APPNPConv`, `S2GRConv`),
>   `layers.py` (`ResidualVectorQuantizer`, `TorchKMeans`, …), `utils.py` (`constrained_km`, `sinkhorn_algorithm`).
> - `gensid/generation/` — `base.py` (`collect_indices` / `report_and_save`), `utils.py` (`format_prefix`, cluster helpers).
> - Bespoke (not yet on the interface): `gensid/hierarchical_mincut_clustering.py`
>   (DMoN / spectral / recursive coarse-to-fine), `gensid/mmgrec_gnn_sid_adapter.py` (GNN-encoder SID),
>   `gensid/generate_graph_cluster_indices.py` (thin CLI wrapper).
> - `genrec/` — generative-recommendation sibling repo (pipeline + model/tokenizer/dataset registries + dict/YAML config);
>   the *consumer* reference; not the thing to develop.
> - `scripts/research/*.sh` — orchestration that strings gensid -> map -> genrec together by hand.
>
> Current testing status: **`tests/` exists and is green** — 14 files, 344 tests collected, full-suite
> statement coverage ≈60% (measured with `pytest-cov`, configured in `pyproject.toml` addopts
> `--cov=gensid --cov-report=term --cov-report=html:htmlcov`; no `fail_under` gate yet).
> Thinnest-covered-but-critical: `pipeline.py` (~0%) and the trainers (~34-36%) — those paths are validated
> end-to-end via `scripts/smoke_test.sh` (exercises every `main_gensid.py` branch on CPU).

## Suggested execution order

1. **Close the remaining Phase 3/4 loops** (they are the tail of the landed #1/#2/#3 items — see the
   "Remaining bits" subsections below): CLI choices derived from `REGISTRY.keys()`, folding the bespoke
   monofiles onto the `SIDMethod`/`SIDPipeline` interface, config-JSON checkpoint round-trip, and
   `MixedFusionDepthDataset` hop determinism.
2. **#6 (balanced SID)** and **#7 (faiss PQ/OPQ/RQ)**: independent research paths that both drop into the
   registry; share the #4/#5 eval+test harness.
3. **#8 (literature methods)**: becomes cheap once the monofiles are folded; driven by research priorities.
   **#9 (CapsID / SemanticBPE)** is the flagship item in this menu — a direct replacement for the RQ-VAE core,
   and the concrete validation of the registry + the variable-length artifact contract needed by #10.
4. **#10 (coupling)**: last, because it depends on stable contracts from #1-#3, on #7 shrinking the duplicate
   logic in `genrec`, and on #9 nailing down the variable-length SID contract.

---

## 1. Reduce reliance on `args` into all objects (dataclass-based configuration) — **DONE**

### What landed
- `gensid/config/` is now a **typed dataclass package** (was a proposed single `gensid/config.py`):
  - `graph_build_config.py` — `GraphBuildingConfig` + `AdjacentCooc/AdjacentCoocProd/RandomWalk(rw)/Windowed` (per-`graph_type` union),
  - `encoder_config.py` — `EncoderConfig` + `GraphEncoderConfig` + `APPNP/S2GR/GNNFusion/GraphSage` (name discriminant),
  - `dataset_config.py` — `DatasetConfig` + `GraphDatasetConfig` + `Plain/GraphFusion/MixedDepth/GRQVAEFusion`,
  - `model_config.py` — `ModelConfig` + `RQVAE/GRQVAE/RKMeans/HierarchicalRKMeans`,
  - `trainer_config.py` — `TrainerConfig` + `RQ/RKMeans/HierarchicalRKMeans`,
  - `docstring.py` — `auto_docstring` splices `Args:` blocks onto model class docstrings.
- Every config exposes `from_args(args)` to read the flat CLI Namespace; `SIDPipeline`/`SIDMethod`/`Trainer`
  consume typed configs, not bare `args`. The old monolithic `gensid/config.py` is gone.
- Checkpoints now persist the typed config dataclass under the `"args"` key plus a **metrics JSON sidecar**
  (`trainer/base.py:155,157-183`: epoch / best_loss / best_collision_rate / ckpt_path; `torch.save(..., pickle_protocol=4)`),
  replacing the old bare-`Namespace`-pickle.

### Remaining bits
- **argparse bool coercion is still broken**: `--bn type=bool default=False` and `--kmeans_init type=bool default=True`
  (`main_gensid.py:41,43`) — argparse does **not** coerce `bool`, so `--bn False` becomes the truthy string `"False"`.
  Prefer `store_true` / `BooleanOptionalAction` (the old `acceptance: --bn False parses correctly` is still open).
- **Config JSON round-trip**: the *metrics* sidecar exists, but the typed *config* is not yet dumped as JSON
  (the old `acceptance: checkpoint JSON config round-trips / fail loudly on incompatible keys` is still open).
- The old `trainer.py Trainer.batch_size AttributeError` bug is gone (typed configs), and `old_main_gensid.py`
  is **deleted** — both objectives of this item are met.

---

## 2. Dedicated dataset resolution for Emb, APPNP, etc. — **DONE**

### What landed
- Resolution moved out of the old `setup_dataset_and_loader` god-function into `gensid/data/base.py`:
  `get_dataset_config(args)` -> `DatasetConfig`, then `build_dataset(config)` / `build_data_loader(config)`
  (NeighborLoader for `GRQVAE`-family train; `EdgeWeightCollator` when `--use_edge_reconstruction_loss`).
- Datasets are now **keyed-dict batches** (`{"emb","index","target", …}` plus optional `spectral_vectors` /
  `cluster_labels` / `edge_weights`) — the fragile positional `__getitem__`-tuple cascade (trainer-side unpack
  `if/elif`) is gone, and dataset families are registry-discovered
  (`plain`, `graph_fusion`, `mixed_depth`, `grqvae_fusion`).
- Dataset families live in `gensid/data/dataset.py`: `EmbDataset`, `GraphAwareEmbDataset`, `GNNFusionEmbDataset`,
  `GRQVAEFusionEmbDataset`, `MixedFusionDepthDataset`; graph-building consolidated in `gensid/data/graph.py`
  (graph builder fns + `get_graph_builder` + `polars_graph_to_pyg` + `compute_graph_spectral_vectors` + `InteractionGraphBuilder`).
- `_load_interactions` duplication (old datasets_module + graph_utils) resolved — interaction loading is shared.

### Remaining bits
- `MixedFusionDepthDataset` still samples its hop **per `__getitem__`** (`gensid/data/dataset.py`) —
  non-deterministic even with a seeded `rng`. The `acceptance: train/eval reproducible / indexed per-item hop`
  from the original item is still open (flagged in `docs/restructure.md` §8).

---

## 3. Registry structure to simplify the dedicated resolution — **DONE**

### What landed
- `gensid/registry.py` — a real `Registry` + `REGISTRY` hub. Registration is **automatic** (package introspection
  over concrete subclasses via `iter_subclasses`); no decorators. Accessors: `REGISTRY.get_<group>(name)` /
  `REGISTRY.get_<group>_config(name)` / `REGISTRY.keys(group)`.
- Live groups today (verified):
  - `sid_method` -> `['grqvae','hierarchical_rkmeans','rkmeans','rqvae']`
  - `graph_building_config` -> `['adjacent_cooc','adjacent_cooc_prod','rw','windowed']`
  - `dataset`/`dataset_config` -> `['graph_fusion','grqvae_fusion','mixed_depth','plain']`
  - `encoder_config` -> `['APPNP','GNNFusion','GraphSage','S2GR']`
  - `trainer`/`trainer_config` -> `['base','hierarchical_rkmeans','rkmeans','rqvae']`
- `SIDPipeline._build_model` resolves the method via `REGISTRY.get_sid_method(name)`; the trainer is resolved via
  the method's `trainer` class attribute. Adding a method now = new module under `gensid/method/` with a `name`
  class attr; the registry picks it up automatically.
- The old ad-hoc dispatch tables are nearly gone. Two interim name->fn tables remain **by design**
  (do NOT grow them): `data/graph.py::get_graph_builder` and `nn/__init__.py::get_encoder`.

### Remaining bits
- **CLI is still hand-synced**: `--sid_model choices=[...]` (`main_gensid.py:30`) and `--graph_type choices=[...]`
  (`main_gensid.py:72`) must be derived from `REGISTRY.keys()` so a new method needs **no edit to `main_gensid.py`**
  (the original `acceptance: adding a method = new module, no CLI edit`).
- **Bespoke monofiles not yet on the interface** (the open Phase-3 work; see `docs/restructure.md` §5):
  - `hierarchical_mincut_clustering.py` (`stacked_dmon` / `recursive_spectral` / `recursive_dmon`) -> registered
    `{recdmon,spectral,recursive}` SID methods,
  - `mmgrec_gnn_sid_adapter.py` -> `SIDPipeline(sid_model="rqvae")` + a registered refiner; delete its toy
    `RQVAE` (mmgrec_gnn_sid_adapter.py:157), its own `DEFAULT_PREFIXES` (:34) and `load_interactions` (:53),
  - `generate_graph_cluster_indices.py` (thin CLI; pure logic already moved to `generation/utils.py`) — **deprecated, kept as reference only** (clustering methods `recdmon`/`spectral`/`stacked_dmon` produce the SID mapping in a single pipeline); do not develop or delete it.

---

## 4. Implement unit tests — **DONE**

### What landed
- `tests/` exists (14 files, 344 tests, green). The old "zero tests" reality and every pure unguarded unit it
  listed (`gensid/utils.py` collision helpers, `gensid/eval.py` metrics incl. the shift-index gini trick, k-means /
  VQ codebook forward/indices, collision-resolution logic) now has coverage.
- Tests are CPU-only and deterministic, per the original guidance.
- The old `batch_size`-style regression scenario is moot (typed configs — construction can't silently mis-unpack).

### Remaining bits
- Coverage is measured but **not gated**: full-suite ≈60% statement coverage; `pipeline.py` and the trainers
  (RQTrainer / RKMeansTrainer / HierarchicalRKMeansTrainer) are the thinnest. The end-to-end `scripts/smoke_test.sh`
  covers these paths, but dedicated unit tests for `pipeline.train()`/`generate()` and the trainers would close
  the gap. (Add a `[tool.coverage.report] fail_under` when ready.)

---

## 5. Easily implement coverage to know which tests to focus on — **DONE**

### What landed
- `pytest-cov` in dev extras; `pyproject.toml` pytest `addopts` now runs `--cov=gensid --cov-report=term
  --cov-report=html:htmlcov`; `[tool.coverage.run]` configured (source = gensid).
- One command (`python -m pytest`) produces a per-file line report + `htmlcov/` for interactive exploration.
- The old risks this item called out are gone (no more god-functions with untested branches):
  `setup_dataset_and_loader` / `_train_epoch` tuple-cascade / `_run_vae_generation` are replaced by
  `data/base.py` + keyed-dict batches + `RQTrainer.generate_indices`.

### Remaining bits
- No `fail_under` gate yet (see #4). Coverage dashboard is produced, not enforced.

---

## 6. Implement balanced SID

### Context (current state)
- **Unbalance is measured but not optimized.** `eval.check_id_distribution` already reports `gini`, `perplexity`,
  `utilization` (`gensid/eval.py:10`; gini at `eval.py:114`) and the trainers report collision rate.
- Imbalance signals in the code:
  - `_calculate_unused_codebooks` (`trainer/base.py:140-145`) tracks unused codebook entries per layer;
    `used_codebooks` sets are accumulated during training (`trainer/base.py:108-110`, `trainer/rq_trainer.py:186-192`)
    — this data is logged but never used to steer training.
  - The generation-time **constrained K-Means** (`k_means_constrained.KMeansConstrained`) re-clusters codebook
    embeddings to bound cluster sizes. It's now **single-sourced** in `gensid/nn/utils.py:31` `constrained_km`
    (the old duplication across `trainer.py` and `builders.py` is gone; consumer: `trainer/rq_trainer.py:261`);
    still numpy/sklearn (CPU), runs *after* training, and only affects the *prefix labels*, not the learned codebooks.
  - Sinkhorn parameters `--sk_epsilons` (`main_gensid.py:49`) are used **post-training** to resolve collisions
    (`RQTrainer.generate_indices`, `trainer/rq_trainer.py:272-295`) and the last VQ layer's `sk_epsilon` is nudged
    to `0.003` at train time (`trainer/rq_trainer.py:277-278`) — not a training-time balanced-assignment mechanism.
  - A `--beta` diversity weight exists (`main_gensid.py:58`, threaded into `RQVAE`) but is a soft loss term,
    not a hard balance guarantee.
- So today "balance" is a post-hoc, hybrid-CPU patch, at SID-extraction time only.

### Why it's needed
- Codebook imbalance -> unused tokens -> wasted vocabulary and worse downstream (generative) modeling; the
  `_extend_semantic_ids` conflict check in `genrec` (TIGER tokenizer) bakes this in: if per-level conflicts exceed
  codebook size it *errors* (`genrec/models/TIGER/tokenizer.py`).
- Making balance a first-class, training-time property is a research lever (better utilization/perplexity/gini)
  and would let the post-hoc `constrained_km` hack be retired.

### Proposed approach
1. Decide on the balance strategy (a research choice — see open questions):
   - **Balanced assignment during training**: Sinkhorn-based soft assignment (already have `sk_iters` infra in
     `gensid/nn/layers.py` / `nn/utils.sinkhorn_algorithm`) applied per batch, driving codebook usage toward uniform.
   - **Balanced-init + auxiliary entropy/commitment losses** on top of `RQVAE` (promote `used_codebooks` /
     `_calculate_unused_codebooks` into a differentiable regularizer).
   - **Balanced K-Means** variant of `RKMeans`/`TorchKMeans` that keeps cluster sizes bounded during `fit`
     (replaces the post-hoc `KMeansConstrained`).
2. Add an explicit `balanced_*` SID method to the registry — e.g. `balanced_rq` wrapping the Sinkhorn-assisted
   training — rather than bolting a flag onto existing ones, so configurations stay orthogonal and comparable.
3. Wire `gini`/`perplexity`/`unused_codebooks` into the trainer's checkpoint-selection metric alongside collision
   rate (`trainer/base.py` ckpt selection) so "balanced" can be optimized, not just measured.

### Open questions to resolve before implementing
- Target metric: minimize unused codebooks vs maximize perplexity vs hard size bounds — pick one primary.
- Balanced at *codebook level* (each VQ layer uniform) vs *joint SID level* (full-tuple uniform) — affects whether
  balance goes in the VQ loss or in a post-hoc assignment (like LETTER-style reranking).

### Acceptance criteria
- A single balanced method runs **without** `KMeansConstrained`; `gini`/`unused_codebooks` measurably improve vs
  the current baseline on an eval dataset, verified through `check_id_distribution`.
- `constrained_km` remains in exactly one place (`gensid/nn/utils.py`) — or is removed entirely.

---

## 7. Implement faiss-based SIDs (PQ, OPQ, RQ, …)

### Context (current state)
- `gensid` has **zero faiss usage**. Its quantizers are:
  - `RQVAE` (learned residual VQ, `gensid/method/rqvae.py`; GRQVAE variant in `method/grqvae.py`)
  - `RKMeans` / `HierarchicalRKMeans` (residual K-Means over `nn/layers.TorchKMeans`, `gensid/method/rkmeans.py`)
  - graph-cluster coarse-to-fine SIDs (`hierarchical_mincut_clustering.py` + `generate_graph_cluster_indices.py`)
  - None of these are product quantizers; no PQ/OPQ/IVF anywhere.
- Meanwhile `genrec` re-implements a **faiss RQ SID inside the tokenizer**: `_generate_semantic_id_faiss`
  lazily `import faiss`, builds `faiss.IndexResidualQuantizer(n_codebooks, n_bits, METRIC_INNER_PRODUCT)` and decodes
  codes via `BitstringReader` (`genrec/models/TIGER/tokenizer.py`). This is (a) a duplicated SID implementation
  living in the wrong repo, and (b) the only faiss reference in the codebase.
- `faiss-cpu>=1.7` is an **optional extra** (`pyproject.toml:45-47`, `GrIS[faiss]` in the dev group) — added with the
  `FaissKMeans` backend (`nn/layers.py:664`); the SID methods themselves still use zero faiss.
- `faiss` is imported lazily via `_get_faiss` (`nn/layers.py:358-366`), so it works only where faiss is installed.

### Status (implemented)
- **PQ and OPQ are implemented, both torch-native and faiss-native**, in `gensid/nn/pq.py`
  (`TorchPQ`/`FaissPQ`/`TorchOPQ`/`FaissOPQ` + `make_pq`/`make_opq` factories). The torch backends are
  device-agnostic (CPU/GPU/NPU via `get_device`); the faiss backends run on CPU and mirror their codebook/rotation
  into torch tensors so the two are interchangeable.
- **Parity is verified by tests** (`tests/test_nn_pq.py`): given the same trained state, torch reproduces faiss's
  packed codes bit-for-bit and decode parity holds; end-to-end distortion is within a ±15% band (a quality-
  equivalence check: both backends are stochastic with <10% run-to-run spread, so the band catches genuine gaps).
- **A scaling harness** (`scripts/pq_opq_scaling.py`) sweeps `N x D x M x nbits` and asserts strict code parity
  (1.0), distortion-equivalence and reports train/encode wall-time + peak RSS per backend, so the CPU-vs-CPU and
  (with `--device`) CPU-vs-GPU/NPU gap can be measured as parameters scale.
- **Registered SID methods** `pq` and `opq` (`gensid/method/pq.py`, configs `PQModelConfig`/`OPQModelConfig`,
  trainer `clustering`), wired into `main_gensid.py` (`--M --nbits --quantizer_backend --opq_niter`) and
  `scripts/smoke_test.sh` (`pq`, `pq_faiss`, `opq` variants). Both emit the standard
  `{item_id: ["<a_i>", ...]}` mapping with `M` codebooks of size `2**nbits`.
- **Still open**: `faiss_rq` (`IndexResidualQuantizer`, port the `BitstringReader` decode from genrec); optional
  `faiss-cpu` extra in `pyproject.toml`; deprecating the faiss RQ inside `genrec/models/TIGER/tokenizer.py`
  (coordinate with #10); head-to-head `eval.check_id_distribution` on a real embedding file.

### Why it's needed
- PQ/OPQ/RQ give fast, scalable, codebook-free quantization with mature CPU/GPU kernels — ideal for large catalogs
  (item counts up to ~94k for Yelp, `map_gensid_to_genrec.py` `DATASET_N_ITEMS`) and for A/B comparison against
  learned RQ-VAE.
- OPQ (product-quantizer preprocessing rotation) typically improves PQ variance — a cheap research lever not
  available today.
- Consolidating faiss into `gensid` also fixes the coupling direction in #10: SIDs should be produced by `gensid`
  and consumed by `genrec`, not re-derived by `genrec`'s tokenizer.

### Proposed approach
1. Add `faiss-cpu` (optional extra) to dev/extra deps.
2. Implement a generic `FaissSID` in `gensid` (new module, e.g. `gensid/faiss_sid.py`) with a `quantizer_type` knob,
   registering variants in the registry:
   - `faiss_pq` -> `faiss.IndexPQ(d, M, nbits)`
   - `faiss_opq` -> `faiss.IndexPreTransform` + `faiss.OPQMatrix(d, M)` + `IndexPQ`
   - `faiss_rq` -> `faiss.IndexResidualQuantizer` (port the `BitstringReader` decode from genrec so both produce
     `n_codebooks x code` integer matrices)
   - (stretch) `faiss_ivfpq` for very large catalogs.
   > Done (PQ/OPQ) in a slightly different shape: `pq`/`opq` SID methods with a `--quantizer_backend torch|faiss`
   > knob (`gensid/nn/pq.py` + `gensid/method/pq.py`) instead of separate `faiss_pq`/`faiss_opq` method names;
   > `faiss_rq` still to do.
3. Map `n_bits -> codebook_size` the same way genrec does (`rq_codebook_size = 1 << n_bits`) and emit SIDs in the
   *same* JSON contract as the other methods (`{item_id: ["<a_i>", ...]}`) so `map_gensid_to_genrec.py` and
   downstream training work unchanged.
4. Evaluate each variant with the shared harness: `eval.check_id_distribution` + collision rate, so faiss vs
   RQ-VAE vs RQ-KMeans are directly comparable.

### Acceptance criteria
- `faiss_pq`, `faiss_opq` both produce a `{item_id: ["<a_i>", ...]}` mapping via the `SIDMethod`/pipeline interface,
  validated on a small embedding file; both are registered and reachable through `main_gensid.py --sid_model`.
- Codebook contract: `n_codebooks == n_subquantizers`, `codebook_sizes == [1 << n_bits] * M`.
- `state_dict`/`load_state_dict` round-trip preserves the faiss index (same codes after reload) and the pipeline's
  generate path (`pipeline.py:78`, `strict=False`) works with no faiss-specific special case.
- Tests assert behavior (code values / round-trip), not just shapes.
- `faiss_rq` and the `BitstringReader` port are explicitly **deferred**; `genrec`'s `_generate_semantic_id_faiss` stays
  in place until #10 (update this criterion when RQ lands).

---

## 8. Implement other SID methods relevant from the literature

### Context (current state)
- Currently implemented methods:
  - RQ-VAE (TIGER-style; `gensid/method/rqvae.py`, `grqvae.py`)
  - RQ-KMeans / hierarchical RQ-KMeans (`gensid/method/rkmeans.py` — the LETTER-style light baseline)
  - Coarse-to-fine graph-cluster SIDs (DMoN/spectral/recursive clustering -> prefixes,
    `hierarchical_mincut_clustering.py`, `generate_graph_cluster_indices.py` — used by the MMGRec/GAE/Letter scripts)
  - faiss RQ (only inside `genrec`, see #7) and the graph-aware GNN-encoder path (`gensid/mmgrec_gnn_sid_adapter.py`)
- All RQ-family methods (and GRQVAE) are on the `SIDMethod` interface; the non-RQ methods are **still bespoke
  entrypoints**, not on `SIDMethod`/`SIDPipeline` — folding them in is the open Phase-3 work (`docs/restructure.md` §5).

### Why it's needed
- "Which tokenization/discretization produces the best SIDs for generative rec?" is an open research question the
  repo is positioned to answer, but the bespoke clustering/GNN paths still lack a common harness, so comparisons
  are laborious and inconsistent.
- A common harness (one interface: `fit(embeddings) -> {item: tokens}` + one evaluation:
  `check_id_distribution`/collision + downstream retrieval) turns "implement another method" from a research pivot
  into a registry entry.

### Proposed approach
1. First fold the bespoke methods onto the interface (see #3 remaining bits), so every method shares the
   `SIDMethod` interface and eval harness.
2. Candidate menu (curate with the user before implementing; each maps to a registry entry):
   - **CAPSID (soft-routed variable-length SIDs) + SemanticBPE** -> dedicated, higher-priority section **#9** —
     it targets this repo's hard-assignment RQ-VAE core; the flagship item in this menu.
   - **PQ/OPQ/RQ via faiss** -> covered by #7.
   - **Balanced RQ-VAE / balanced VQ** (uniform-codebook-usage training objective) -> covered by #6.
   - **HVQ-VAE (Hierarchical VQ)** latent-code hierarchies; and **ViT-VQGAN / tokenizer-style VQ** for
     encoder-decoder repurposing.
   - **LSA/traversal-based codes** (graph-ordered traversal assigning sequential IDs) — complements the existing
     spectral/coarse-to-fine clustering already in the repo.
   - **Return-based / reranking SIDs** (LETTER-style iterative refinement) — the current
     `RQTrainer.generate_indices` Sinkhorn collision-resolution loop (`trainer/rq_trainer.py:272-295`) is a
     primitive form of this; formalize it as a post-processor.
   - **Index-retrieval-oriented SIDs** (approximate-neighbor index structure over embeddings, index IDs as SIDs) —
     reuse the faiss infrastructure from #7.
3. For each: implement `fit/generate_indices` against the registry, add one `tests/` entry, and log
   `check_id_distribution` metrics to `results/` for head-to-head comparison.

### Open question for the user
- Which literature families are the research priority (linguistic/PUUQ-style? TIGER-lineage? graph-traversal)?
  The menu above is deliberately broad; the registry makes adding any of them cheap once the direction is chosen.

### Acceptance criteria
- Every new method runs through the same `SIDMethod`/pipeline interface + eval; one table in `results/` compares
  all methods by collision/utilization/gini/perplexity on a given dataset.

---

## 9. Implement CapsID + SemanticBPE (soft-routed, variable-length SIDs) — arXiv:2605.05096

> Paper: *CapsID: Soft-Routed Variable-Length Semantic IDs for Generative Recommendation* (arXiv:2605.05096, cs.IR).
> Abstract/summary of the method below is an original paraphrase written from the paper's framing; consult the PDF for
> the exact formulations and numbers before implementing. Key insight: in generative recommendation the **bottleneck is
> the tokenizer, not the transformer** — hard winner-take-all residual quantization collapses multi-faceted item
> semantics at cluster boundaries and lets early-layer errors propagate into later SID positions.

### The method in one paragraph (context for future sessions)
- **Hard-assignment pain point.** Today's SIDs (TIGER/RQ-VAE and every RQ-VAE descendant in `gensid/`) assign the
  item residual to exactly one code per depth via a nearest-neighbor argmax. Boundary/multi-facet items (e.g. a
  "travel cooking kit") get snapped to one noisy code, and the discarded facets become error signal for the next
  layer. The paper's framing categorizes prior fixes as *patch-based* (COBRA-style dense vector or UniRec-style
  attribute prefix appended alongside the SID — restores information but inflates inference and breaks the "single
  generative interface" ideal) vs *tokenizer-centric* (fix the SID itself). This repo's own
  `RQTrainer.generate_indices` Sinkhorn collision fixup (`trainer/rq_trainer.py:272-295`) and the LETTER-reranking
  lineage are exactly the "patch-ish" family.
- **CapsID** replaces the hard residual quantizer with *soft capsule routing*:
  - Each depth holds several capsules; each capsule has a pose transform + bias and produces a "vote" from the
    current residual. Routing-by-agreement is iterated a few times: normalized coupling weights over capsules, an
    aggregated capsule output, a squash nonlinearity, then coupling logits updated by vote-vs-output agreement.
  - The emitted token is the argmax coupling, and a *confidence* is derived from coupling mass x winner output norm.
  - Crucially, the residual is updated by the **routed reconstruction** (the coupling-weighted sum over all
    capsules), not by subtracting only the winning code — so secondary semantic facets keep contributing and only
    the truly unexplained part flows on. Input is L2-normalized before routing; per-depth capsule parameters
    specialize coarse to fine.
  - **Variable length**: the SID stops when confidence exceeds a threshold, OR residual norm is small, OR a hard
    depth cap is reached, plus a training-time length regularizer. Reported mean length lands around 3.4-3.9 with a
    cap of 6, and the cap fires for only ~8-10% of items (confidence/residual rules dominate).
- **SemanticBPE** builds reusable subword tokens on top of the CapsID sequence: adjacent SID tokens are merged only
  when they are supported by **both** co-occurrence frequency **and** embedding compatibility (cosine of token
  embeddings), with a Gumbel-Softmax gate for differentiability at train time and hard merges at inference; a
  similarity threshold is annealed from strict to moderate during training to avoid merging popular-but-unrelated
  prefix pairs.
- **Training protocol**: two-stage. Stage 1 trains the tokenizer-side losses only (reconstruction, a spread/entropy
  term with an annealed margin to keep capsules from collapsing, the length penalty, and a frequency-based BPE
  warm-up) with the sequence generator frozen. Stage 2 freezes capsule centers and keeps the BPE merge weights
  frozen while jointly training the generator with low-rank routing adapters (small rank) plus a learnable scalar
  bias on the BPE gate; the objective adds next-token cross entropy to the tokenizer terms.
- **Reported evidence** (from the abstract/experiments; verify in the PDF): +4.9%/6.7%/2.2% R@10 over ReSID on
  Beauty/Sports/Toys for CapsID alone, +8.9%/11.0%/8.8% with SemanticBPE, matching or exceeding a COBRA-style
  sparse+dense system at ~51% of its latency. Ablations attribute the gain primarily to the soft residual update
  (hard winner-only assignment is the largest single regression), then to iterative agreement (T=1 hurts), variable
  length (fixed L=2 over-compresses, fixed L=4 over-encodes), and the spread loss. Gains concentrate on tail items
  (largest relative R@10 jump on the tail) and the collision rate drops well below hard RQ baselines.

### Why it's needed in this repo
- `gensid/` **is** the hard-assignment RQ-VAE stack the paper targets: `gensid/method/rqvae.py`, `nn/layers.py`
  (per-layer nearest-neighbor codebook), with the shared loop in `trainer/base.py`, RQ-specific loop in
  `trainer/rq_trainer.py`, and orchestration in `SIDPipeline` (`pipeline.py`). CapsID is the most direct possible
  research upgrade to the core product here.
- The repo already carries many of the needed ingredients as byproducts: Sinkhorn infra (`--sk_epsilons`, used for
  post-training collision fixup, `trainer/rq_trainer.py:272-295`), `eval.check_id_distribution`
  (utilization/gini/perplexity), the post-hoc `constrained_km` codebook re-clustering (`nn/utils.py:31`,
  `trainer/rq_trainer.py:261` — conceptually related to balancing, but post-hoc, so CapsID's *routing-time* softness
  is the principled replacement), and a two-stage training culture already visible in the LETTER/mmgrec research
  scripts.
- It is the **best end-to-end stress test** for the roadmap in this file: it needs the registry (#3) + config
  dataclasses (#1) + the test/coverage harness (#4/#5) — all landed — AND it forces the variable-length artifact
  contract that #10 (coupling) must support — `genrec` today assumes a fixed SID depth (`rq_n_codebooks` + dedup/
  conflict codebook in `genrec`'s `BaseSIDTokenizer` and TIGER tokenizer).

### Proposed approach
1. **Prereqs**: all landed (#1 config dataclasses, #3 registry, #4/#5 test+coverage); just ensure a new `capsid`
   method needs no edit to `main_gensid.py` CLI (close the #3 remaining bit first), and that `RQVAE` exposes the
   `compute_loss`-style hooks `capsid` will reuse.
2. **Model** — new `gensid/method/capsid.py`:
   - `CapsuleLayer`: per-capsule pose transform (Linear) + bias, iterative routing (T rounds) mixing coupling
     softmax, weighted aggregation, squash, and agreement logit updates; expose per-capsule outputs for the
     residual update.
   - Soft residual update: subtract the coupling-weighted routed reconstruction (NOT only the winner).
   - Variable-length forward: stopping rules (confidence / residual-norm / depth cap), return tokens + confidences
     + lengths; keep one forward path for training (differentiable, confidences+gates) and one for inference
     (argmax, hard stops) — mirror the train/infer split the repo already uses for Sinkhorn.
   - Two-stage support: frozen centers in stage 2 + low-rank routing adapters + learnable BPE-gate scalar bias.
3. **Losses**: reconstruction, spread/entropy (margin annealed, e.g. 0.2->0.9), length penalty, SemanticBPE loss.
   Integrate through `RQVAE.compute_loss`-style hooks so the shared `trainer/base.py` epoch loop stays
   model-agnostic; add config fields only where needed (respect #1's dataclasses).
4. **SemanticBPE** — new `gensid/method/sembpe.py` (or under a `capsid/` package): score adjacent-token pairs by
   `alpha*freq + (1-alpha)*cos(token_emb, token_emb)`, Gumbel-Softmax gate (train) / argmax (infer), annealed
   similarity threshold; output shorter composed SID sequences.
5. **Generation**: emit variable-length JSON SIDs through `collect_indices`/`report_and_save`
   (`generation/base.py`). Update the artifact contract (with #10) so `map_gensid_to_genrec.py` and the genrec
   tokenizer can consume variable-length sequences (pad-with-stop or native variable-length support).
6. **Eval** — extend `gensid/eval.py` with the paper's tokenizer diagnostics beyond collision/utilization/gini:
   intra-code semantic similarity (purity), code predictability (CodeRecall@M — first-token recall of a SASRec-style
   generator over the SID sequences), head/torso/tail R@10, average SID length, and normalized inference cost. Wire
   into the generation-time report like `check_id_distribution` is used today (`report_and_save`,
   `generation/base.py`).
7. **Validation**: first reproduce the in-repo RQ-VAE (TIGER-style) baseline on an existing dataset (Beauty/Sports/
   Toys already flow through `prepare_dataset.py` + `main_gensid.py`), then add CapsID and SemanticBPE, and ablate
   each mechanism following the paper's ablation structure (soft residual vs hard, T=1 vs T~3, fixed vs variable
   length, with/without spread loss, with/without SemanticBPE).

### Open questions / decisions to confirm with the user before coding
- Reproduce on the small public Amazon datasets (5-core, leave-one-out) as the primary comparison, or also target
  the repo's graph/spectral SID variants as extra baselines?
- Should `sem_ids` embeddings (EmbDataset / GNN-fused features) serve as the item representation x_i — the paper
  uses content (+behavior) encoders per dataset; confirm we treat `*.emb-*.td.npy` as x_i for a first faithful port.
- SemanticBPE should live in `gensid` (single SID producer, aligns with #10) rather than `genrec` — recommended.
- Hyperparameters to port: routing rounds T (paper reports agreement saturating around T=3), depth cap L_max=6,
  spread margin anneal schedule, per-loss weights, low-rank adapter rank (~8).

### Acceptance criteria
- `capsid` and `capsid_sembpe` are registered SID methods that run end-to-end via the pipeline interface with typed
  config dataclasses (no bare-`args` access).
- Unit tests cover: the routing math (coupling softmax, squash, agreement monotonicity, L2-normalization), the three
  stopping rules, the soft-vs-hard residual reconstruction relationship, the length-boundedness behavior, and
  SemanticBPE merge scoring + train/infer gate.
- Eval harness reports the paper's diagnostics (collision, utilization, gini, intra-code similarity, CodeRecall@M,
  tail R@10, avg length).
- Direction of results matches the paper's ablation story: soft residual assignment is the dominant factor, gains
  are largest on tail items, and variable-length yields mean length < cap with the hard cap rarely binding.
- `genrec` consumes the variable-length SIDs without relying on a fixed-depth dedup-codebook assumption
  (coordinated with #10).

---

## 10. Verify the coupling with `gr_unified` (generative recommendation) — behave like two submodules of one repo

### Context (current state)
- Hand-rolled, file-based orchestration couples the two today:
  - `scripts/research/run_letter.sh`, `run_s2gr.sh`, `run_gae_components_ablation.sh`, `run_mmgrec.sh`,
    `run_recdmon.sh` run `main_gensid.py` (or the adapter), then call `map_gensid_to_genrec.py`, which writes a
    `.sem_ids` JSON + an **env file** (`_write_shell_env_file`, `map_gensid_to_genrec.py:47-63`) with
    `GR_SEM_IDS_*` vars that downstream training sources.
  - `map_gensid_to_genrec.py` **hardcodes** `DATASET_N_ITEMS = {Beauty: 12101, Sports_and_Outdoors: 18357, ...}`
    (`map_gensid_to_genrec.py:9-14`) — item counts that must match `genrec/datasets/*` exactly, with no shared
    source of truth.
  - `genrec` contains its **own SID generation** (tokenizer `_generate_semantic_rqvae` /
    `_generate_semantic_id_faiss`, `genrec/models/TIGER/tokenizer.py`) and its own model registry
    (`genrec.utils.get_model`, `get_tokenizer`, `get_dataset`, `get_pipeline`), so the "ID producer" and "ID
    consumer" each hold a partial, duplicated view of SIDs.
  - `old_main_gensid.py` (the legacy third flavor) is **deleted** — only `genrec/trainer.py` still mirrors gensid's
    trainer externally.
- Both packages already live in **one repo/venv** and share `pyproject.toml` (packages `gensid*` and `genrec*`
  both included) — the physical foundation for a monorepo/submodule arrangement already exists; only the interfaces
  are ad-hoc.

### Why it's needed
- The env-file + magic-item-count contract is fragile (silent mismatch -> wrong SIDs -> wasted training runs) and
  opaque to both sides.
- Duplicated SID logic (gensid quantizers vs genrec tokenizer faiss/RQ-VAE) means improvements land in one repo
  and not the other.
- The `genrec` structure (abstract `Pipeline`/`AbstractDataset`/`AbstractModel`/`AbstractTokenizer` + registry
  `utils` + dict/YAML config) is a clean pattern that `gensid` now mirrors (registry + typed config); the remaining
  work is one orchestration entry point + a shared artifact layer.

### Proposed approach
1. **Define a formal SID artifact contract** (JSON schema) produced by `gensid` and consumed by `genrec`:
   `{item_id: ["<a_i>", ...]}` plus a sidecar `meta.json` (config, codebook sizes, dataset, item count, method
   name) — extending the precedent of the checkpoint **metrics JSON sidecar** (`trainer/base.py:155-183`) and
   `InteractionGraphBuilder.dump()` (`data/graph.py`).
2. **Single source of truth for dataset sizes**: replace `DATASET_N_ITEMS` with a lookup derived from
   `genrec.datasets` metadata (or a shared `data_catalog`), and validate item count inside the mapper.
3. **Move SID production into `gensid` entirely**: retire the faiss/RQ-VAE paths inside
   `genrec/models/TIGER/tokenizer.py` once #7/#8 land, so `genrec` only *consumes* `.sem_ids` files (tokenizer
   already supports loading them via `item2sem_ids`).
4. **Unify configuration**: both sides now use registries + typed config / dict+YAML conventions; provide a single
   orchestrator (Python `pipeline` module or one `run.sh`) that does gensid-train -> gensid-generate -> map ->
   genrec-train as one callable flow instead of hand-rolled shell + env files.
5. **Repo hygiene**: if a real "two submodules" split is desired later, restructure the repo into
   `submodules/gensid` + `submodules/genrec` (or git submodules) with a top-level manifest — worthwhile only
   *after* the interfaces in 1-4 are stable. (`old_main_gensid.py` is already gone.)

### Acceptance criteria
- No `GR_SEM_IDS_*` env-file handshake; mapper reads/writes a validated artifact + meta.
- `genrec` has no SID *generating* quantizer code left (only consumption + validation against the codebook-size
  contract).

---

## 11. Match GrIS_v1's collision-free generation + argparse-standardized SID formatting — **DONE**

### Context (what was wrong)
- GrIS_v1 (`GRIS_v1/RQ-VAE/generate_indices.py`) generates SIDs with a **single** nearest-neighbour pass
  (`model.get_indices(..., use_sk=False)`) and **never** resolves residual collisions (it only prints the collision
  rate). The in-repo `RQTrainer.generate_indices` diverged: it always ran a **two-pass** flow — pass 1 fast,
  pass 2 a Sinkhorn collision-fix loop (capped by `max_sk_fix_collision_iters`, last-layer `sk_epsilon` nudged to
  `0.003`). The 2nd pass is significantly slower and prone to NaN/inf (the Sinkhorn output is only checked, not
  recovered), which is exactly why GrIS_v1 dropped it.
- SID formatting params were hardcoded: `format_prefix` (`generation/utils.py`) fixed the `<a_{}>`/`<b_{}>`…
  letter prefixes, and the clustering trainers hardcoded `unclustered_strategy="special"` +
  `unique_prefix="<z_{}>"` (recursive_clustering_trainer.py / stacked_dmon_trainer.py).

### What landed
- **Sinkhorn fix pass is now an opt-in branch**: `RQTrainerConfig.fix_collisions` (CLI `--fix_collisions`),
  **default False** = single nearest-neighbour pass with collisions left unresolved, matching GrIS_v1
  (`trainer/rq_trainer.py::generate_indices`). The `sk_epsilon` reset/bump and the Sinkhorn loop only run inside
  the branch, still capped by `max_sk_fix_collision_iters`. Training-time Sinkhorn (`disable_sk`) is unchanged.
- **SID formatting is argparse-controlled** on `main_gensid.py`:
  - `--prefixes` (per-level prefix format strings, default None = letter-based `<a_{}>`/`<b_{}>`…),
  - `--unclustered_strategy` (`special`/`unique`/`error`, default `special`),
  - `--unique_prefix` (leaf prefix, default `<z_{}>`).
  Threaded through `TrainerConfig` (`prefixes`/`unclustered_strategy`/`unique_prefix`) and consumed by
  `generation/base.collect_indices` (RQ path, new `prefixes` arg) and `build_codes` (clustering path, replacing
  the hardcoded `special`/`<z_{}>` in the recursive/stacked-DMoN trainers). Defaults preserve prior behavior.
- Tests: existing two-pass tests now opt in via `fix_collisions=True`; new `test_single_pass_default_matches_gris_v1`
  pins the default single-pass behavior (no Sinkhorn calls, no `sk_epsilon` bump, collisions left as-is);
  `test_custom_prefixes_used_in_string_buffer` covers the new `collect_indices(prefixes=...)` arg.
  `scripts/smoke_test.sh` gained a `rqvae_fix_collisions` variant exercising the branch end-to-end.
- **Compatibility break is surfaced, not silent**: this is an intentional breaking default (single-pass vs the old
  always-on 2nd pass). To preserve the exact prior behavior of the existing RQ/GRQ-VAE pipelines that feed
  collision-sensitive consumers (e.g. genrec's TIGER tokenizer errors when per-level conflicts exceed the codebook
  size), the following callers now pass `--fix_collisions` explicitly: `scripts/research/run_letter.sh`,
  `run_gae_components_ablation.sh`, `run_s2gr.sh`, `scripts/production/run_gae.sh`, `run_gae_mtp.sh`,
  `train_tokenizer.sh`, and `scripts/benchmark_prod.sh` (in its shared BASE; a no-op for clustering methods).
  `RQTrainer.generate_indices` additionally logs a `WARNING` whenever `fix_collisions` is off, so a direct
  `main_gensid.py` run (no script wrapper) can't silently drop the collision-fix pass.

### Remaining bits
- `mmgrec_gnn_sid_adapter.py:34 DEFAULT_PREFIXES` still has its own prefix list (U6) — to fold with the adapter
  itself (see #3 remaining bits).
- The single-pass default leaves residual collisions unresolved by design (GrIS_v1 parity); the collision rate is
  still reported by `report_and_save`, so the trade-off is visible per run.
- Clustering methods (`recdmon`/`spectral`/`stacked_dmon`) never had the Sinkhorn fix pass, so `--fix_collisions`
  is a no-op for them (their trainer configs don't carry the field).
- A single entry point can run the end-to-end flow (SID -> generative training) reproducibly on one dataset.
