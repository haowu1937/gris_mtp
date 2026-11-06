import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from gensid.eval import id_collision_rate
from gensid.generation.base import collect_indices, report_and_save
from gensid.generation.utils import (
    build_c2f_fc2_labels,
    build_codes,
    build_recdmon_semantic_ids,
    process_cluster_label_levels,
    project_labels_back_to_item_space,
)
from gensid.trainer.base import Trainer


class StackedDMoNTrainer(Trainer):
    """Trainer for the stacked DMoN SID method (``stacked_dmon``).

    Runs a single dataset-wide forward per step (no batching): ``_train_epoch``
    accumulates the per-level DMoN losses and reports each level's metrics;
    ``_valid_epoch`` scores the coarse-to-fine indices by collision rate;
    ``generate_indices`` persists both the stock SID mapping JSON and the
    bespoke coarse-to-fine hierarchy artifacts.

    Args:
        config (StackedDMoNTrainerConfig): Trainer hyperparameters. See
            ``gensid.config.trainer_config.StackedDMoNTrainerConfig``.
    """

    name = "stacked_dmon"

    def _train_epoch(self, train_data, epoch_idx):
        """Run one dataset-wide DMoN step and return the scalar loss + per-level metrics.

        Args:
            train_data: Iterable/loader whose ``dataset`` attribute the model
                consumes (the model forward runs on the whole graph).
            epoch_idx (int): Epoch index (informational only).

        Returns:
            tuple[float, dict]: the mean scalar loss and a dict keyed like
            ``level<d>_{loss,modularity,balance,entropy}`` for every level.
        """
        self.model.train()

        dataset = train_data.dataset.to(self.device)

        self.optimizer.zero_grad()
        _, levels, loss = self.ddp_model(dataset)

        self._check_nan(loss)
        loss.backward()
        if self.config.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()

        train_info = {}
        for lvl in levels:
            for key in ["loss", "modularity", "balance", "entropy"]:
                train_info[f"level{lvl['level']}_{key}"] = lvl[key]

        return loss.item(), train_info

    @torch.no_grad()
    def _valid_epoch(self, valid_data):
        """Score the generated coarse-to-fine SIDs by collision rate.

        Args:
            valid_data: Loader whose ``dataset`` attribute the model consumes.

        Returns:
            float: the fraction of duplicated SIDs (``0.0`` if collision-free).
        """
        self.model.eval()

        indices, _ = self.model.generate(valid_data.dataset.to(self.device))
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        indices_set = set(["-".join(index.astype(str)) for index in indices])

        collision_rate = id_collision_rate(indices_set, len(valid_data.dataset))

        return collision_rate

    @torch.no_grad()
    def generate_indices(self, dataset, data_loader, output_file):
        """Generate indices and persist the bespoke hierarchy artifacts.

        Writes the coarse-to-fine label/node-map/metadata/NPZ artifacts under
        ``mincut_runs/<name>_seed<seed>`` via :meth:`save_outputs` (which also
        overwrites ``output_file`` with the final codes JSON).
        """
        self.model.eval()

        # TODO: make this process cleaner
        # TODO: generalize better the output dir as an argparse parameter
        output_dir = Path(os.path.dirname(output_file)) / "mincut_runs" / f"{self.model.name}_seed{self.config.seed}"
        indices, artifacts = self.model.generate(dataset.to(self.device), batch_size=data_loader.batch_size)

        num_items = int(getattr(dataset, "num_items", dataset.x_graph.shape[0]))
        item_labels_coarse_to_fine = project_labels_back_to_item_space(
            graph_labels=indices.numpy(),
            item_ids=dataset.item_ids,
            num_items=num_items,
        )

        all_indices, all_indices_str = collect_indices(
            [(None, dataset)], lambda _: torch.from_numpy(item_labels_coarse_to_fine), self.model.n_codebooks
        )
        report_and_save(all_indices, all_indices_str, self.model.n_codebooks, self.model.codebook_sizes, output_file)

        # overwrites output_file saved by report_and_save
        self.save_outputs(
            output_file,
            output_dir,
            indices.numpy(),
            item_labels_coarse_to_fine,
            num_items,
            dataset,
            dataset.item_ids,
            dataset.x_graph,
            dataset.edge_index,
            artifacts=artifacts,
            save_json=self.config.save_json,
        )

    def save_outputs(
        self,
        output_file: str | Path,
        output_dir: str | Path,
        graph_labels_coarse_to_fine: np.ndarray,
        item_labels_coarse_to_fine: np.ndarray,
        num_items: int,
        dataset,
        item_ids: np.ndarray,
        raw_embeddings: np.ndarray,
        edge_index: torch.Tensor,
        output_prefix: str = "graph_hierarchy",
        history: list[dict[str, float]] | None = None,
        artifacts: dict[str, np.ndarray] | None = None,
        save_json: bool = False,
    ) -> None:
        """Persist the coarse-to-fine hierarchy artifacts for downstream consumers.

        Writes ``<prefix>_labels.json`` (item- and graph-space labels per level),
        ``<prefix>_node_map.csv``, ``<prefix>_meta.json`` and
        ``<prefix>_artifacts.npz`` under ``output_dir``.

        Args:
            output_dir (str or Path): Directory to write the artifacts into.
            graph_labels_coarse_to_fine np.ndarray: Graph-node-order label
                vectors, coarsest level first [num_nodes, depth].
            dataset: Dataset the labels were computed on (provides ``num_items``,
                ``config``, ``embeddings_path``, ``interactions_path``).
            item_ids (np.ndarray): Global item id of every graph node.
            raw_embeddings (np.ndarray): Graph-node embeddings (unused; kept for
                signature parity with the recursive trainer).
            edge_index (torch.Tensor): ``[2, num_edges]`` graph adjacency.
            output_prefix (str, optional): Artifact file prefix.
                Defaults to "graph_hierarchy".
            history (list[dict[str, float]] or None, optional): Per-split training
                history to persist as JSON. Defaults to None.
            artifacts (dict[str, np.ndarray] or None, optional): Extra arrays to
                bundle into the NPZ. Defaults to None.
            save_json (bool, optional): If True, overwrite ``output_file`` with
                the token-code JSON mapping; otherwise write the default recdmon
                parquet artifact. Defaults to False.
        """
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = output_prefix

        labels_json = build_c2f_fc2_labels(graph_labels_coarse_to_fine.T, item_labels_coarse_to_fine.T)

        labels_path = output_dir / f"{prefix}_labels.json"
        with open(labels_path, "w") as f:
            json.dump(labels_json, f)

        node_map_path = output_dir / f"{prefix}_node_map.csv"
        pd.DataFrame(
            {
                "node_id": np.arange(len(item_ids), dtype=np.int64),
                "item_id": item_ids.astype(np.int64),
            }
        ).to_csv(node_map_path, index=False)

        metadata = {
            "method": self.model.name,
            "embeddings_path": str(dataset.embeddings_path),
            "interactions_path": str(dataset.interactions_path),
            "graph_type": dataset.config.graph_building.graph_type,
            "graph_building_kwargs": dataset.config.graph_building.to_kwargs(),
            "num_items": int(num_items),
            "num_graph_nodes_used": len(item_ids),
            "num_edges_used": int(edge_index.size(1)),
            "make_directed": dataset.config.graph_building.make_directed,
            "skip_n_last": dataset.config.skip_n_last,
            "graph_source": "graph_utils.InteractionGraphBuilder",
        }
        metadata.update(
            {
                "cluster_sizes_fine_to_coarse": self.model.codebook_sizes[::-1],
                "cluster_sizes_coarse_to_fine": self.model.codebook_sizes,
            }
        )

        metadata_path = output_dir / f"{prefix}_meta.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        if history is not None:
            history_path = output_dir / f"{prefix}_history.json"
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)
        else:
            history_path = None

        artifacts = dict(artifacts)
        artifacts["item_ids_graph_order"] = item_ids.astype(np.int64)
        for i, labels in enumerate(graph_labels_coarse_to_fine.T, start=1):
            artifacts[f"coarse_to_fine_level{i}_graph_labels"] = labels.astype(np.int64)
        np.savez_compressed(output_dir / f"{prefix}_artifacts.npz", **artifacts)

        print(f"Saved labels to:   {labels_path}")
        print(f"Saved metadata to: {metadata_path}")
        print(f"Saved node map to: {node_map_path}")
        print(f"Saved artifacts to:{output_dir / f'{prefix}_artifacts.npz'}")
        if history_path is not None:
            print(f"Saved history to:  {history_path}")
        print("\nFor RVQ prefix alignment, consume labels_json['coarse_to_fine_item_labels'].")
        print("  level_1 -> coarsest supervision for prefix-1")
        print("  level_2 -> finer supervision for prefix-2, etc.")

        if save_json:
            level_names, labels = process_cluster_label_levels(labels_json)
            codes = build_codes(
                labels=labels,
                unclustered_strategy=self.config.unclustered_strategy,
                append_unique_leaf=True,
                unique_prefix=self.config.unique_prefix,
                prefixes=self.config.prefixes,
            )
            with open(output_file, "w") as f:
                json.dump(codes, f)
        else:
            build_recdmon_semantic_ids(
                item_labels_coarse_to_fine, output_file, all_items_index=self.config.all_items_index
            )
