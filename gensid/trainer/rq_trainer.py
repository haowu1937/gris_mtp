import logging
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from gensid.eval import id_collision_rate
from gensid.generation.base import collect_indices, report_and_save
from gensid.generation.utils import save_parquet_semantic_ids
from gensid.nn.utils import custom_constrained_km
from gensid.trainer.base import Trainer
from gensid.utils import check_collision, get_collision_item, set_color

logger = logging.getLogger(__name__)


class RQTrainer(Trainer):
    name = "rqvae"

    def __init__(self, config, model, **kwargs):
        super().__init__(config, model, **kwargs)
        self.use_sk = not config.disable_sk
        self._use_sk = self.use_sk
        self.disable_vq_init = config.disable_vq_init

        # TODO: self.labels is re-assigned later. Why is this dictionary needed?
        self.labels = {"0": [], "1": [], "2": [], "3": [], "4": [], "5": []}

        self.max_cluster_level = config.max_cluster_level
        if self.max_cluster_level is None:
            self.max_cluster_level = (
                len(self.model.codebook_sizes) - 1
            )  # last one reserved for edge reconstruction loss

    def _prepare_batch(self, data, train=False):
        """Move a dataloader batch to the device.

        PyG batches (``NeighborLoader``, detected via ``n_id``) are sliced to the
        seed nodes and kept as ``graph_data`` for the online graph encoder
        (GRQVAE). ``train=True`` also moves every tensor feature to the device
        (and keeps ``index``/``target`` for the training losses), otherwise only
        ``emb`` is moved.
        """
        if hasattr(data, "n_id"):
            data = data.to(self.device, non_blocking=True)
            batch = {"emb": data.x[: data.batch_size], "graph_data": data}
            if train:
                batch["index"] = data.n_id[: data.batch_size]
                batch["target"] = data.x[: data.batch_size]
            return batch
        if train:
            for feat, data_feat in data.items():
                if isinstance(data_feat, torch.Tensor):
                    data[feat] = data_feat.to(self.device, non_blocking=True)
            return data
        data["emb"] = data["emb"].to(self.device, non_blocking=True)
        return data

    def _generate(self, batch):
        return self.model.generate(self._prepare_batch(batch), self.labels, use_sk=self._use_sk)

    def _load_collision_batch(self, dataset, data_loader, items):
        """Fetch the (unprepared) batch for the given item ids (collision-fix pass).

        Generation always runs on a plain dataloader over the eval dataset, so the
        collision-fixing pass fetches samples straight from the dataset.
        """
        return dataset[items]

    @torch.no_grad()
    def vq_initialization(self, dataset):
        self.model.eval()
        init_loader = DataLoader(
            dataset,
            num_workers=self.config.num_workers,
            batch_size=len(dataset),
            shuffle=True,
            pin_memory=True,
        )

        logger.debug(f"Dataloader size: {len(init_loader)}")

        iter_data = tqdm(
            init_loader,
            total=len(init_loader),
            ncols=100,
            desc=set_color("Initialization of vq", "pink"),
            disable=not sys.stderr.isatty(),
        )
        # Train
        for data in iter_data:
            emb_data = data["emb"]
            emb_data = emb_data.to(self.device, non_blocking=True)

            self.model.vq_initialization(emb_data, self.use_sk)

    def _train_epoch(self, train_data, epoch_idx):
        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        total_cf_loss = 0
        total_quant_loss = 0
        used_codebooks = [set() for _ in self.model.rq.vq_layers]

        iter_data = tqdm(
            train_data,
            total=len(train_data),
            ncols=100,
            desc=set_color(f"Train {epoch_idx}", "pink"),
            disable=not sys.stderr.isatty(),
        )

        self.labels = None
        for data in iter_data:
            # if self.plain_reconstruction:
            #     data, emb_idx, recon_tgt = _data[0], _data[1], _data[2]
            #     data = data.to(self.device, non_blocking=True)
            #     recon_tgt = recon_tgt.to(self.device, non_blocking=True)
            # elif self.args.spectral_dim > 0:
            #     # data, emb_idx, recon_tgt, spectral_vectors = (
            #     #     _data[0],
            #     #     _data[1],
            #     #     _data[2],
            #     #     _data[3],
            #     # )
            #     data, emb_idx, recon_tgt, spectral_vectors = _data
            #     data = data.to(self.device, non_blocking=True)
            #     recon_tgt = recon_tgt.to(self.device, non_blocking=True)
            #     spectral_vectors = spectral_vectors.to(self.device, non_blocking=True)
            # elif self.args.use_edge_reconstruction_loss and self.args.cluster_labels_path is not None:
            #     # data, emb_idx, recon_tgt, cluster_labels, edges_weights = (
            #     #     _data[0],
            #     #     _data[1],
            #     #     _data[2],
            #     #     _data[3],
            #     #     _data[4],
            #     # )
            #     data, emb_idx, recon_tgt, cluster_labels, edges_weights = _data
            #     data = data.to(self.device, non_blocking=True)
            #     recon_tgt = recon_tgt.to(self.device, non_blocking=True)
            #     cluster_labels = cluster_labels[:, : self.max_cluster_level]
            #     cluster_labels = cluster_labels.to(self.device, non_blocking=True)
            # elif self.args.use_edge_reconstruction_loss:
            #     data, emb_idx, _, edges_weights = _data
            #     data = data.to(self.device, non_blocking=True)
            #     recon_tgt = data
            # else:
            #     data, emb_idx = _data[0], _data[1]
            #     data = data.to(self.device, non_blocking=True)
            #     recon_tgt = data

            self.optimizer.zero_grad()

            data = self._prepare_batch(data, train=True)
            if "cluster_labels" in data:
                data["cluster_labels"] = data["cluster_labels"][:, : self.max_cluster_level]
            out, rq_loss, indices, dense_out, dense_out_codebooks, x_res_list = self.ddp_model(
                data, self.labels, use_sk=self.use_sk
            )

            for codebook_idx in range(indices.shape[-1]):
                used_codebooks[codebook_idx].update(indices[:, codebook_idx].detach().cpu().tolist())

            loss, cf_loss, loss_recon, quant_loss = self.model.compute_loss(
                out,
                rq_loss,
                data["index"],
                dense_out,
                dense_out_codebooks=dense_out_codebooks,
                x_res_list=x_res_list,
                xs=data["target"],
                edges_weights=data.get("edges_weights", None),
                spectral_vectors=data.get("spectral_vectors", None),
                cluster_labels=data.get("cluster_labels", None),
            )
            self._check_nan(loss)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_cf_loss += cf_loss.item() if cf_loss != 0 else cf_loss
            total_quant_loss += quant_loss.item()

        unused_codebooks = self._calculate_unused_codebooks(used_codebooks)

        train_info = {
            "reconstruction loss": total_recon_loss,
            "cf loss": total_cf_loss,
            "quantization loss": quant_loss.item(),
            "unused codebooks": unused_codebooks,
        }
        return total_loss, train_info

    @torch.no_grad()
    def _valid_epoch(self, valid_data):
        self.model.eval()

        iter_data = tqdm(
            valid_data,
            total=len(valid_data),
            ncols=100,
            desc=set_color("Evaluate   ", "pink"),
            disable=not sys.stderr.isatty(),
        )

        indices_set = set()
        num_sample = 0
        self.labels = None
        for data in iter_data:
            data = self._prepare_batch(data)
            num_sample += len(data["emb"])
            indices = self.model.generate(data, self.labels)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

        collision_rate = id_collision_rate(indices_set, num_sample)

        return collision_rate

    def fit(self, dataset, data):
        if not self.disable_vq_init:
            self.vq_initialization(dataset)
        else:
            logger.info(set_color("Skip vq initialization", "yellow") + ": config.disable_vq_init=True")

        result = super().fit(dataset, data)
        if not (self.is_distributed and self.rank != 0):
            self._save_fused_cache(dataset)
        return result

    @torch.no_grad()
    def _save_fused_cache(self, dataset):
        """Persist the post-training graph fusion for online-GNN datasets (GRQVAE).

        The online-GNN dataset (``GRQVAEFusionEmbDataset``) exposes a
        ``save_fused_cache(model, device)`` hook that fuses the whole training
        graph with the trained ``graph_encoder``; generation reloads that cache
        and applies the eval restore logic. Offline/plain datasets are no-ops.
        """
        save_fn = getattr(dataset, "save_fused_cache", None)
        if save_fn is None:
            return
        logger.info("Saving post-training fused embeddings cache for generation")
        save_fn(self.model, self.device)

    @torch.no_grad()
    def generate_indices(self, dataset, data_loader, output_file):
        """SID generation (moved from RQVAEBuilder._run_vae_generation).

        Always runs a single nearest-neighbour pass (``use_sk=False``), matching
        GrIS_v1. When ``config.fix_collisions`` is set, a second Sinkhorn pass
        re-runs the colliding rows in place, resolving residual collisions for at
        most ``max_sk_fix_collision_iters`` iterations. Both passes fold their
        batches through ``collect_indices``.
        """
        self.model.eval()

        # Extract embeddings from VQ layers
        embs = [layer.embedding.weight for layer in self.model.rq.vq_layers]

        self.labels = {str(i): [] for i in range(self.model.n_codebooks)}
        for idx, emb in enumerate(embs):
            _, label = custom_constrained_km(
                emb, n_clusters=self.model.config.n_clusters, max_size_multiplier=6, seed=self.config.seed
            )
            self.labels[str(idx)] = label

        # First pass: Generate indices without Sinkhorn (fast)
        self._use_sk = False
        all_indices, all_indices_str = collect_indices(
            ((None, data) for data in tqdm(data_loader, desc="Generating Indices (Pass 1)")),
            self._generate,
            self.model.n_codebooks,
            prefixes=self.config.prefixes,
        )

        if self.config.fix_collisions:
            # Second pass: Resolve collisions using Sinkhorn for at most `max_sk_fix_collision_iters`
            # Reset SK epsilon for layers except last
            for vq in self.model.rq.vq_layers[:-1]:
                vq.sk_epsilon = 0.0

            if self.model.rq.vq_layers[-1].sk_epsilon <= 0.0:
                self.model.rq.vq_layers[-1].sk_epsilon = 0.003

            self._use_sk = True
            fix_collisions_iter = tqdm(
                range(self.config.max_sk_fix_collision_iters),
                desc="2nd Phase Fixing Collisions",
                disable=not sys.stdout.isatty(),
            )
            for _ in fix_collisions_iter:
                if check_collision(all_indices_str):
                    break
                collision_item_groups = get_collision_item(all_indices_str)
                if not collision_item_groups:
                    break

                batches = [
                    (collision_items, self._load_collision_batch(dataset, data_loader, collision_items))
                    for collision_items in collision_item_groups
                ]
                all_indices, all_indices_str = collect_indices(
                    batches,
                    self._generate,
                    self.model.n_codebooks,
                    all_indices,
                    all_indices_str,
                    prefixes=self.config.prefixes,
                )

        if self.config.save_json:
            report_and_save(
                all_indices, all_indices_str, self.model.n_codebooks, self.model.codebook_sizes, output_file
            )
        else:
            save_parquet_semantic_ids(all_indices, output_file, embeddings_path=getattr(dataset, "data_path", None))
