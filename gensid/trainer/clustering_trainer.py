import logging

import torch
from tqdm import tqdm

from gensid.config.trainer_config import ClusteringTrainerConfig
from gensid.eval import check_id_distribution
from gensid.generation.base import collect_indices, report_and_save
from gensid.trainer.base import Trainer

logger = logging.getLogger(__name__)


class ClusteringTrainer(Trainer):
    """Trainer for fit-and-generate SID methods (k-means, recursive clustering).

    These methods need no gradient loop: :meth:`fit` fits the model once on the
    whole dataset (e.g. ``RKMeans``'s residual k-means or ``RecDMoN``'s
    recursive hierarchy) and stores the produced SIDs, then generation simply
    reports the cached indices.

    Args:
        config (ClusteringTrainerConfig): Trainer hyperparameters (learning rate,
            checkpointing, ...). See ``gensid.config.trainer_config.ClusteringTrainerConfig``.
    """

    name = "clustering"

    def __init__(
        self,
        config: ClusteringTrainerConfig,
        model,
        device=None,
        ckpt_dir=None,
        saved_model_dir=None,
        dist_state=None,
    ):
        super().__init__(
            config,
            model,
            device=device,
            ckpt_dir=ckpt_dir,
            saved_model_dir=saved_model_dir,
            dist_state=dist_state,
        )
        self.indices = None

    def _build_optimizer(self):
        if not any(self.model.parameters()):
            self._dummy_param = torch.nn.Parameter(torch.zeros(1))
            return torch.optim.AdamW([self._dummy_param], lr=self.lr)
        return super()._build_optimizer()

    def fit(self, dataset, data_loader=None):
        """Fit the model once and cache the produced SIDs.

        Calls ``model.fit`` then ``model.generate``, computes the SID-quality
        metrics (:func:`gensid.eval.check_id_distribution`) and stores the
        collision rate as the checkpoint-selection criterion.

        Args:
            dataset: Dataset consumed by the model (``model.fit(dataset, ...)``).
            data_loader (optional): Data loader whose ``batch_size`` is forwarded
                to the model. Defaults to None (full-batch).

        Returns:
            tuple: ``(best_loss, best_collision_rate)``.
        """
        batch_size = data_loader.batch_size if data_loader is not None else None

        self.model.fit(dataset, batch_size=batch_size)
        indices = self.model.generate(dataset, batch_size=batch_size)
        self.indices = indices
        metrics = check_id_distribution(indices, self.model.n_codebooks, self.model.codebook_sizes, verbose=True)

        self.best_collision_rate = metrics["collision"]
        self._save_checkpoint(0, collision_rate=self.best_collision_rate, ckpt_file=self.best_collision_ckpt)
        self._set_trained_flag(ckpt_file=self.best_collision_ckpt)
        return self.best_loss, self.best_collision_rate

    def generate_indices(self, dataset, data_loader, output_file):
        """Generate SID indices for the dataset and write the mapping JSON.

        Runs ``model.generate``, maps graph-node-ordered indices back to the
        global item space (:meth:`_to_item_space`) and persists the SID mapping
        via :func:`gensid.generation.base.report_and_save`.
        """

        def _generate(batch):
            emb_data = batch["emb"].to(self.device)
            return self.model.generate(emb_data, batch_size=data_loader.batch_size)

        all_indices, all_indices_str = collect_indices(
            ((None, data) for data in tqdm(data_loader, desc="Clustering Trainer Generating Indices")),
            _generate,
            self.model.n_codebooks,
        )
        report_and_save(all_indices, all_indices_str, self.model.n_codebooks, self.model.codebook_sizes, output_file)
