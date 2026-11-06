from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TrainerConfig:
    """Base for any SID trainer configuration.

    Carries only the training hyperparameters the stock ``Trainer`` consumes; the
    execution context (device, checkpoint directory) is owned by the pipeline and
    passed to the trainer at construction time.

    Args:
        name (str or None, optional): Registered name of the trainer, used as a
            discriminant and for lookups. Defaults to "base".
    """

    name: str = "base"
    lr: float = 1e-3
    learner: str = "AdamW"
    weight_decay: float = 1e-4
    epochs: int = 20000
    eval_step: int = 2000
    compile: bool = False

    # Generation-time SID formatting (shared by every trainer; argparse-controlled
    # via main_gensid.py --prefixes / --unique_prefix / --unclustered_strategy).
    prefixes: list[str] | None = None
    unique_prefix: str = "<z_{}>"
    unclustered_strategy: str = "special"

    def to_kwargs(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "name"}

    @classmethod
    def from_args(cls, args) -> "TrainerConfig":
        return cls(**{k: getattr(args, k, v) for k, v in cls().to_kwargs().items()})


@dataclass(frozen=True)
class RQTrainerConfig(TrainerConfig):
    """Configuration for the RQ-VAE trainer (``RQTrainer``).

    Args:
        name (Literal["rqvae"], optional): Registered name of the trainer, fixed
            to "rqvae". Defaults to "rqvae".

        disable_sk (bool, optional): If True, skip the Sinkhorn tie-breaking in
            the training forward. Defaults to False.

        disable_vq_init (bool, optional): If True, skip the one-pass k-means
            codebook initialization before training. Defaults to False.

        fix_collisions (bool, optional): If True, run the post-training Sinkhorn
            second pass over the generated indices to resolve residual
            collisions (capped by ``max_sk_fix_collision_iters``). If False
            (default), generation is a single nearest-neighbour pass and
            collisions are left unresolved, matching GrIS_v1's behavior.
            Defaults to False.

        max_cluster_level (int or None, optional): Highest cluster level used for
            the edge-reconstruction loss; None defers to the model's codebook
            depth minus one. Defaults to None.

        num_workers (int, optional): Number of dataloader workers for the VQ
            initialization pass. Defaults to 1.

        max_sk_fix_collision_iters (int, optional): Max Sinkhorn iterations
            post-training for SID generation to reduce collisions.
            Defaults to 20.

        save_json (bool, optional): If True, write the SID mapping as the
            integer-code JSON instead of the default parquet artifact.
            Defaults to False.

        seed (int or None, optional): Random seed (reserved). Defaults to None.
    """

    name: Literal["rqvae"] = "rqvae"
    disable_sk: bool = False
    disable_vq_init: bool = False
    fix_collisions: bool = False
    max_cluster_level: int | None = None
    num_workers: int = 1
    max_sk_fix_collision_iters: int = 20
    save_json: bool = False
    seed: int | None = None


@dataclass(frozen=True)
class ClusteringTrainerConfig(TrainerConfig):
    """Configuration for the fit-and-generate trainer (``ClusteringTrainer``).

    Covers the non-gradient SID methods (``rkmeans``, ``hierarchical_rkmeans``,
    ``recdmon``, ``spectral``): the trainer fits the model once on the whole
    dataset and then generates indices from the fitted state.

    Args:
        name (Literal["clustering"], optional): Registered name of the trainer,
            fixed to "clustering". Defaults to "clustering".
    """

    name: Literal["clustering"] = "clustering"


@dataclass(frozen=True)
class RecursiveClusteringTrainerConfig(TrainerConfig):
    """Configuration for the hierarchical-clustering trainer (``RecursiveClusteringTrainer``).

    Extends ``ClusteringTrainerConfig`` for the recursive graph-clustering
    methods (``recdmon``, ``spectral``). On top of the stock fit-and-generate
    flow, it persists the bespoke coarse-to-fine hierarchy artifacts (labels
    JSON, node map, metadata, NPZ) that other codebases consume.

    Args:
        name (Literal["recursive_clustering"], optional): Registered name of the
            trainer, fixed to "recursive_clustering".
            Defaults to "recursive_clustering".
        seed (int or None, optional): Random seed (reserved). Defaults to None.

        save_json (bool, optional): If True, write the SID mapping as the
            token-code JSON instead of the default recdmon parquet artifact.
            Defaults to False.

        all_items_index (str or None, optional): Parquet file with
            ``item_id``/``raw_id`` for every dataset item; used to extend the
            clustering SIDs to OOV items and recover ``raw_id``.
            Defaults to None.
    """

    name: Literal["recursive_clustering"] = "recursive_clustering"
    seed: int | None = None
    save_json: bool = False
    all_items_index: str | None = None


@dataclass(frozen=True)
class StackedDMoNTrainerConfig(TrainerConfig):
    """Configuration for the stacked-DMoN trainer (``StackedDMoNTrainer``).

    Extends ``TrainerConfig`` for the joint-level DMoN trainer, adding the
    gradients-norm clipping knob used in the dataset-wide updates.

    Args:
        name (Literal["stacked_dmon"], optional): Registered name of the
            trainer, fixed to "stacked_dmon".
            Defaults to "stacked_dmon".
        seed (int or None, optional): Random seed (reserved). Defaults to None.
        grad_clip (float or None, optional): Max gradient norm;
            None disables clipping. Defaults to 5.0.

        save_json (bool, optional): If True, write the SID mapping as the
            token-code JSON instead of the default recdmon parquet artifact.
            Defaults to False.

        all_items_index (str or None, optional): Parquet file with
            ``item_id``/``raw_id`` for every dataset item; used to extend the
            clustering SIDs to OOV items and recover ``raw_id``.
            Defaults to None.
    """

    name: Literal["stacked_dmon"] = "stacked_dmon"
    seed: int | None = None
    grad_clip: float | None = 5.0
    save_json: bool = False
    all_items_index: str | None = None
