import logging

import torch

from gensid.config.model_config import OPQModelConfig, PQModelConfig
from gensid.method.base import SIDMethod
from gensid.nn.pq import make_opq, make_pq

logger = logging.getLogger(__name__)


class _PQBaseMethod(SIDMethod):
    """Shared fit/generate/checkpoint logic for the PQ and OPQ SID methods.

    Both methods quantize each item embedding into ``n_subquantizers`` sub-codes
    (one per codebook); the SID is the ``n_subquantizers`` concatenated codes, so
    ``n_codebooks == n_subquantizers`` and every codebook holds ``2**n_bits``
    entries. Fitting and generation are driven by the
    :class:`~gensid.trainer.clustering_trainer.ClusteringTrainer`.
    """

    trainer = "clustering"

    #: Whether the quantizer carries an OPQ rotation ``A`` (checkpoint load validation).
    _has_rotation = False

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_dim = config.in_dim
        self.n_subquantizers = config.n_subquantizers
        self.n_bits = config.n_bits
        self._codebook_sizes = [1 << config.n_bits] * config.n_subquantizers
        self.device = torch.device("cpu")
        self.quantizer = None

    def _make_quantizer(self):
        raise NotImplementedError

    def fit(self, dataset, batch_size=None):
        """Fit the quantizer on the dataset embeddings."""
        data = dataset if torch.is_tensor(dataset) else dataset[:]["emb"]
        self.quantizer = self._make_quantizer()
        self.quantizer.fit(data, batch_size)
        return self

    def generate(self, dataset, batch_size=None) -> torch.Tensor:
        """Generate a ``[num_items, n_subquantizers]`` SID matrix (one sub-code per codebook)."""
        if self.quantizer is None:
            raise ValueError("Model not fitted yet.")
        data = dataset if torch.is_tensor(dataset) else dataset[:]["emb"]
        return self.quantizer.encode(data, batch_size)

    def to(self, device: torch.device | str):
        """Move the fitted quantizer (torch backend) to ``device``."""
        self.device = device
        if self.quantizer is not None:
            self.quantizer.to(device)
        return self

    def state_dict(self):
        """Return the quantizer state (centroids [+ rotation]) for checkpointing.

        Besides the tensors, the method ``name`` and ``quantizer_backend`` are
        persisted so a checkpoint can be validated at load time (and, in the
        future, routed to the right path for the unified-scripts work).
        """
        if self.quantizer is None:
            return {}
        state = {
            "name": self.name,
            "quantizer_backend": self.config.quantizer_backend,
            "centroids": self.quantizer.centroids.detach().cpu(),
        }
        rotation = getattr(self.quantizer, "A", None)
        if rotation is not None:
            state["A"] = rotation.detach().cpu()
        return state

    def load_state_dict(self, state_dict: dict, **kwargs):
        """Rebuild the quantizer from a saved state dict, validating the load.

        Raises a clear error when the checkpoint comes from a different method
        (e.g. a PQ checkpoint loaded into an OPQ model), from the legacy
        pre-restructure ``faiss_index`` format, or is missing the required
        tensors. A ``quantizer_backend`` mismatch only warns: the tensor state is
        backend-agnostic, so faiss-trained checkpoints load into torch models and
        vice-versa.
        """
        if "faiss_index" in state_dict and "centroids" not in state_dict:
            raise ValueError(
                "checkpoint uses the legacy faiss_index format (pre-restructure); "
                "regenerate it with the current code (main_gensid.py --sid_model="
                f"{self.name})"
            )
        if "centroids" not in state_dict:
            raise ValueError(f"checkpoint state_dict is missing 'centroids'; got keys {sorted(state_dict)}")
        saved_name = state_dict.get("name")
        if saved_name is not None and saved_name != self.name:
            raise ValueError(
                f"checkpoint was produced by SID method '{saved_name}' but the model is "
                f"'{self.name}': cannot load it; regenerate the checkpoint with "
                f"--sid_model={saved_name}"
            )
        saved_backend = state_dict.get("quantizer_backend")
        if saved_backend is not None and saved_backend != self.config.quantizer_backend:
            logger.warning(
                "checkpoint quantizer_backend='%s' differs from model backend='%s'; "
                "cross-backend load works (tensor state) but verify the intended backend",
                saved_backend,
                self.config.quantizer_backend,
            )
        has_rotation = "A" in state_dict
        if has_rotation and not self._has_rotation:
            raise ValueError(
                f"checkpoint contains an OPQ rotation 'A' but the model is '{self.name}', "
                "which has no rotation; load it with --sid_model=opq"
            )
        if not has_rotation and self._has_rotation:
            raise ValueError(
                "checkpoint for an OPQ model is missing the 'A' rotation (got a PQ "
                "checkpoint?); regenerate it with --sid_model=opq"
            )
        self.quantizer = self._make_quantizer()
        if has_rotation:
            self.quantizer.set_state(state_dict["A"], state_dict["centroids"])
        else:
            self.quantizer.set_centroids(state_dict["centroids"])
        return self

    def compute_loss(self):
        """No gradient loss: clustering methods report a constant zero."""
        return torch.tensor(0.0)


class PQMethod(_PQBaseMethod):
    """Flat product-quantizer SID method (``name = "pq"``).

    Fits one ``TorchPQ``/``FaissPQ`` on the item embeddings and encodes each
    item into ``n_subquantizers`` sub-codes.

    Args:
        config (PQModelConfig): Hyperparameters of the model. See
            ``gensid.config.model_config.PQModelConfig``.
    """

    name = "pq"

    def __init__(self, config: PQModelConfig):
        super().__init__(config)
        self.kmeans_iters = config.kmeans_iters

    def _make_quantizer(self):
        kwargs = {}
        if self.config.quantizer_backend == "torch":
            kwargs["kmeans_iters"] = self.kmeans_iters
        return make_pq(
            self.config.quantizer_backend, self.in_dim, self.n_subquantizers, self.n_bits, device=self.device, **kwargs
        )


class OPQMethod(_PQBaseMethod):
    """Optimized product-quantizer SID method (``name = "opq"``).

    Learns an orthonormal rotation on top of the product quantizer (``TorchOPQ``
    or ``FaissOPQ``) to reduce quantization distortion.

    Args:
        config (OPQModelConfig): Hyperparameters of the model. See
            ``gensid.config.model_config.OPQModelConfig``.
    """

    name = "opq"
    _has_rotation = True

    def __init__(self, config: OPQModelConfig):
        super().__init__(config)
        self.niter = config.niter
        self.kmeans_iters = config.kmeans_iters

    def _make_quantizer(self):
        kwargs = {"niter": self.niter}
        if self.config.quantizer_backend == "torch":
            kwargs["kmeans_iters"] = self.kmeans_iters
        return make_opq(
            self.config.quantizer_backend, self.in_dim, self.n_subquantizers, self.n_bits, device=self.device, **kwargs
        )


__all__ = ["PQMethod", "OPQMethod"]
