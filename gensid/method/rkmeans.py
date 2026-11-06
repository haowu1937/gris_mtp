import torch
import torch.nn.functional as F
from tqdm import tqdm

from gensid.config.docstring import auto_docstring
from gensid.config.model_config import HierarchicalRKMeansModelConfig, RKMeansModelConfig
from gensid.method.base import SIDMethod
from gensid.nn.layers import FaissKMeans, TorchKMeans
from gensid.nn.utils import DISTANCE_FUNCTIONS, MEAN_FUNCTIONS


class BaseRKMeans(SIDMethod):
    def __init__(self, config):
        """Shared setup for flat and hierarchical residual k-means methods.

        Reads the per-level cluster counts, distance/mean functions and the
        residual-normalization flags from ``config``, and initializes the
        ``TorchKMeans`` state (``self.k_means``) expected by subclasses.
        """
        super().__init__()

        self.config = config
        self.depth = config.depth
        if isinstance(config.n_clusters, int):
            n_clusters = [config.n_clusters] * config.depth
        else:
            n_clusters = list(config.n_clusters)
            if len(n_clusters) != config.depth:
                raise ValueError("`n_clusters` must be an int or a list of ints of length `depth`")

        self.n_clusters = n_clusters
        self.distance_function = DISTANCE_FUNCTIONS[config.distance_function]
        self.mean_function = MEAN_FUNCTIONS[config.mean_function]
        self.device = torch.device("cpu")
        self.kmeans_backend = config.kmeans_backend
        self.kmeanspp_exact = config.kmeanspp_exact

        self._codebook_sizes = self.n_clusters

        # Normalize residuals when using cosine distance and normalized mean
        self.normalize_residuals = config.distance_function == "cosine" and config.mean_function == "normalized_mean"
        self.norm_input = config.norm_input or self.normalize_residuals

    def _make_kmeans(self, n_clusters: int):
        """Instantiate the configured k-means backend (``TorchKMeans`` or ``FaissKMeans``)."""
        cls = FaissKMeans if self.kmeans_backend == "faiss" else TorchKMeans
        return cls(n_clusters, self.distance_function, self.mean_function, self.device, self.kmeanspp_exact)

    def fit(self, dataset: torch.utils.data.Dataset, batch_size: int | None = None):
        """Fit the method on the dataset.

        Resolves the node features (raw tensor or ``dataset[:]["emb"]``) and
        dispatches to :meth:`fit_mini_batch` or :meth:`_fit` depending on
        ``run_fit_with_mini_batch``.

        Args:
            dataset (torch.utils.data.Dataset or torch.Tensor): Dataset exposing
                an ``"emb"`` key, or a raw ``[num_items, dim]`` tensor.
            batch_size (int or None, optional): Mini-batch size for the
                underlying ``TorchKMeans``. Defaults to None (full-batch fit).
        """
        data = dataset if torch.is_tensor(dataset) else dataset[:]["emb"]
        if self.run_fit_with_mini_batch:
            return self.fit_mini_batch(data, batch_size)
        else:
            return self._fit(data, batch_size)


@auto_docstring
class RKMeans(BaseRKMeans):
    """Flat recursive k-means SID method.

    Fits one ``TorchKMeans`` per level on the (optionally normalized) residuals of
    the previous level. Fitting and generation are driven by the
    :class:`~gensid.trainer.clustering_trainer.ClusteringTrainer`; this model only
    holds the fitted ``k_means`` state and knows how to generate SIDs.

    Args:
        config (RKMeansModelConfig): Hyperparameters of the model. See
            ``gensid.config.model_config.RKMeansModelConfig``.
    """

    name = "rkmeans"
    trainer = "clustering"

    def __init__(self, config: RKMeansModelConfig):
        super().__init__(config)

        # fit mini-batch config
        self.run_fit_with_mini_batch = config.fit_mini_batch
        self.mini_batch_epochs = config.mini_batch_epochs
        self.max_steps = config.max_steps
        self.init_mode = config.init_mode
        self.init_batch_size = config.init_batch_size

        self.k_means = []

    def to(self, device: torch.device | str):
        """Move all fitted k-means to ``device``."""
        self.device = device
        for k_means in self.k_means:
            k_means.to(device)

        return self

    def _fit(self, data: torch.Tensor, batch_size: int | None = None):
        """Full-batch residual k-means fit.

        Fits one ``TorchKMeans`` per depth level on the (optionally normalized)
        residuals of the previous level, appending each to ``self.k_means``.
        """
        residuals = data

        if self.norm_input:
            residuals = F.normalize(residuals)

        for i in tqdm(range(self.depth), desc="RKMeans fit"):
            k_means = self._make_kmeans(self.n_clusters[i])
            indexes = k_means.fit_predict(residuals, batch_size=batch_size)
            self.k_means.append(k_means)
            residuals = residuals - k_means.centroids[indexes].to(data.device)
            if self.normalize_residuals:
                residuals = F.normalize(residuals)

    def fit_mini_batch(self, data: torch.Tensor, batch_size: int):
        """Mini-batch residual k-means fit.

        Like :meth:`_fit`, but each level's ``TorchKMeans`` is trained with
        :meth:`TorchKMeans.fit_mini_batch` instead of a full-batch fit.
        """
        residuals = data

        if self.norm_input:
            residuals = F.normalize(residuals)

        for i in tqdm(range(self.depth), desc="RKMeans fit mini-batch"):
            k_means = self._make_kmeans(self.n_clusters[i])
            init_batch = residuals[: self.init_batch_size] if self.init_batch_size is not None else residuals
            k_means.fit_mini_batch(
                residuals, self.mini_batch_epochs, batch_size, self.init_mode, self.max_steps, init_batch
            )
            indexes = k_means.predict(residuals, batch_size=batch_size)
            self.k_means.append(k_means)
            residuals = residuals - k_means.centroids[indexes].to(data.device)
            if self.normalize_residuals:
                residuals = F.normalize(residuals)

    def generate(self, dataset: torch.utils.data.Dataset, batch_size: int | None = None) -> torch.Tensor:
        """Generate a ``[num_items, depth]`` SID matrix.

        Each level assigns the residual of the previous level to its nearest
        centroid, then subtracts it to produce the next residual.

        Args:
            dataset (torch.utils.data.Dataset or torch.Tensor): Dataset exposing
                an ``"emb"`` key, or a raw ``[num_items, dim]`` tensor.
            batch_size (int or None, optional): Mini-batch size for the centroid
                predictions. Defaults to None.

        Returns:
            torch.Tensor: ``[num_items, depth]`` int64 SIDs.
        """
        data = dataset if torch.is_tensor(dataset) else dataset[:]["emb"]
        residuals = data
        indexes = torch.zeros((data.shape[0], self.depth), dtype=torch.int64, device=data.device)
        for i in range(self.depth):
            k_means = self.k_means[i]
            indexes[:, i] = k_means.predict(residuals, batch_size)
            residuals = residuals - k_means.centroids[indexes[:, i]].to(data.device)

        return indexes.to(self.device)

    def state_dict(self):
        """Return the fitted centroids, keyed by depth level."""
        return {i: k.centroids for i, k in enumerate(self.k_means)}

    def save(self, path: str):
        """Persist the fitted centroids with ``torch.save``."""
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Restore centroids saved by :meth:`save`.

        Raises:
            AssertionError: If the checkpoint depth differs from ``self.depth``.
        """
        k_means_export = torch.load(path, weights_only=False)
        assert self.depth == len(k_means_export), "Cannot load RKMeans checkpoint with different depth"
        self.k_means = [None] * len(k_means_export)
        for depth, centroids in k_means_export.items():
            k_means = self._make_kmeans(centroids.shape[0])
            k_means.centroids = centroids
            self.k_means[depth] = k_means

    def load_state_dict(self, state_dict: dict, **kwargs):
        """Rebuild ``self.k_means`` from a ``{depth: centroids}`` dict."""
        self.k_means = [None] * len(state_dict)
        for depth, centroids in state_dict.items():
            k_means = self._make_kmeans(centroids.shape[0])
            k_means.centroids = centroids
            self.k_means[depth] = k_means


@auto_docstring
class HierarchicalRKMeans(BaseRKMeans):
    """Hierarchical (recursive) k-means SID method.

    Like ``RKMeans``, but clusters each prefix branch independently: a separate
    ``TorchKMeans`` is fitted for every node of the recursion tree, keyed by its
    (depth, ...) prefix. Fitting and generation are driven by the
    :class:`~gensid.trainer.clustering_trainer.ClusteringTrainer`.

    Args:
        config (HierarchicalRKMeansModelConfig): Hyperparameters of the model. See
            ``gensid.config.model_config.HierarchicalRKMeansModelConfig``.
    """

    name = "hierarchical_rkmeans"
    trainer = "clustering"

    def __init__(self, config: HierarchicalRKMeansModelConfig):
        super().__init__(config)

        self.k_means = {}
        self.run_fit_with_mini_batch = False

    def _fit(self, data: torch.Tensor, batch_size: int | None = None):
        """Full-batch recursive residual k-means fit.

        Fits a separate ``TorchKMeans`` per node of the recursion tree, keyed by
        its ``(depth, ...)`` prefix. Branches with fewer unique values than
        requested clusters use the unique values themselves as centroids.
        """
        if self.norm_input:
            data = F.normalize(data)

        prefixes = {(): data}

        for depth in range(self.depth):
            n_clusters = self.n_clusters[depth]
            next_prefixes = {}

            for prefix, prefix_data in prefixes.items():
                # If there are fewer unique values in the data than there should be clusters, set the centroids to the
                # unique values
                unique_values = torch.unique(prefix_data, dim=0)
                n_unique_values = unique_values.shape[0]
                if n_unique_values < n_clusters:
                    k_means = self._make_kmeans(n_unique_values)
                    k_means.centroids = unique_values.to(self.device)
                    indexes = k_means.predict(prefix_data, batch_size)
                else:
                    k_means = self._make_kmeans(n_clusters)
                    indexes = k_means.fit_predict(prefix_data, batch_size=batch_size)

                self.k_means[prefix] = k_means
                for index in torch.unique(indexes):
                    selected_centroid = k_means.centroids[index].to(prefix_data.device)
                    data_indexes = (indexes == index).to(prefix_data.device)
                    residuals = prefix_data[data_indexes] - selected_centroid
                    if self.normalize_residuals:
                        residuals = F.normalize(residuals)
                    next_prefixes[prefix + (index.item(),)] = residuals

            prefixes = next_prefixes

    def fit_mini_batch(self, *args, **kwargs):
        """Not supported: hierarchical recursion has no mini-batch fit."""
        raise NotImplementedError("Hierarchical RQ-KMeans does not support mini-batch fit")

    def generate(self, dataset: torch.utils.data.Dataset, batch_size: int | None = None) -> torch.Tensor:
        """Generate a ``[num_items, depth]`` SID matrix.

        Walks the recursion tree per item, assigning each prefix's residual to
        the nearest centroid of the matching ``self.k_means[prefix]``.

        Args:
            dataset (torch.utils.data.Dataset or torch.Tensor): Dataset exposing
                an ``"emb"`` key, or a raw ``[num_items, dim]`` tensor.
            batch_size (int or None, optional): Mini-batch size for the centroid
                predictions. Defaults to None.

        Returns:
            torch.Tensor: ``[num_items, depth]`` int64 SIDs.
        """
        data = dataset if torch.is_tensor(dataset) else dataset[:]["emb"]
        indexes = torch.zeros((data.shape[0], self.depth), dtype=torch.int64, device=data.device)
        for i in range(data.shape[0]):
            prefix = ()
            residual = torch.unsqueeze(data[i], dim=0).to(self.device)
            for depth in range(self.depth):
                k_means = self.k_means[prefix]
                index = k_means.predict(residual, batch_size).item()
                prefix = prefix + (index,)
                residual = residual - k_means.centroids[index]
                indexes[i, depth] = index

        return indexes

    def to(self, device: torch.device | str):
        """Move all per-prefix k-means to ``device``."""
        self.device = device
        for sub_k_means in self.k_means.values():
            sub_k_means.to(device)

        return self

    def state_dict(self):
        """Return the fitted centroids, keyed by prefix."""
        return {k: v.centroids for k, v in self.k_means.items()}

    def save(self, path: str):
        """Persist the fitted centroids with ``torch.save``."""
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Restore centroids saved by :meth:`save`."""
        k_means_export = torch.load(path, weights_only=False)
        for key, centroids in k_means_export.items():
            k_means = self._make_kmeans(centroids.shape[0])
            k_means.centroids = centroids
            self.k_means[key] = k_means

    def load_state_dict(self, state_dict: dict, **kwargs):
        """Rebuild ``self.k_means`` from a ``{prefix: centroids}`` dict."""
        for key, centroids in state_dict.items():
            k_means = self._make_kmeans(centroids.shape[0])
            k_means.centroids = centroids
            self.k_means[key] = k_means
