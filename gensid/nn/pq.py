import logging

import numpy as np
import torch
from tqdm import tqdm

from gensid.nn.layers import TorchKMeans, _get_faiss
from gensid.nn.utils import MEAN_FUNCTIONS

logger = logging.getLogger(__name__)


DEFAULT_KMEANS_ITERS = 25  # matches faiss ``Clustering`` default ``niter``
DEFAULT_OPQ_NITER = 25  # matches faiss ``OPQMatrix`` default ``niter``


def _unpack_codes(codes: np.ndarray, M: int, nbits: int) -> np.ndarray:
    """Decode faiss-packed PQ codes (``(N, code_size)`` uint8) into ``(N, M)`` sub-codes.

    faiss stores sub-code ``j`` in bits ``[j*nbits, (j+1)*nbits)`` of the byte string,
    least-significant bit first within each byte (``BitstringWriter`` layout).
    """
    total_bits = M * nbits
    bits = np.unpackbits(codes, axis=1, bitorder="little")[:, :total_bits]
    return bits.reshape(codes.shape[0], M, nbits).dot(1 << np.arange(nbits)).astype(np.int64)


def _pack_codes(codes: np.ndarray, nbits: int) -> np.ndarray:
    """Pack ``(N, M)`` sub-codes into faiss-layout ``(N, code_size)`` uint8 bytes.

    Inverse of :func:`_unpack_codes`: sub-code ``j`` goes into bits ``[j*nbits, (j+1)*nbits)``,
    LSB-first within each byte.
    """
    N, M = codes.shape
    total_bits = M * nbits
    bits = np.zeros((N, total_bits), dtype=np.uint8)
    for b in range(nbits):
        bits[:, b::nbits] = (codes >> b) & 1
    pad = (8 - total_bits % 8) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros((N, pad), dtype=np.uint8)], axis=1)
    return np.packbits(bits, axis=1, bitorder="little")


def _subspace_distances(x_view: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Squared-L2 distances from each sub-vector to its subspace's centroids.

    Args:
        x_view (torch.Tensor): ``(N, M, dsub)`` sub-vector view of the data.
        centroids (torch.Tensor): ``(M, ksub, dsub)`` codebook.

    Returns:
        torch.Tensor: ``(N, M, ksub)`` squared distances.
    """
    return (
        torch.einsum("nmd,nmd->nm", x_view, x_view).unsqueeze(-1)
        + torch.einsum("mkd,mkd->mk", centroids, centroids).unsqueeze(0)
        - 2 * torch.einsum("nmd,mkd->nmk", x_view, centroids)
    )


def _orthogonal_procrustes(x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Orthogonal ``A`` minimizing ``||x @ A.T - q||_F`` (the OPQ rotation update).

    Solves the orthogonal Procrustes problem via the SVD of ``q.T @ x`` and returns
    ``u @ vt``, matching faiss's ``OPQMatrix`` update (its ``sgemm``/``sgesvd``
    sequence computes exactly this). This is the transpose of the naive
    ``svd(x.T @ q) -> u @ vt`` formula and the direction matters: the wrong sign
    makes the rotation *increase* quantization error.

    Args:
        x (torch.Tensor): ``(N, D)`` original vectors.
        q (torch.Tensor): ``(N, D)`` PQ reconstruction in rotated space.

    Returns:
        torch.Tensor: ``(D, D)`` orthonormal rotation ``A`` (transform is ``x @ A.T``).
    """
    u, _, vt = torch.linalg.svd(q.T @ x, full_matrices=False)
    return u @ vt


def _encode_by_argmin(
    data: torch.Tensor,
    centroids: torch.Tensor,
    batch_size: int | None = None,
) -> torch.Tensor:
    """Per-subspace nearest-centroid codes for data (device-agnostic, chunked).

    Args:
        data (torch.Tensor): ``(N, D)`` vectors.
        centroids (torch.Tensor): ``(M, ksub, dsub)`` codebook.
        batch_size (int or None, optional): Chunk size to bound memory; None = single batch.

    Returns:
        torch.Tensor: ``(N, M)`` int64 codes.
    """
    M, ksub, dsub = centroids.shape
    out = torch.empty(data.shape[0], M, dtype=torch.int64, device=data.device)
    if batch_size is None:
        batch_size = data.shape[0]
    for start in range(0, data.shape[0], batch_size):
        chunk = data[start : start + batch_size].view(-1, M, dsub)
        out[start : start + chunk.shape[0]] = _subspace_distances(chunk, centroids).argmin(-1)
    return out


class _PQBase:
    """Shared state and encode/decode logic for product quantizers.

    A product quantizer splits each ``D``-dim vector into ``M`` sub-vectors of size
    ``dsub = D // M`` and quantizes each with its own ``ksub = 2**nbits``-entry
    codebook; the code is the concatenation of the per-subspace nearest-centroid
    indices. Subclasses provide :meth:`fit` (torch-native vs faiss-cpu); the
    codebook lives as a ``(M, ksub, dsub)`` torch tensor so the surrounding SID
    pipeline (state_dict, device moves) works unchanged.
    """

    def __init__(self, d: int, M: int, nbits: int, device: torch.device | str = "cpu"):
        if d % M != 0:
            raise ValueError(f"d ({d}) must be divisible by M ({M})")
        if not (1 <= nbits <= 16):
            raise ValueError(f"nbits must be in [1, 16], got {nbits}")
        self.d = d
        self.M = M
        self.nbits = nbits
        self.ksub = 1 << nbits
        self.dsub = d // M
        self.device = torch.device(device)
        self.centroids: torch.Tensor | None = None
        self.is_trained = False

    def encode(self, data: torch.Tensor, batch_size: int | None = None) -> torch.Tensor:
        """Return the ``(N, M)`` int64 sub-codes for ``data``."""
        if not self.is_trained:
            raise ValueError("Model not fitted yet.")
        return _encode_by_argmin(data.to(self.device), self.centroids, batch_size)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Reconstruct ``(N, D)`` vectors from ``(N, M)`` sub-codes."""
        if not self.is_trained:
            raise ValueError("Model not fitted yet.")
        codes = codes.to(self.device)
        return self.centroids[
            torch.arange(self.M, device=codes.device).unsqueeze(0).expand(codes.shape[0], -1), codes
        ].view(codes.shape[0], self.d)

    def distortion(self, data: torch.Tensor, batch_size: int | None = None) -> float:
        """Mean squared reconstruction error ``E[||x - decode(encode(x))||^2]``."""
        codes = self.encode(data, batch_size)
        recon = self.decode(codes)
        return float(torch.mean((data.to(self.device) - recon) ** 2))

    def set_centroids(self, centroids: torch.Tensor):
        """Inject a ``(M, ksub, dsub)`` codebook (used for parity checks / checkpoint load)."""
        if centroids.shape != (self.M, self.ksub, self.dsub):
            raise ValueError(
                f"expected centroids of shape {(self.M, self.ksub, self.dsub)}, got {tuple(centroids.shape)}"
            )
        self.centroids = centroids.to(self.device)
        self.is_trained = True

    def to(self, device: torch.device | str):
        """Move the codebook to ``device`` (torch backends only)."""
        self.device = torch.device(device)
        if self.centroids is not None:
            self.centroids = self.centroids.to(device)
        return self


class TorchPQ(_PQBase):
    """torch-native product quantizer (device-agnostic: CPU/GPU/NPU).

    Fits one ``TorchKMeans`` per subspace on the sub-vectors and encodes by
    per-subspace nearest-centroid argmin. Matches faiss ``IndexPQ`` output exactly
    given identical codebooks (see tests).
    """

    def __init__(
        self,
        d: int,
        M: int,
        nbits: int,
        device: torch.device | str = "cpu",
        kmeans_iters: int = DEFAULT_KMEANS_ITERS,
        kmeans_tol: float = 1e-4,
    ):
        super().__init__(d, M, nbits, device)
        self.kmeans_iters = kmeans_iters
        self.kmeans_tol = kmeans_tol

    def fit(self, data: torch.Tensor, batch_size: int | None = None):
        """Fit per-subspace k-means codebooks on ``data`` ``(N, D)``."""
        x_view = data.to(self.device).view(-1, self.M, self.dsub)
        centroids = torch.empty(self.M, self.ksub, self.dsub, dtype=data.dtype, device=self.device)
        for m in tqdm(range(self.M), desc="TorchPQ fit", leave=False):
            km = TorchKMeans(self.ksub, torch.cdist, MEAN_FUNCTIONS["mean"], self.device)
            km.fit(x_view[:, m], max_iterations=self.kmeans_iters, tolerance=self.kmeans_tol, batch_size=batch_size)
            centroids[m] = km.centroids
        self.centroids = centroids
        self.is_trained = True
        return self


class FaissPQ(_PQBase):
    """faiss-cpu ``IndexPQ`` exposing the same API as :class:`TorchPQ`.

    The heavy lifting (sub-quantizer k-means and packed-code encode) runs on faiss
    CPU; the codebook is mirrored to a torch tensor so state is interchangeable
    with :class:`TorchPQ`.
    """

    def __init__(self, d: int, M: int, nbits: int, device: torch.device | str = "cpu"):
        super().__init__(d, M, nbits, "cpu")
        self._index = None

    def _build_index(self):
        faiss = _get_faiss()
        index = faiss.IndexPQ(self.d, self.M, self.nbits)
        if self.centroids is not None:
            vec = faiss.Float32Vector()
            faiss.copy_array_to_vector(self.centroids.detach().cpu().numpy().reshape(-1), vec)
            index.pq.centroids = vec
            index.is_trained = True
        return index

    def fit(self, data: torch.Tensor, batch_size: int | None = None):
        """Train faiss ``IndexPQ`` on ``data`` (numpy/torch ``(N, D)`` float32)."""
        faiss = _get_faiss()
        arr = data.detach().cpu().numpy().astype("float32")
        index = faiss.IndexPQ(self.d, self.M, self.nbits)
        index.train(arr)
        self.centroids = torch.from_numpy(faiss.vector_to_array(index.pq.centroids)).view(self.M, self.ksub, self.dsub)
        self.is_trained = True
        self._index = index
        return self

    def encode(self, data: torch.Tensor, batch_size: int | None = None) -> torch.Tensor:
        """Encode with faiss (packed) then unpack to ``(N, M)`` int64 codes."""
        if not self.is_trained:
            raise ValueError("Model not fitted yet.")
        if self._index is None:
            self._index = self._build_index()
        arr = data.detach().cpu().numpy().astype("float32")
        return torch.from_numpy(_unpack_codes(self._index.sa_encode(arr), self.M, self.nbits))

    def set_centroids(self, centroids: torch.Tensor):
        """Inject a codebook and rebuild the underlying faiss index."""
        super().set_centroids(centroids)
        self._index = None

    def to(self, device: torch.device | str):
        """faiss always runs on CPU; ignore the requested device (kept for the API)."""
        self.device = torch.device("cpu")
        if self.centroids is not None:
            self.centroids = self.centroids.to("cpu")
        return self


class TorchOPQ(_PQBase):
    """torch-native optimized product quantizer (rotation + :class:`TorchPQ`).

    Learns an orthonormal rotation ``A`` (``D x D``) by iterating faiss's
    ``OPQMatrix`` scheme: per-subspace k-means on the rotated data, then an
    orthogonal-Procrustes update of ``A`` aligning the PQ reconstruction to the
    originals. Transform follows the faiss convention ``y = x @ A.T``; the PQ
    codebook lives in rotated space. Distortion matches faiss OPQ to tolerance
    (rotations need not be bit-identical).
    """

    def __init__(
        self,
        d: int,
        M: int,
        nbits: int,
        device: torch.device | str = "cpu",
        niter: int = DEFAULT_OPQ_NITER,
        kmeans_iters: int = DEFAULT_KMEANS_ITERS,
        kmeans_tol: float = 1e-4,
    ):
        super().__init__(d, M, nbits, device)
        self.niter = niter
        self.kmeans_iters = kmeans_iters
        self.kmeans_tol = kmeans_tol
        self.A: torch.Tensor | None = None

    def fit(self, data: torch.Tensor, batch_size: int | None = None):
        """Iteratively learn the rotation ``A`` and the rotated-space PQ codebook."""
        x = data.to(self.device)
        a = torch.eye(self.d, dtype=x.dtype, device=self.device)
        centroids = torch.empty(self.M, self.ksub, self.dsub, dtype=x.dtype, device=self.device)
        for _ in tqdm(range(self.niter), desc="TorchOPQ fit", leave=False):
            y = x @ a.T
            y_view = y.view(-1, self.M, self.dsub)
            for m in range(self.M):
                km = TorchKMeans(self.ksub, torch.cdist, MEAN_FUNCTIONS["mean"], self.device)
                km.fit(y_view[:, m], max_iterations=self.kmeans_iters, tolerance=self.kmeans_tol, batch_size=batch_size)
                centroids[m] = km.centroids
            codes = _encode_by_argmin(y, centroids, batch_size)
            q = centroids[torch.arange(self.M, device=self.device).unsqueeze(0).expand(x.shape[0], -1), codes].view(
                x.shape[0], self.d
            )
            a = _orthogonal_procrustes(x, q)
        self.A = a
        self.centroids = centroids
        self.is_trained = True
        return self

    def encode(self, data: torch.Tensor, batch_size: int | None = None) -> torch.Tensor:
        """Rotate ``data`` by ``A`` then per-subspace argmin encode."""
        if not self.is_trained:
            raise ValueError("Model not fitted yet.")
        y = data.to(self.device) @ self.A.T
        return _encode_by_argmin(y, self.centroids, batch_size)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Reconstruct ``(N, D)`` vectors, rotating the PQ reconstruction back by ``A``."""
        if not self.is_trained:
            raise ValueError("Model not fitted yet.")
        codes = codes.to(self.device)
        q = self.centroids[
            torch.arange(self.M, device=codes.device).unsqueeze(0).expand(codes.shape[0], -1), codes
        ].view(codes.shape[0], self.d)
        return q @ self.A

    def set_state(self, A: torch.Tensor, centroids: torch.Tensor):
        """Inject a rotation + rotated-space codebook (parity checks / checkpoint load)."""
        if A.shape != (self.d, self.d):
            raise ValueError(f"expected A of shape {(self.d, self.d)}, got {tuple(A.shape)}")
        self.A = A.to(self.device)
        self.set_centroids(centroids)

    def to(self, device: torch.device | str):
        """Move both the rotation and the rotated-space codebook to ``device``."""
        super().to(device)
        if self.A is not None:
            self.A = self.A.to(device)
        return self


class FaissOPQ(_PQBase):
    """faiss-cpu ``IndexPreTransform(OPQMatrix, IndexPQ)`` exposing the TorchOPQ API."""

    def __init__(self, d: int, M: int, nbits: int, device: torch.device | str = "cpu", niter: int = DEFAULT_OPQ_NITER):
        super().__init__(d, M, nbits, "cpu")
        self.niter = niter
        self.A: torch.Tensor | None = None
        self._pre = None

    def _build_pre(self):
        faiss = _get_faiss()
        opq = faiss.OPQMatrix(self.d, self.M)
        opq.niter = self.niter
        pq = faiss.IndexPQ(self.d, self.M, self.nbits)
        if self.A is not None:
            vec = faiss.Float32Vector()
            faiss.copy_array_to_vector(self.A.detach().cpu().numpy().reshape(-1), vec)
            opq.A = vec
            opq.is_trained = True
        if self.centroids is not None:
            vec = faiss.Float32Vector()
            faiss.copy_array_to_vector(self.centroids.detach().cpu().numpy().reshape(-1), vec)
            pq.pq.centroids = vec
            pq.is_trained = True
        return faiss.IndexPreTransform(opq, pq)

    def fit(self, data: torch.Tensor, batch_size: int | None = None):
        """Train faiss OPQ + PQ on ``data`` ``(N, D)`` float32."""
        faiss = _get_faiss()
        arr = data.detach().cpu().numpy().astype("float32")
        pre = self._build_pre()
        pre.train(arr)
        opq = faiss.downcast_VectorTransform(pre.chain.at(0))
        pq = faiss.downcast_index(pre.index)
        self.A = torch.from_numpy(faiss.vector_to_array(opq.A)).view(self.d, self.d)
        self.centroids = torch.from_numpy(faiss.vector_to_array(pq.pq.centroids)).view(self.M, self.ksub, self.dsub)
        self.is_trained = True
        self._pre = pre
        return self

    def encode(self, data: torch.Tensor, batch_size: int | None = None) -> torch.Tensor:
        """Encode with faiss (rotation + packed PQ) then unpack to ``(N, M)`` codes."""
        if not self.is_trained:
            raise ValueError("Model not fitted yet.")
        if self._pre is None:
            self._pre = self._build_pre()
        arr = data.detach().cpu().numpy().astype("float32")
        return torch.from_numpy(_unpack_codes(self._pre.sa_encode(arr), self.M, self.nbits))

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Reconstruct ``(N, D)`` vectors via faiss (rotation + PQ then rotate back)."""
        if not self.is_trained:
            raise ValueError("Model not fitted yet.")
        if self._pre is None:
            self._pre = self._build_pre()
        packed = _pack_codes(codes.detach().cpu().numpy(), self.nbits)
        return torch.from_numpy(self._pre.sa_decode(packed))

    def set_state(self, A: torch.Tensor, centroids: torch.Tensor):
        """Inject a rotation + rotated-space codebook and rebuild the faiss index."""
        if A.shape != (self.d, self.d):
            raise ValueError(f"expected A of shape {(self.d, self.d)}, got {tuple(A.shape)}")
        self.A = A.to("cpu")
        self.set_centroids(centroids)
        self._pre = None

    def to(self, device: torch.device | str):
        """faiss always runs on CPU; ignore the requested device (kept for the API)."""
        self.device = torch.device("cpu")
        if self.centroids is not None:
            self.centroids = self.centroids.to("cpu")
        if self.A is not None:
            self.A = self.A.to("cpu")
        return self


def make_pq(backend: str, d: int, M: int, nbits: int, device: torch.device | str = "cpu", **kwargs):
    """Instantiate a PQ/OPQ quantizer by backend name (``torch``/``faiss``)."""
    cls = {"torch": TorchPQ, "faiss": FaissPQ}[backend]
    return cls(d, M, nbits, device, **kwargs)


def make_opq(backend: str, d: int, M: int, nbits: int, device: torch.device | str = "cpu", **kwargs):
    """Instantiate an OPQ quantizer by backend name (``torch``/``faiss``)."""
    cls = {"torch": TorchOPQ, "faiss": FaissOPQ}[backend]
    return cls(d, M, nbits, device, **kwargs)


__all__ = [
    "TorchPQ",
    "FaissPQ",
    "TorchOPQ",
    "FaissOPQ",
    "make_pq",
    "make_opq",
    "_pack_codes",
    "_unpack_codes",
]
