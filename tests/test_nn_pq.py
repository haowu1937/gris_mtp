import math

import numpy as np
import pytest
import torch

from gensid.nn.pq import (
    FaissOPQ,
    FaissPQ,
    TorchOPQ,
    TorchPQ,
    _pack_codes,
    _unpack_codes,
    make_opq,
    make_pq,
)


@pytest.fixture
def gaussian_data():
    torch.manual_seed(0)
    return torch.randn(2000, 32)


@pytest.fixture
def structured_data():
    """Low-rank data: OPQ's rotation must reduce distortion vs plain PQ."""
    torch.manual_seed(0)
    latent = torch.randn(1000, 4)
    proj = torch.randn(4, 32)
    return latent @ proj


class TestPackUnpack:
    def test_round_trip_identity(self):
        codes = np.array([[0, 1, 2, 3], [15, 14, 13, 12], [7, 8, 9, 10]])
        assert np.array_equal(_unpack_codes(_pack_codes(codes, 4), 4, 4), codes)

    def test_round_trip_eight_bits(self):
        codes = np.array([[0, 255, 1], [128, 64, 200]])
        assert np.array_equal(_unpack_codes(_pack_codes(codes, 8), 3, 8), codes)

    def test_pack_layout_low_nibble_first(self):
        # M=1, nbits=4: code 0x0D -> single byte 0x0D; code 0x01 -> 0x01
        packed = _pack_codes(np.array([[13], [1]]), 4)
        assert packed.tolist() == [[13], [1]]

    def test_matches_faiss_byte_layout(self, gaussian_data):
        faiss = pytest.importorskip("faiss")
        xs = gaussian_data.numpy().astype("float32")
        index = faiss.IndexPQ(32, 4, 4)
        index.train(xs)
        bytes_faiss = index.sa_encode(xs)
        # decode faiss bytes, re-pack, must reproduce the exact byte string
        codes = _unpack_codes(bytes_faiss, 4, 4)
        assert np.array_equal(_pack_codes(codes, 4), bytes_faiss)


class TestTorchPQ:
    def _make(self, **kwargs):
        kwargs.setdefault("d", 32)
        kwargs.setdefault("M", 4)
        kwargs.setdefault("nbits", 4)
        kwargs.setdefault("kmeans_iters", 10)
        return TorchPQ(**kwargs)

    def test_init_validation(self):
        with pytest.raises(ValueError, match="divisible by M"):
            TorchPQ(31, 4, 4)
        with pytest.raises(ValueError, match="nbits must be in"):
            TorchPQ(32, 4, 0)

    def test_encode_before_fit_raises(self, gaussian_data):
        with pytest.raises(ValueError, match="not fitted"):
            self._make().encode(gaussian_data)

    def test_fit_encode_decode_shapes_and_bounds(self, gaussian_data):
        pq = self._make()
        pq.fit(gaussian_data)
        codes = pq.encode(gaussian_data)
        assert codes.shape == (2000, 4)
        assert codes.dtype == torch.int64
        assert codes.min() >= 0 and codes.max() < 16
        recon = pq.decode(codes)
        assert recon.shape == gaussian_data.shape
        assert math.isfinite(pq.distortion(gaussian_data))
        assert pq.distortion(gaussian_data) >= 0

    def test_encode_chunked_matches_full(self, gaussian_data):
        pq = self._make()
        pq.fit(gaussian_data)
        assert torch.equal(pq.encode(gaussian_data, batch_size=256), pq.encode(gaussian_data))

    def test_deterministic_given_same_seed(self, gaussian_data):
        torch.manual_seed(0)
        a = self._make()
        a.fit(gaussian_data)
        torch.manual_seed(0)
        b = self._make()
        b.fit(gaussian_data)
        assert torch.equal(a.encode(gaussian_data), b.encode(gaussian_data))

    def test_set_centroids_validates_shape(self):
        pq = self._make()
        with pytest.raises(ValueError, match="expected centroids"):
            pq.set_centroids(torch.zeros(2, 16, 16))

    def test_to_moves_centroids(self, gaussian_data):
        pq = self._make()
        pq.fit(gaussian_data)
        assert pq.to("cpu") is pq
        assert pq.centroids.device.type == "cpu"

    def test_encode_matches_faiss_given_same_centroids(self, gaussian_data):
        # deterministic encode parity: torch argmin on faiss's codebook == faiss codes
        faiss = pytest.importorskip("faiss")
        xs = gaussian_data.numpy().astype("float32")
        index = faiss.IndexPQ(32, 4, 4)
        index.train(xs)
        faiss_codes = torch.from_numpy(_unpack_codes(index.sa_encode(xs), 4, 4))
        pq = TorchPQ(32, 4, 4)
        pq.set_centroids(torch.from_numpy(faiss.vector_to_array(index.pq.centroids)).view(4, 16, 8))
        assert torch.equal(pq.encode(gaussian_data), faiss_codes)

    def test_distortion_close_to_faiss(self, gaussian_data):
        # same-seed end-to-end: torch PQ distortion must track faiss's within 10%
        pytest.importorskip("faiss")
        torch.manual_seed(0)
        torch_pq = TorchPQ(32, 4, 4, kmeans_iters=25)
        torch_pq.fit(gaussian_data)
        faiss_pq = FaissPQ(32, 4, 4)
        faiss_pq.fit(gaussian_data)
        ratio = torch_pq.distortion(gaussian_data) / faiss_pq.distortion(gaussian_data)
        assert 0.9 < ratio < 1.1


class TestFaissPQ:
    def _make(self, **kwargs):
        kwargs.setdefault("d", 32)
        kwargs.setdefault("M", 4)
        kwargs.setdefault("nbits", 4)
        return FaissPQ(**kwargs)

    def test_fit_encode_round_trip(self, gaussian_data):
        pytest.importorskip("faiss")
        pq = self._make()
        pq.fit(gaussian_data)
        codes = pq.encode(gaussian_data)
        assert codes.shape == (2000, 4)
        assert codes.min() >= 0 and codes.max() < 16
        assert pq.decode(codes).shape == gaussian_data.shape

    def test_encode_matches_torch_given_torch_centroids(self, gaussian_data):
        # reverse direction: faiss encoding with a torch-trained codebook == torch codes
        pytest.importorskip("faiss")
        torch.manual_seed(0)
        torch_pq = TorchPQ(32, 4, 4, kmeans_iters=10)
        torch_pq.fit(gaussian_data)
        faiss_pq = FaissPQ(32, 4, 4)
        faiss_pq.set_centroids(torch_pq.centroids.clone())
        assert torch.equal(faiss_pq.encode(gaussian_data), torch_pq.encode(gaussian_data))

    def test_to_ignores_requested_device(self, gaussian_data):
        pytest.importorskip("faiss")
        pq = self._make()
        pq.fit(gaussian_data)
        pq.to("cuda:0")
        assert pq.device.type == "cpu"
        assert pq.centroids.device.type == "cpu"


class TestTorchOPQ:
    def _make(self, **kwargs):
        kwargs.setdefault("d", 32)
        kwargs.setdefault("M", 4)
        kwargs.setdefault("nbits", 4)
        kwargs.setdefault("niter", 6)
        kwargs.setdefault("kmeans_iters", 10)
        return TorchOPQ(**kwargs)

    def test_fit_encode_decode_shapes(self, gaussian_data):
        opq = self._make()
        opq.fit(gaussian_data)
        codes = opq.encode(gaussian_data)
        assert codes.shape == (2000, 4)
        assert codes.min() >= 0 and codes.max() < 16
        assert opq.decode(codes).shape == gaussian_data.shape

    def test_rotation_is_orthonormal(self, gaussian_data):
        opq = self._make()
        opq.fit(gaussian_data)
        eye = torch.eye(32)
        assert torch.allclose(opq.A @ opq.A.T, eye, atol=1e-4)

    def test_rotation_reduces_distortion_vs_plain_pq(self, structured_data):
        # the whole point of OPQ: on structured data the learned rotation must
        # cut the quantization error well below plain PQ
        torch.manual_seed(0)
        opq = TorchOPQ(32, 4, 4, niter=8, kmeans_iters=15)
        opq.fit(structured_data)
        torch.manual_seed(0)
        pq = TorchPQ(32, 4, 4, kmeans_iters=15)
        pq.fit(structured_data)
        assert opq.distortion(structured_data) < pq.distortion(structured_data) * 0.9

    def test_deterministic_given_same_seed(self, gaussian_data):
        torch.manual_seed(0)
        a = self._make()
        a.fit(gaussian_data)
        torch.manual_seed(0)
        b = self._make()
        b.fit(gaussian_data)
        assert torch.equal(a.encode(gaussian_data), b.encode(gaussian_data))

    def test_set_state_validates(self):
        opq = self._make()
        with pytest.raises(ValueError, match="expected A"):
            opq.set_state(torch.zeros(5, 5), torch.zeros(4, 16, 8))

    def test_encode_matches_faiss_given_same_state(self, gaussian_data):
        # deterministic encode parity for OPQ: inject faiss's A + codebook
        faiss = pytest.importorskip("faiss")
        xs = gaussian_data.numpy().astype("float32")
        pre = faiss.IndexPreTransform(faiss.OPQMatrix(32, 4), faiss.IndexPQ(32, 4, 4))
        pre.train(xs)
        opq = faiss.downcast_VectorTransform(pre.chain.at(0))
        pq = faiss.downcast_index(pre.index)
        A = torch.from_numpy(faiss.vector_to_array(opq.A)).view(32, 32)
        C = torch.from_numpy(faiss.vector_to_array(pq.pq.centroids)).view(4, 16, 8)
        torch_opq = TorchOPQ(32, 4, 4)
        torch_opq.set_state(A, C)
        expected = torch.from_numpy(_unpack_codes(pre.sa_encode(xs), 4, 4))
        assert torch.equal(torch_opq.encode(gaussian_data), expected)
        # decode parity too: reconstruction must match faiss's rotated-back decode
        assert torch.allclose(torch_opq.decode(expected), torch.from_numpy(pre.sa_decode(pre.sa_encode(xs))), atol=1e-5)

    def test_distortion_not_worse_than_faiss(self, structured_data):
        # torch OPQ distortion must be within tolerance of (or better than) faiss's
        pytest.importorskip("faiss")
        torch.manual_seed(0)
        torch_opq = TorchOPQ(32, 4, 4, niter=8, kmeans_iters=15)
        torch_opq.fit(structured_data)
        faiss_opq = FaissOPQ(32, 4, 4)
        faiss_opq.fit(structured_data)
        assert torch_opq.distortion(structured_data) <= faiss_opq.distortion(structured_data) * 1.1

    def test_to_moves_rotation_and_codebook(self, structured_data):
        # regression: TorchOPQ.to() must move A too, not just the centroids
        # (a stale CPU rotation broke generation on CUDA)
        opq = TorchOPQ(32, 4, 4, niter=2, kmeans_iters=5)
        opq.fit(structured_data)
        opq.to(torch.device("meta"))
        assert opq.A.device.type == "meta"
        assert opq.centroids.device.type == "meta"


class TestFaissOPQ:
    def _make(self, **kwargs):
        kwargs.setdefault("d", 32)
        kwargs.setdefault("M", 4)
        kwargs.setdefault("nbits", 4)
        return FaissOPQ(**kwargs)

    def test_fit_encode_round_trip(self, gaussian_data):
        pytest.importorskip("faiss")
        opq = self._make()
        opq.fit(gaussian_data)
        codes = opq.encode(gaussian_data)
        assert codes.shape == (2000, 4)
        assert opq.decode(codes).shape == gaussian_data.shape

    def test_encode_matches_torch_given_torch_state(self, gaussian_data):
        pytest.importorskip("faiss")
        torch.manual_seed(0)
        torch_opq = TorchOPQ(32, 4, 4, niter=3, kmeans_iters=10)
        torch_opq.fit(gaussian_data)
        faiss_opq = FaissOPQ(32, 4, 4)
        faiss_opq.set_state(torch_opq.A.clone(), torch_opq.centroids.clone())
        assert torch.equal(faiss_opq.encode(gaussian_data), torch_opq.encode(gaussian_data))

    def test_to_ignores_requested_device(self, gaussian_data):
        pytest.importorskip("faiss")
        opq = self._make()
        opq.fit(gaussian_data)
        opq.to("cuda:0")
        assert opq.device.type == "cpu"


class TestMakeFactories:
    def test_make_pq_and_opq(self):
        assert isinstance(make_pq("torch", 32, 4, 4), TorchPQ)
        assert isinstance(make_pq("faiss", 32, 4, 4), FaissPQ)
        assert isinstance(make_opq("torch", 32, 4, 4), TorchOPQ)
        assert isinstance(make_opq("faiss", 32, 4, 4), FaissOPQ)

    def test_unknown_backend_raises(self):
        with pytest.raises(KeyError):
            make_pq("bogus", 32, 4, 4)
        with pytest.raises(KeyError):
            make_opq("bogus", 32, 4, 4)
