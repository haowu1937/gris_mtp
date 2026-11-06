import argparse
import json

import numpy as np
import pytest
import torch

from gensid.utils import (
    check_collision,
    ensure_dir,
    get_collision_item,
    get_device,
    get_trained_flag,
    init_dist_state,
    init_seed,
    is_distributed,
    yaml_to_argv,
)


class TestCheckCollision:
    @pytest.mark.parametrize(
        ("indices", "expected"),
        [
            (np.array([[0, 1], [1, 0], [2, 2]]), True),
            (np.array([[0, 1], [0, 1], [2, 2]]), False),
            (np.array([[0, 1], [0, 1]]), False),
        ],
    )
    def test_collision(self, indices, expected):
        assert check_collision(indices) is expected


class TestGetCollisionItem:
    @pytest.mark.parametrize(
        ("indices", "expected"),
        [
            ([[0, 1], [1, 0]], []),
            ([[0, 1], [0, 1], [2, 2], [2, 2], [3, 0]], [[0, 1], [2, 3]]),
            ([[1], [1]], [[0, 1]]),
        ],
    )
    def test_identifies_colliding_groups(self, indices, expected):
        assert get_collision_item(indices) == expected


class TestEnsureDir:
    def test_creates_directory(self, tmp_path):
        target = tmp_path / "nested" / "dir"
        ensure_dir(str(target))
        assert target.is_dir()

    def test_existing_directory_is_idempotent(self, tmp_path):
        ensure_dir(str(tmp_path))
        assert tmp_path.is_dir()


class TestGetDevice:
    @pytest.mark.parametrize("device_arg", ["cpu", "CPU"])
    def test_cpu_returns_cpu(self, device_arg):
        assert get_device(device_arg) == torch.device("cpu")

    def test_auto_falls_back_to_cpu_when_no_accelerator(self, monkeypatch):
        monkeypatch.setattr(torch, "cuda", type("CudaMock", (), {"is_available": staticmethod(lambda: False)})())
        if hasattr(torch, "npu"):
            monkeypatch.setattr(torch, "npu", type("NpuMock", (), {"is_available": staticmethod(lambda: False)})())
        assert get_device("auto") == torch.device("cpu")

    def test_cuda_raises_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(torch, "cuda", type("CudaMock", (), {"is_available": staticmethod(lambda: False)})())
        with pytest.raises(RuntimeError, match="CUDA requested but not available"):
            get_device("cuda")

    def test_unknown_device_raises(self):
        with pytest.raises(ValueError, match="Unknown device argument"):
            get_device("tpu")


class TestGetTrainedFlag:
    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            (None, False),  # file missing
            ({"fully_trained": True}, True),
            ({"fully_trained": False, "loss": 0.1}, False),
            ({"loss": 0.1}, False),  # key absent
        ],
    )
    def test_flag_parsing(self, tmp_path, payload, expected):
        path = tmp_path / "metrics.json"
        if payload is not None:
            path.write_text(json.dumps(payload))
        assert get_trained_flag(str(path)) is expected


class TestYamlToArgv:
    def test_scalars(self):
        assert yaml_to_argv({"lr": 0.5, "epochs": 3}) == ["--lr", "0.5", "--epochs", "3"]

    def test_true_emits_bare_flag(self):
        assert yaml_to_argv({"graph_signal": True}) == ["--graph_signal"]

    def test_false_emits_negation_flag(self):
        assert yaml_to_argv({"directed": False}) == ["--no_directed"]

    def test_none_is_skipped(self):
        assert yaml_to_argv({"tmp_dir": None}) == []

    def test_lists_expand_to_repeated_values(self):
        assert yaml_to_argv({"layers": [512, 256]}) == ["--layers", "512", "256"]

    def test_mixed_config_roundtrip(self):
        argv = yaml_to_argv(
            {"lr": 0.01, "graph_signal": True, "chronological": False, "layers": [128, 64], "embedder": None}
        )
        assert argv == ["--lr", "0.01", "--graph_signal", "--no_chronological", "--layers", "128", "64"]

    def test_parser_aware_skips_values_equal_to_default(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--no_directed", action="store_false", dest="directed", default=True)
        parser.add_argument("--graph_signal", action="store_true")
        assert yaml_to_argv({"directed": True, "graph_signal": False}, parser) == []
        assert yaml_to_argv({"directed": False, "graph_signal": True}, parser) == ["--no_directed", "--graph_signal"]


class TestInitSeed:
    def test_reproducible_results_after_reseed(self):
        init_seed(42)
        a = torch.randn(5)
        init_seed(42)
        b = torch.randn(5)
        assert torch.equal(a, b)

    def test_numpy_random_is_seeded(self):
        init_seed(7)
        a = np.random.rand(4)
        init_seed(7)
        b = np.random.rand(4)
        assert np.array_equal(a, b)


class TestIsDistributed:
    def test_false_without_torchrun_env(self, monkeypatch):
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        assert is_distributed() is False

    def test_false_when_world_size_one(self, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        assert is_distributed() is False

    def test_true_with_multi_process_env(self, monkeypatch):
        monkeypatch.setenv("RANK", "1")
        monkeypatch.setenv("WORLD_SIZE", "2")
        assert is_distributed() is True

    def test_init_dist_state_requires_torchrun_env(self, monkeypatch):
        monkeypatch.delenv("RANK", raising=False)
        with pytest.raises(KeyError):
            init_dist_state("cuda")
