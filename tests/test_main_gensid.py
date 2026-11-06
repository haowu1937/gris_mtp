import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main_gensid import parse_args


def _parse(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["main_gensid.py", *argv])
    return parse_args()


class TestConfigYaml:
    def test_no_config_uses_defaults(self, monkeypatch):
        args = _parse(monkeypatch, [])
        assert args.lr == 1e-3
        assert args.epochs == 20000
        assert args.checkpoint == "best_collision_model.pth"

    def test_yaml_keys_translated_to_args(self, monkeypatch, tmp_path):
        conf = str(tmp_path / "conf.yaml")
        with open(conf, "w") as f:
            yaml.dump({"lr": 0.5, "epochs": 3, "layers": [512, 256]}, f)
        args = _parse(monkeypatch, ["--config", conf])
        assert args.lr == 0.5
        assert args.epochs == 3
        assert args.layers == [512, 256]

    def test_cli_overrides_yaml(self, monkeypatch, tmp_path):
        conf = str(tmp_path / "conf.yaml")
        with open(conf, "w") as f:
            yaml.dump({"lr": 0.5, "epochs": 3}, f)
        args = _parse(monkeypatch, ["--config", conf, "--epochs", "7", "--seed", "1"])
        assert args.lr == 0.5  # yaml value kept
        assert args.epochs == 7  # CLI wins
        assert args.seed == 1

    def test_flat_yaml_dict_used(self, monkeypatch, tmp_path):
        conf = str(tmp_path / "conf.yaml")
        with open(conf, "w") as f:
            yaml.dump({"lr": 0.25, "graph_signal": True}, f)
        args = _parse(monkeypatch, ["--config", conf])
        assert args.lr == 0.25
        assert args.graph_signal is True

    def test_yaml_true_flag_and_prod_graph_args(self, monkeypatch, tmp_path):
        conf = str(tmp_path / "conf.yaml")
        with open(conf, "w") as f:
            yaml.dump(
                {
                    "graph_signal": True,
                    "parquet_folder": "/data/interactions",
                    "seq_col": "item_seq",
                    "max_files": 5,
                },
                f,
            )
        args = _parse(monkeypatch, ["--config", conf])
        assert args.graph_signal is True
        assert args.parquet_folder == "/data/interactions"
        assert args.seq_col == "item_seq"
        assert args.max_files == 5

    def test_missing_config_file_raises(self, monkeypatch, tmp_path):
        with pytest.raises(FileNotFoundError):
            _parse(monkeypatch, ["--config", str(tmp_path / "nope.yaml")])

    def test_checkpoint_override(self, monkeypatch):
        args = _parse(monkeypatch, ["--checkpoint", "latest_model.pth"])
        assert args.checkpoint == "latest_model.pth"

    def test_embeddings_path_kept_when_provided(self, monkeypatch):
        args = _parse(monkeypatch, ["--embeddings_path", "/data/embs.parquet"])
        assert args.embeddings_path == "/data/embs.parquet"

    def test_yaml_false_emits_negation_flags(self, monkeypatch, tmp_path):
        conf = str(tmp_path / "conf.yaml")
        with open(conf, "w") as f:
            yaml.dump({"directed": False, "chronological": False, "graph_signal": True}, f)
        args = _parse(monkeypatch, ["--config", conf])
        assert args.directed is False
        assert args.chronological is False
        assert args.graph_signal is True

    def test_yaml_booleans_equal_to_default_are_skipped(self, monkeypatch, tmp_path):
        conf = str(tmp_path / "conf.yaml")
        with open(conf, "w") as f:
            yaml.dump({"directed": True, "use_edge_weight": False}, f)
        args = _parse(monkeypatch, ["--config", conf])
        assert args.directed is True
        assert args.use_edge_weight is False

    def test_cli_no_directed_flag(self, monkeypatch):
        args = _parse(monkeypatch, ["--no_directed"])
        assert args.directed is False
