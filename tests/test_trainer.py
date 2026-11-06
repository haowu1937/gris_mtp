import json
import os

import torch

from gensid.config.trainer_config import TrainerConfig
from gensid.trainer.base import Trainer


class TestTrainerFit:
    def _make_trainer(self, tmp_path, epochs=3):
        return Trainer(
            TrainerConfig(lr=0.01, learner="AdamW", weight_decay=0.0, epochs=epochs, eval_step=1, compile=False),
            torch.nn.Linear(4, 4),
            device="cpu",
            ckpt_dir=str(tmp_path),
            saved_model_dir="test",
        )

    def _run(self, tmp_path, monkeypatch, train_losses, collision_rates):
        tr = self._make_trainer(tmp_path)
        losses = iter(train_losses)
        rates = iter(collision_rates)
        monkeypatch.setattr(tr, "_train_epoch", lambda data, epoch: (next(losses), {}))
        monkeypatch.setattr(tr, "_valid_epoch", lambda data: next(rates))
        result = tr.fit(dataset=None, data=None)
        return tr, result

    def _read_sidecar(self, tr):
        metrics_path = os.path.join(tr.ckpt_dir, "best_collision_model.json")
        with open(metrics_path) as f:
            return json.load(f)

    def test_fit_saves_best_collision_ckpt_and_trained_flag(self, tmp_path, monkeypatch):
        # improvement only until epoch 1 -> best_collision_model.pth is not rewritten after
        tr, (best_loss, best_collision_rate) = self._run(
            tmp_path, monkeypatch, train_losses=[2.0, 1.5, 1.0], collision_rates=[0.5, 0.3, 0.4]
        )
        assert best_loss == 1.0
        assert best_collision_rate == 0.3
        assert tr.best_loss == 1.0
        assert tr.best_collision_rate == 0.3
        pth = os.path.join(tr.ckpt_dir, "best_collision_model.pth")
        assert os.path.exists(pth)
        metrics = self._read_sidecar(tr)
        assert metrics["best_collision_rate"] == 0.3
        assert metrics["epoch"] == 1
        assert metrics["fully_trained"] is True
        assert metrics["ckpt_path"] == pth

    def test_fit_keeps_first_best_when_collision_never_improves(self, tmp_path, monkeypatch):
        tr, (_, best_collision_rate) = self._run(
            tmp_path, monkeypatch, train_losses=[1.0, 1.0, 1.0], collision_rates=[0.5, 0.5, 0.5]
        )
        assert best_collision_rate == 0.5
        metrics = self._read_sidecar(tr)
        assert metrics["best_collision_rate"] == 0.5
        assert metrics["epoch"] == 0
        assert metrics["fully_trained"] is True

    def test_fit_saves_epoch_checkpoint_when_no_best_ckpt_name(self, tmp_path, monkeypatch):
        # when ckpt_file is not the best_collision_ckpt (e.g. a plain epoch dump), a
        # sidecar is still written and the trained flag is set on best_collision_ckpt
        tr = self._make_trainer(tmp_path, epochs=1)
        monkeypatch.setattr(tr, "_train_epoch", lambda data, epoch: (1.0, {}))
        monkeypatch.setattr(tr, "_valid_epoch", lambda data: 0.5)
        epoch_ckpt = os.path.join(tr.ckpt_dir, "epoch_x.pth")
        tr._save_checkpoint(0, collision_rate=0.5, ckpt_file="epoch_x.pth")
        assert os.path.exists(epoch_ckpt)
        assert os.path.exists(epoch_ckpt.replace(".pth", ".json"))

    def test_fit_saves_latest_ckpt_with_final_state(self, tmp_path, monkeypatch):
        # latest_model.pth must be written after the final epoch with the final
        # model state, regardless of whether the last eval improved the best rate
        tr = self._make_trainer(tmp_path, epochs=2)
        monkeypatch.setattr(tr, "_train_epoch", lambda data, epoch: (1.0, {}))
        monkeypatch.setattr(tr, "_valid_epoch", lambda data: 0.5)
        tr.fit(dataset=None, data=None)
        latest = os.path.join(tr.ckpt_dir, "latest_model.pth")
        assert os.path.exists(latest)
        ckpt = torch.load(latest, weights_only=False)
        assert ckpt["epoch"] == 1  # last epoch idx (2 epochs -> 0,1)
        assert set(ckpt["state_dict"].keys()) == set(tr.model.state_dict().keys())

    def test_fit_saves_latest_ckpt_when_collision_never_improves(self, tmp_path, monkeypatch):
        # the non-improving eval branch must also persist latest_model.pth
        tr = self._make_trainer(tmp_path, epochs=2)
        losses = iter([1.0, 1.0])
        rates = iter([0.5, 0.5])
        monkeypatch.setattr(tr, "_train_epoch", lambda data, epoch: (next(losses), {}))
        monkeypatch.setattr(tr, "_valid_epoch", lambda data: next(rates))
        tr.fit(dataset=None, data=None)
        latest = os.path.join(tr.ckpt_dir, "latest_model.pth")
        assert os.path.exists(latest)
        ckpt = torch.load(latest, weights_only=False)
        assert ckpt["epoch"] == 1
