import json
import logging
import os
import sys
from time import time

import numpy as np
import torch
import torch.distributed as dist
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from gensid.eval import id_collision_rate
from gensid.generation.base import collect_indices, report_and_save
from gensid.utils import ensure_dir, get_local_time, set_color, wrap_model_for_ddp

logger = logging.getLogger(__name__)


class Trainer:
    name = "base"

    def __init__(self, config, model, device=None, ckpt_dir=None, saved_model_dir=None, dist_state=None):
        self.config = config
        self.model = model

        self.lr = config.lr
        self.learner = config.learner
        self.weight_decay = config.weight_decay
        self.epochs = config.epochs
        self.eval_step = min(config.eval_step, self.epochs)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.ckpt_dir = os.path.join(ckpt_dir, saved_model_dir or get_local_time())
        ensure_dir(self.ckpt_dir)
        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_collision_ckpt = "best_collision_model.pth"
        self.latest_ckpt = "latest_model.pth"
        self.dist_state = dist_state
        self.is_distributed = dist_state is not None and dist_state.world_size > 1
        self.rank = dist_state.rank if self.is_distributed else 0
        self.optimizer = self._build_optimizer()
        self.model = self.model.to(self.device)
        if config.compile:
            self.model = torch.compile(self.model)
        # ``self.model`` stays the raw module so trainers can read model attributes
        # and run inference; ``self.ddp_model`` is used only for the training forward
        # so gradient all-reduce fires under DDP.
        if self.is_distributed and isinstance(self.model, torch.nn.Module):
            self.ddp_model = wrap_model_for_ddp(self.model, dist_state)
        else:
            self.ddp_model = self.model

        # TensorBoard
        self.tb_writer = SummaryWriter(log_dir=os.path.join(self.ckpt_dir, "tensorboard"))

    def _build_optimizer(self):
        params = self.model.parameters()
        learner = self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(params, lr=learning_rate, weight_decay=weight_decay)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device, non_blocking=True)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adamw":
            optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        else:
            logger.warning("Received unrecognized optimizer, set default Adam optimizer")
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def _train_epoch(self, train_data, epoch_idx):
        self.model.train()

        total_loss = 0
        used_codebooks = [set() for _ in range(self.model.n_codebooks)]

        iter_data = tqdm(
            train_data,
            total=len(train_data),
            ncols=100,
            desc=set_color(f"Train {epoch_idx + 1}", "pink"),
            disable=not sys.stderr.isatty(),
        )
        for data in iter_data:
            self.optimizer.zero_grad()

            for feat, data_feat in data.items():
                if isinstance(data_feat, torch.Tensor):
                    data[feat] = data_feat.to(self.device, non_blocking=True)

            out, indices = self.ddp_model(data)

            for codebook_idx in range(indices.shape[-1]):
                used_codebooks[codebook_idx].update(indices[:, codebook_idx].detach().cpu().tolist())

            loss, train_info = self.model.compute_loss(data, out)
            self._check_nan(loss)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        train_info = train_info or {}
        unused_codebooks = self._calculate_unused_codebooks(used_codebooks)

        return total_loss, {**train_info, "unused codebooks": unused_codebooks}

    @torch.no_grad()
    def _valid_epoch(self, valid_data):
        self.model.eval()

        iter_data = tqdm(
            valid_data,
            total=len(valid_data),
            ncols=100,
            desc=set_color("Evaluate", "pink"),
            disable=not sys.stderr.isatty(),
        )

        indices_set = set()
        num_sample = 0
        for data in iter_data:
            emb_data = data["emb"]
            num_sample += len(emb_data)
            emb_data = emb_data.to(self.device, non_blocking=True)
            indices = self.model.generate(emb_data)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

        collision_rate = id_collision_rate(indices_set, num_sample)

        return collision_rate

    def _calculate_unused_codebooks(self, used_codebooks):
        unused_codebooks = []
        for idx, codebook_size in enumerate(self.model.codebook_sizes):
            num_used_codebooks = len(used_codebooks[idx])
            unused_codebooks.append(codebook_size - num_used_codebooks)
        return sum(unused_codebooks)

    def _set_trained_flag(self, ckpt_file: str):
        if self.is_distributed and self.rank != 0:
            return
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_file)
        metrics_path = ckpt_path.replace(".pth", ".json")
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        metrics["fully_trained"] = True
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):
        if self.is_distributed and self.rank != 0:
            return

        ckpt_path = (
            os.path.join(self.ckpt_dir, ckpt_file)
            if ckpt_file
            else os.path.join(self.ckpt_dir, f"epoch_{epoch}_collision_{collision_rate:.4f}_model.pth")
        )
        state_dict = self.model.state_dict()
        state = {
            "args": self.config,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": state_dict,
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        metrics = {
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "ckpt_path": ckpt_path,
        }

        metrics_path = ckpt_path.replace(".pth", ".json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(set_color("Saving current", "blue") + f": {ckpt_path}")

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, **kwargs):
        self.tb_writer.add_scalar("train/epoch_time_sec", e_time - s_time, epoch_idx)
        self.tb_writer.add_scalar("train/train_loss", loss, epoch_idx)
        for info_key, info_value in kwargs.items():
            if isinstance(info_value, (int, float, str)):
                self.tb_writer.add_scalar(f"train/{info_key}", info_value, epoch_idx)

        train_loss_output = set_color(f"epoch {epoch_idx} training", "green") + " ["
        train_loss_output += set_color("time", "blue") + f": {e_time - s_time:.2f}"
        train_loss_output += ", "
        train_loss_output += set_color("train loss", "blue") + f": {loss:.4f}"
        for info_key, info_value in kwargs.items():
            train_loss_output += ", "
            train_loss_output += set_color(info_key, "blue") + f": {info_value:.4f}"

        return train_loss_output + "]"

    def fit(self, dataset, data):
        curr_eval_step = 0
        collision_rate = 1.0
        for epoch_idx in range(self.epochs):
            if self.is_distributed:
                sampler = getattr(data, "sampler", None)
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch_idx)
            training_start_time = time()
            train_loss, train_info = self._train_epoch(data, epoch_idx)
            training_end_time = time()

            train_loss_output = self._generate_train_loss_output(
                epoch_idx + 1, training_start_time, training_end_time, train_loss, **train_info
            )
            logger.info(train_loss_output)

            self.best_loss = min(self.best_loss, train_loss)

            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                collision_rate = self._valid_epoch(data)

                # TODO: why is only the collision rate as a validation metric for best model?
                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    curr_eval_step = 0
                    self._save_checkpoint(
                        epoch_idx,
                        collision_rate=collision_rate,
                        ckpt_file=self.best_collision_ckpt,
                    )
                else:
                    curr_eval_step += 1
                    self._save_checkpoint(
                        epoch_idx,
                        collision_rate=collision_rate,
                        ckpt_file=self.latest_ckpt,
                    )

                valid_end_time = time()
                valid_score_output = (
                    set_color(f"epoch {epoch_idx + 1} evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + f": {valid_end_time - valid_start_time:.2f}, "
                    + set_color("collision_rate", "blue")
                    + f": {collision_rate:f}]"
                )
                self.tb_writer.add_scalar("train/collision_rate", collision_rate, epoch_idx)

                logger.info(valid_score_output)

        self._save_checkpoint(epoch_idx, collision_rate=collision_rate, ckpt_file=self.latest_ckpt)
        self._set_trained_flag(ckpt_file=self.best_collision_ckpt)
        if self.is_distributed:
            dist.barrier()
        logger.info(
            "Training finished. Best loss: %.4f, Best collision rate: %.4f, Checkpoint: %s",
            self.best_loss,
            self.best_collision_rate,
            os.path.join(self.ckpt_dir, self.best_collision_ckpt),
        )
        return self.best_loss, self.best_collision_rate

    @torch.no_grad()
    def generate_indices(self, dataset, data_loader, output_file):
        """Single-pass SID generation for the stock contract.

        Assumes ``model.generate(emb_tensor)`` returns ``[n, n_codebooks]``
        indices; the two-pass collision-fixing flow lives in ``RQTrainer``.
        """
        self.model.eval()

        def _generate(batch):
            emb_data = batch["emb"].to(self.device, non_blocking=True)
            return self.model.generate(emb_data)

        all_indices, all_indices_str = collect_indices(
            ((None, data) for data in tqdm(data_loader, desc="Generating SID Indices")),
            _generate,
            self.model.n_codebooks,
        )
        report_and_save(all_indices, all_indices_str, self.model.n_codebooks, self.model.codebook_sizes, output_file)
