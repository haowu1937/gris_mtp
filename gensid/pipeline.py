import dataclasses
import logging

import torch

from gensid.data.base import build_data_loader, build_dataset, get_dataset_config
from gensid.registry import REGISTRY

logger = logging.getLogger(__name__)

_NO_DIST = object()


class SIDPipeline:
    def __init__(self, args, device, dist_state=None):
        self.args, self.device = args, device
        self.dist_state = dist_state
        logger.info("SIDPipeline config: %s", vars(args))

    def _build_model(self, dataset_dim):
        name = self.args.sid_model
        config = REGISTRY.get_sid_method_config(name).from_args(self.args)
        field_names = {f.name for f in dataclasses.fields(config)}
        if "in_dim" in field_names:
            config = dataclasses.replace(config, in_dim=dataset_dim)
        if "cf_embedding" in field_names and isinstance(config.cf_embedding, str):
            cf_embedding = torch.load(config.cf_embedding, weights_only=False).squeeze().detach().cpu().numpy()
            config = dataclasses.replace(config, cf_embedding=cf_embedding)
        return REGISTRY.get_sid_method(name)(config)

    def _build_trainer(self, model, dist_state=_NO_DIST):
        if dist_state is _NO_DIST:
            dist_state = self.dist_state
        name = model.trainer
        config = REGISTRY.get_trainer_config(name).from_args(self.args)
        return REGISTRY.get_trainer(name)(
            config,
            model,
            device=self.device,
            ckpt_dir=self.args.ckpt_dir,
            saved_model_dir=self.args.saved_model_dir,
            dist_state=dist_state,
        )

    def _build_dataset_and_loader(self, eval_mode: bool):
        config = get_dataset_config(self.args)
        dataset = build_dataset(config, self.args.embeddings_path, self.args.interactions_path, eval_mode=eval_mode)
        loader = build_data_loader(dataset, config, self.args, eval_mode=eval_mode, dist_state=self.dist_state)
        return dataset, loader

    def _log_dataset(self, dataset, eval_mode: bool):
        logger.info(
            "Data: mode=%s dataset=%s dim=%d items=%d train_items=%d device=%s",
            "eval" if eval_mode else "train",
            getattr(dataset, "name", type(dataset).__name__),
            dataset.dim,
            len(dataset),
            len(getattr(dataset, "train_item_ids", [])),
            self.device,
        )

    def _log_model(self, model):
        num_params = sum(p.numel() for p in model.parameters())
        logger.info("Model: %s (%s) params=%d device=%s", model.name, type(model).__name__, num_params, self.device)

    def train(self):
        dataset, loader = self._build_dataset_and_loader(eval_mode=False)
        self._log_dataset(dataset, eval_mode=False)
        model = self._build_model(dataset.dim).to(self.device)
        self._log_model(model)
        trainer = self._build_trainer(model)
        return trainer.fit(dataset, loader)

    def generate(self, ckpt_path, output_file):
        dataset, loader = self._build_dataset_and_loader(eval_mode=True)
        self._log_dataset(dataset, eval_mode=True)
        model = self._build_model(dataset.dim)
        model.load_state_dict(torch.load(ckpt_path, weights_only=False)["state_dict"], strict=False)
        model = model.to(self.device).eval()
        self._log_model(model)
        # Generation is single-process inference on the calling rank: never DDP-wrap
        # the model again (a fresh wrap would fire a collective on a dead group).
        trainer = self._build_trainer(model, dist_state=None)
        trainer.generate_indices(dataset, loader, output_file)
