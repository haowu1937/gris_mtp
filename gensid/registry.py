import importlib
import inspect
import pkgutil
from typing import Union, get_origin, get_type_hints

from torch.utils.data import Dataset as TorchDataset

from gensid.config import _CONFIGS
from gensid.config.dataset_config import DatasetConfig
from gensid.config.encoder_config import EncoderConfig
from gensid.config.graph_build_config import GraphBuildingConfig
from gensid.config.trainer_config import TrainerConfig
from gensid.method.base import SIDMethod
from gensid.trainer.base import Trainer


GROUPS = {
    "sid_method": ("gensid.method", SIDMethod, "name"),
    "trainer": ("gensid.trainer", Trainer, "name"),
    "encoder_config": ("gensid.config", EncoderConfig, "name"),
    "graph_building_config": ("gensid.config", GraphBuildingConfig, "graph_type"),
    "dataset_config": ("gensid.config", DatasetConfig, "name"),
    "trainer_config": ("gensid.config", TrainerConfig, "name"),
    "dataset": ("gensid.data", TorchDataset, "name"),
}


class Registry:
    def __init__(self, name):
        self.name = name
        self._entries: dict[str, type] = {}
        self._configs: dict[type, type] = {}

    def add(self, name, cls, config_cls=None):
        if name in self._entries:
            raise KeyError(f"{name}: already present in the registry with value {self._entries[name]}")
        self._entries[name] = cls
        if config_cls is not None:
            self._configs[cls] = config_cls

    def get(self, name):
        if name not in self._entries:
            raise KeyError(f"{self.name}: unknown {name!r}; available: {sorted(self._entries)}")
        return self._entries[name]

    def get_config(self, name):
        if self._entries[name] not in self._configs:
            raise KeyError(f"{self.name}: no config available for {name!r}")
        return self._configs[self._entries[name]]

    def keys(self):
        return sorted(self._entries)


class _Hub:
    """Self-building registry. Generates get_<group> and get_<group>_config."""

    def __init__(self):
        self._groups: dict[str, Registry] = {}
        self._discovered = False

    def _ensure(self):
        if not self._discovered:
            discover(self._groups)
            self._discovered = True

    def __getattr__(self, attr: str):
        if not attr.startswith("get_"):
            raise AttributeError(attr)
        self._ensure()
        breakpoint
        target = attr[len("get_") :]
        if target in self._groups:  # exact group
            reg = self._groups[target]
            return lambda name: reg.get(name)
        if target.endswith("_config") and target[: -len("_config")] in self._groups:
            reg = self._groups[target[: -len("_config")]]  # paired config
            return lambda name: reg.get_config(name)
        raise AttributeError(f"No registry group for {attr!r}; available: {sorted(self._groups)}")

    def keys(self, group: str):
        self._ensure()
        return self._groups[group].keys()


REGISTRY = _Hub()


def config_from_signature(cls) -> type | None:
    """Find the ModelConfig-typed parameter in cls.__init__ (unwraps Optional/Union)."""
    try:
        hints = get_type_hints(cls.__init__)
    except Exception:
        hints = {}
    for hint in hints.values():
        for candidate in (hint, *getattr(hint, "__args__", ())):
            origin = get_origin(candidate)
            if origin in (Union, getattr(__import__("types"), "UnionType", None)):
                for arg in getattr(candidate, "__args__", ()):
                    if isinstance(arg, type) and issubclass(arg, _CONFIGS):
                        return arg
            elif isinstance(candidate, type) and issubclass(candidate, _CONFIGS):
                return candidate
    return None


def iter_subclasses(package_name: str, base_cls, name_attr: str):
    """Yield (name, cls) for every concrete subclass of base_cls in a package."""
    package = importlib.import_module(package_name)
    for module_info in pkgutil.iter_modules(package.__path__, package_name + "."):
        module = importlib.import_module(module_info.name)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj is base_cls or not issubclass(obj, base_cls):
                continue
            if obj.__module__ != module.__name__:
                continue  # only classes defined directly in this package's modules
            name = getattr(obj, name_attr, None)
            if name is None:
                continue  # abstract / unnamed classes are not registered
            yield name, obj


def discover(groups):
    for name, (pkg, base, attr) in GROUPS.items():
        reg = groups[name] = Registry(name)
        for key, cls in iter_subclasses(pkg, base, attr):
            reg.add(key, cls, config_from_signature(cls))
    # ``iter_subclasses`` deliberately skips the base classes themselves; register the
    # stock Trainer/TrainerConfig under their conventional "base" name so a SID method
    # can default to them by declaring ``trainer = "base"``.
    groups["trainer"].add("base", Trainer)
    groups["trainer_config"].add("base", TrainerConfig)
