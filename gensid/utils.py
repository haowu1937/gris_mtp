import argparse
import datetime
import json
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def check_collision(all_indices):
    return all_indices.shape[0] == np.unique(all_indices, axis=0).shape[0]


def get_collision_item(all_indices):
    index2id = {}
    for i, index in enumerate(all_indices):
        index = str(index)
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = [idx_group for idx_group in index2id.values() if len(idx_group) > 1]

    return collision_item_groups


def relabel_compact(labels: np.ndarray) -> np.ndarray:
    mask = labels >= 0
    out = np.full_like(labels, -1)
    out[mask] = np.unique(labels[mask], return_inverse=True)[1]
    return out


def yaml_to_argv(config: dict, parser: argparse.ArgumentParser | None = None) -> list[str]:
    """Flatten a YAML config dict into CLI argv (argparse-compatible).

    Keys become ``--key`` flags (underscores preserved; argparse maps them to the
    matching dest). ``None`` values emit nothing. Booleans emit a bare ``--key``
    for ``True`` and a ``--no_key`` negation flag for ``False``. When a parser is
    given, values equal to the flag's default are skipped and booleans are emitted
    in the form the parser knows (e.g. a default-True ``store_false`` flag like
    ``--no_directed`` is emitted only for ``False``, never for ``True``). Lists
    expand to repeated values. Later argv entries win over earlier ones, so real
    CLI arguments override the YAML file.

    Args:
        config (dict): Flat mapping of arg name -> value.
        parser (argparse.ArgumentParser | None): Optional parser used to resolve
            boolean defaults and flag forms.

    Returns:
        list[str]: Argument vector to prepend to ``sys.argv[1:]``.
    """
    defaults: dict[str, object] = {}
    if parser is not None:
        defaults = {action.dest: action.default for action in parser._actions if action.dest}

    argv: list[str] = []
    for key, value in config.items():
        if value is None:
            continue

        if isinstance(value, bool):
            if key in defaults and value == defaults[key]:
                continue
            argv.append(f"--{key}" if value else f"--no_{key}")
            continue

        argument = f"--{key}"
        if isinstance(value, (list, tuple)):
            argv.append(argument)
            argv.extend(map(str, value))
            continue

        argv.extend([argument, str(value)])

    return argv


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def set_color(log, color, highlight=True):
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except Exception:  # noqa: BLE001
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()  # noqa: DTZ005
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur


def get_device(device_arg: str = "auto") -> torch.device:
    """
    Determine the appropriate torch device based on the argument and availability.

    Priority:
    1. If device_arg is 'npu' and NPU is available → use 'npu'
    2. Else if device_arg is 'cuda' and CUDA is available → use 'cuda'
    3. Else if device_arg is 'cpu' → use 'cpu'
    4. Else if device_arg is 'auto':
       - Try NPU if available
       - Else try CUDA if available
       - Else fall back to CPU

    Args:
        device_arg: Device string from command-line argument (e.g., 'npu', 'cuda', 'cpu', 'auto')

    Returns:
        torch.device: The selected device.
    """
    device_arg = device_arg.lower()

    # Check NPU availability (Huawei Ascend)
    npu_available = hasattr(torch, "npu") and torch.npu.is_available()

    # Check CUDA availability (NVIDIA GPU)
    cuda_available = torch.cuda.is_available()

    if device_arg == "npu":
        if npu_available:
            return torch.device("npu")
        else:
            raise RuntimeError(
                "NPU requested but not available. Please install CANN and ensure NPU is properly configured."
            )

    elif device_arg == "cuda":
        if cuda_available:
            return torch.device("cuda")
        else:
            raise RuntimeError(
                "CUDA requested but not available. Please ensure CUDA is properly installed and configured."
            )

    elif device_arg == "cpu":
        return torch.device("cpu")

    elif device_arg == "auto":
        # Auto-detect: prefer NPU, then CUDA, then CPU
        if npu_available:
            return torch.device("npu")
        elif cuda_available:
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        raise ValueError(f"Unknown device argument: '{device_arg}'. Use 'npu', 'cuda', 'cpu', or 'auto'.")


def get_trained_flag(metrics_path: str) -> bool:
    if not os.path.exists(metrics_path):
        return False
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    return metrics.get("fully_trained", False)


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class DistTrainingState:
    """Distributed-training bookkeeping (mirrors the prod ``main.py`` dataclass).

    ``device`` is the per-rank device string (e.g. ``cuda:1`` / ``npu:0``),
    ``rank``/``local_rank``/``world_size`` come from the ``torchrun`` env.
    """

    device: str = "npu:0"
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1


def is_distributed() -> bool:
    """True when launched under ``torchrun`` with more than one process."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1


def init_dist_state(device: str) -> DistTrainingState:
    """Initialize the process group from ``torchrun`` env vars (prod ``main.py``).

    Backend is ``nccl`` for CUDA and ``hccl`` for NPU; the returned state carries
    the per-rank device string. Must be called on every rank before training.
    """
    addr = os.getenv("MASTER_ADDR")
    port = os.getenv("MASTER_PORT")
    rank_id = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device_type = str(device).split(":", 1)[0]
    backend = {"cuda": "nccl", "npu": "hccl"}.get(device_type, "gloo")
    dist.init_process_group(
        backend,
        init_method=f"tcp://{addr}:{port}",
        rank=rank_id,
        world_size=world_size,
    )
    return DistTrainingState(
        device=f"{device_type}:{local_rank}",
        rank=rank_id,
        local_rank=local_rank,
        world_size=world_size,
    )


def wrap_model_for_ddp(model: torch.nn.Module, dist_state: DistTrainingState) -> torch.nn.Module:
    """Wrap ``model`` in DDP; CPU fallback wraps without ``device_ids``."""
    if dist_state.device.startswith(("cuda", "npu")):
        return DDP(model, device_ids=[dist_state.local_rank], broadcast_buffers=False)
    return DDP(model, broadcast_buffers=False)
