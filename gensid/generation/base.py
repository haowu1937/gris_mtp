"""Encode driver: turn generated integer indices into SID JSON + stats sidecar."""

import json
import logging

import numpy as np

from gensid.eval import check_id_distribution
from gensid.generation.utils import format_prefix

logger = logging.getLogger(__name__)


def save_generation_mapping(indices: np.ndarray, output_file: str) -> None:
    """Write {item_id: [token, ...]} JSON (the gensid artifact contract)."""
    mapping = {str(i): row.tolist() for i, row in enumerate(indices)}
    with open(output_file, "w") as fp:
        json.dump(mapping, fp)


def collect_indices(batches, generate_fn, n_codebooks, all_indices=None, all_indices_str=None, prefixes=None):
    """Fold per-batch SID indices into (all_indices, all_indices_str) buffers.

    batches: iterable of (positions, batch). positions is None to append new rows,
        or an index array locating the rows of ``batch`` in the buffers (used by the
        collision-fixing pass, which overwrites those rows in place).
    generate_fn: callable(batch) -> ``model.generate`` output, shape [n, depth].
    prefixes (list[str] or None, optional): Per-level prefix format strings, e.g.
        ``["<a_{}>", "<b_{}>"]``. When None, the default letter-based
        ``format_prefix`` is used. Defaults to None.

    Returns (all_indices, all_indices_str). The buffers are created on the first call
    and can be passed back to run a second, collision-fixing pass.
    """
    if all_indices is None:
        all_indices, all_indices_str = [], []
        append = True
    else:
        append = False

    for positions, batch in batches:
        indices = generate_fn(batch)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        if append:
            all_indices.append(indices)
            all_indices_str.extend([[_format_code(i, int(v), prefixes) for i, v in enumerate(row)] for row in indices])
        else:
            for pos, index in zip(positions, indices):
                all_indices[pos] = index
                all_indices_str[pos][-1] = _format_code(n_codebooks - 1, int(index[-1]), prefixes)

    if append:
        return np.vstack(all_indices), np.array(all_indices_str)
    return all_indices, all_indices_str


def _format_code(level: int, index: int, prefixes: list[str] | None = None) -> str:
    if prefixes is not None:
        return prefixes[level].format(index)
    return format_prefix(level, index)


def report_and_save(indices, indices_str, n_codebooks, codebook_sizes, output_file):
    indices_str = np.asarray(indices_str)
    _, indices_count = np.unique(indices_str, return_counts=True, axis=0)
    logger.debug("[Generation] All indices number: %s", indices.shape[0])
    logger.debug("[Generation] Max number of conflicts: %s", indices_count.max())
    check_id_distribution(indices, n_codebooks, codebook_sizes, verbose=True)
    logger.info("Saved SID mapping for %d items to %s", indices.shape[0], output_file)
    save_generation_mapping(indices, output_file)
