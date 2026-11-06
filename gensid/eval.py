import logging
from typing import Iterable

import numpy as np
import torch

logger = logging.getLogger(__name__)


def check_id_distribution(
    semantic_ids: torch.Tensor | np.ndarray, n_codebooks: int, max_n_ids: torch.Tensor | list[int], verbose: bool = True
) -> dict[str, float]:
    """
    Function for checking distribution of semantic IDs relative to the possible semantic ID space.

    :param semantic_ids: tensor of semantic IDs
    :param n_codebooks: number of codebooks
    :param max_n_ids: maximum number of IDs supported by this ID space
    :param verbose: print all the results in addition to returning them
    :return: utilization rate, perplexity ratio, collision rate, Gini coefficient
    """
    metrics = {}
    if n_codebooks > 1:
        for i in range(n_codebooks):
            if verbose:
                logger.info(f"Checking distribution for ID {i + 1}")
            cb_metrics = check_id_distribution(semantic_ids[:, i], 1, [max_n_ids[i]], verbose=verbose)
            metrics |= {f"cb{i}_{k}": v for k, v in cb_metrics.items()}
        if verbose:
            logger.info("Checking distribution for whole-IDs")

    n_items = semantic_ids.shape[0]
    max_ids = int(np.prod(max_n_ids))
    ids_shifting = np.cumsum(np.insert(max_n_ids, 0, 0))[:-1]

    if isinstance(semantic_ids, torch.Tensor):
        semantic_ids = semantic_ids.cpu().numpy()
    shifted_numpy_ids = semantic_ids + ids_shifting

    unique_ids, counts = np.unique(shifted_numpy_ids, return_counts=True, axis=0)
    metrics["utilization_rate"] = id_utilization_rate(unique_ids, max_ids)
    metrics["perplexity"] = id_perplexity_ratio(counts, max_ids)
    metrics["collision"] = id_collision_rate(unique_ids, n_items)
    vocab2code_ratio = n_items / max_ids
    metrics["utilization"] = metrics["utilization_rate"] / vocab2code_ratio
    metrics["gini"] = id_gini_coefficient(counts, max_ids)
    if verbose:
        logger.info(f"\tUtilization rate: {metrics['utilization_rate']:.4f} (Optim: 1.0)")
        logger.info(f"\tVocab2Space ratio: {vocab2code_ratio:.4f}")
        logger.info(f"\tUtilization ratio: {metrics['utilization']:.4f}")
        logger.info(f"\tPerplexity ratio: {metrics['perplexity']:.4f} (Optim: 1.0)")
        logger.info(f"\tCollision rate: {metrics['collision']:.4f} (Optim: 0.0)")
        logger.info(f"\tGini coefficient: {metrics['gini']:.4f} (Optim: 0.0)")

    return metrics


def id_utilization_rate(unique_ids: Iterable[int], n_ids: int) -> float:
    """
    Calculates the utilization rate for the given IDs given the ID space defined by the number of IDs.
    Utilization rate for non-empty lists is in the range [1/n_ids, 1] where utilization rate = 1/n_ids if only a single
    ID is used, and 1 if all IDs are used.

    :param unique_ids: Iterable of unique IDs.
    :param n_ids: Total number of possible IDs in the ID space.
    :return: The utilization rate.
    """
    n_unique = len(unique_ids)
    utilization_rate = n_unique / n_ids

    return utilization_rate


def id_perplexity_ratio(counts: np.ndarray | list[int], n_ids: int) -> float:
    """
    Calculates the perplexity ratio for the given IDs given the ID space.
    Perplexity ratio is in the range [1/n_ids, 1] where perplexity ratio = 1/n_ids if only a single ID is used, and 1 if
    samples are uniformly distributed across all IDs.

    :param ids: Counts returned by numpy unique on the list of unique IDs.
    :param n_ids: Total number of possible IDs in the ID space.
    :return: Perplexity ratio.
    """
    counts = np.asarray(counts)
    # Calculate empirical probabilities
    probabilities = counts / counts.sum()
    # Calculate Shannon entropy
    entropy = -(probabilities * np.log(probabilities)).sum()
    # Calculate perplexity
    perplexity = np.exp(entropy)
    # Calculate perplexity ratio
    perplexity_ratio = perplexity / n_ids

    return perplexity_ratio


def id_collision_rate(unique_ids: Iterable[int], n_items: int) -> float:
    """
    Calculates the collision rate for the given IDs.
    The collision rate is the number of ID collisions per sample. Collision rate = 0 if every ID appears only once and
    1 - 1 / n_samples if all IDs are identical (max collisions).

    :param ids: Iterable of unique IDs.
    :param n_items: Total number of items encoded in the ID space.
    :return: Collision rate.
    """
    n_unique = len(unique_ids)
    collisions = n_items - n_unique
    collision_rate = collisions / n_items

    return collision_rate


def id_gini_coefficient(counts: np.ndarray | list[int], n_ids: int) -> float:
    """
    Calculates the Gini coefficient for the given IDs given the ID space.

    Comparing to the conventional example for the Gini coefficient of income or wealth distribution within a population,
    the population consists of the theoretical IDs and the income or wealth of each individual is the number of times
    that ID appears in the list of IDs.

    The implementation is now optimized to handle large ID spaces (N codebooks of size K need frequency of K^N).

    :param counts: Counts of IDs.
    :param n_ids: Total number of possible IDs in the ID space.
    :return: Gini coefficient.
    """
    counts = np.asarray(counts, dtype=np.float64)

    n_unique = len(counts)
    total = counts.sum()

    # Sort the non-zero counts in ascending order
    counts_sorted = np.sort(counts)

    # Create indices for the non-zero counts
    offset = n_ids - n_unique
    indices = np.arange(1, n_unique + 1, dtype=np.float64) + offset

    # Compute sum(i * x_i) for the non-zero parts, while zero elements are ignored
    sum_ix = np.dot(indices, counts_sorted)

    # Gini formula: (2 * sum(i * x_i) - (n + 1) * total) / (n * total)
    gini = (2.0 * sum_ix - (n_ids + 1) * total) / (n_ids * total)

    return float(gini)
