import numpy as np
import pytest
import torch

from gensid.eval import (
    check_id_distribution,
    id_collision_rate,
    id_gini_coefficient,
    id_perplexity_ratio,
    id_utilization_rate,
)


class TestIdUtilizationRate:
    @pytest.mark.parametrize(
        ("unique_ids", "n_ids", "expected"),
        [
            ([0, 1, 2, 3], 4, 1.0),  # all IDs in a 4-slot space used
            ([2], 5, 0.2),  # only a single ID used
            ([1, 2], 4, 0.5),  # partial usage
        ],
    )
    def test_basic_cases(self, unique_ids, n_ids, expected):
        assert id_utilization_rate(unique_ids, n_ids=n_ids) == pytest.approx(expected)

    def test_accepts_numpy_array(self):
        assert id_utilization_rate(np.array([10, 11, 12]), n_ids=10) == pytest.approx(0.3)


class TestIdPerplexityRatio:
    @pytest.mark.parametrize(
        ("counts", "n_ids", "expected"),
        [
            ([5, 5, 5, 5], 4, 1.0),  # uniform -> entropy log(4), ratio 1
            ([20], 4, 0.25),  # single ID -> perplexity 1, ratio 1/4
            (np.array([1, 1]), 2, 1.0),  # numpy input, uniform
        ],
    )
    def test_basic_cases(self, counts, n_ids, expected):
        assert id_perplexity_ratio(counts, n_ids=n_ids) == pytest.approx(expected)

    def test_skewed_but_non_trivial(self):
        counts = [8, 1, 1]
        probs = np.array([0.8, 0.1, 0.1])
        entropy = -(probs * np.log(probs)).sum()
        assert id_perplexity_ratio(counts, n_ids=3) == pytest.approx(np.exp(entropy) / 3)


class TestIdCollisionRate:
    # NOTE: id_collision_rate receives UNIQUE ids (not raw ids) plus the total item count.
    @pytest.mark.parametrize(
        ("unique_ids", "n_items", "expected"),
        [
            ([0, 1, 2], 3, 0.0),  # every id unique
            ([0], 4, 0.75),  # 1 unique among 4
            ([0, 1], 3, 1 / 3),  # 2 unique among 3
            ([], 5, 1.0),  # no distinct ids at all
        ],
    )
    def test_basic_cases(self, unique_ids, n_items, expected):
        assert id_collision_rate(unique_ids, n_items=n_items) == pytest.approx(expected)


class TestIdGiniCoefficient:
    @pytest.mark.parametrize(
        ("counts", "n_ids", "expected"),
        [
            ([1, 1, 1, 1], 4, 0.0),  # perfect equality
            ([1, 0, 0], 3, 2 / 3),  # all mass on a single id
            ([1, 2, 3], 3, 4 / 18),  # known unbalanced case -> (2*14 - 24)/18
            (np.array([1, 1]), 2, 0.0),  # numpy input, equality
        ],
    )
    def test_basic_cases(self, counts, n_ids, expected):
        assert id_gini_coefficient(counts, n_ids=n_ids) == pytest.approx(expected, abs=1e-9)


class TestCheckIdDistribution:
    @pytest.mark.parametrize(
        ("ids", "max_n_ids", "expected"),
        [
            ([[0], [1], [2], [3]], [4], {"utilization_rate": 1.0, "collision": 0.0, "perplexity": 1.0, "gini": 0.0}),
            ([[0], [0], [2]], [3], {"utilization_rate": 2 / 3, "collision": 1 / 3}),
        ],
    )
    def test_single_codebook(self, ids, max_n_ids, expected):
        metrics = check_id_distribution(torch.tensor(ids), n_codebooks=1, max_n_ids=max_n_ids, verbose=False)
        for key, value in expected.items():
            assert metrics[key] == pytest.approx(value)

    def test_single_codebook_has_no_per_codebook_keys(self):
        metrics = check_id_distribution(torch.tensor([[0], [1]]), n_codebooks=1, max_n_ids=[2], verbose=False)
        assert "cb0_" not in metrics

    def test_accepts_numpy_input(self):
        metrics = check_id_distribution(np.array([[0], [1]]), n_codebooks=1, max_n_ids=[2], verbose=False)
        assert metrics["utilization_rate"] == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("ids", "n_codebooks", "max_n_ids", "expected_key"),
        [
            ([[0, 0], [0, 1], [1, 0], [1, 1]], 2, [2, 2], "cb1_gini"),
            ([[0, 0], [0, 1], [1, 0], [1, 1]], 2, [2, 2], "cb0_perplexity"),
        ],
    )
    def test_multiple_codebooks_include_per_codebook_metrics(self, ids, n_codebooks, max_n_ids, expected_key):
        metrics = check_id_distribution(torch.tensor(ids), n_codebooks=n_codebooks, max_n_ids=max_n_ids, verbose=False)
        assert metrics["cb0_utilization_rate"] == pytest.approx(1.0)
        assert metrics["cb1_utilization_rate"] == pytest.approx(1.0)
        assert metrics["utilization_rate"] == pytest.approx(1.0)
        assert metrics[expected_key] >= 0.0

    def test_multiple_codebooks_shifted_whole_ids(self):
        # (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3 : distinct whole ids
        ids = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        metrics = check_id_distribution(ids, n_codebooks=2, max_n_ids=[2, 2], verbose=False)
        assert metrics["utilization"] == pytest.approx(1.0)

    def test_verbose_logs_do_not_crash(self):
        metrics = check_id_distribution(torch.tensor([[0], [1]]), n_codebooks=1, max_n_ids=[2], verbose=True)
        assert metrics["collision"] == pytest.approx(0.0)
