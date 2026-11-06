import json

import numpy as np
import torch

from gensid.generation.base import (
    collect_indices,
    report_and_save,
    save_generation_mapping,
)
from gensid.generation.utils import format_prefix


class TestSaveGenerationMapping:
    def test_writes_item_keyed_mapping_json(self, tmp_path):
        indices = np.array([[0, 3], [1, 17], [2, 0]])
        output_file = str(tmp_path / "index.json")
        save_generation_mapping(indices, output_file)
        with open(output_file) as fp:
            mapping = json.load(fp)
        assert mapping == {"0": [0, 3], "1": [1, 17], "2": [2, 0]}


class TestCollectIndices:
    def _scripted_generate(self, outputs):
        """Return a generate_fn yielding the next scripted [n, depth] output each call."""
        it = iter(outputs)

        def generate_fn(batch):
            return next(it)

        return generate_fn

    def test_append_pass_accumulates_all_batches(self):
        batches = [
            (None, np.zeros((2, 2))),
            (None, np.zeros((2, 2))),
        ]
        outputs = [torch.tensor([[0, 3], [1, 17]]), torch.tensor([[2, 0], [0, 1]])]
        all_indices, all_indices_str = collect_indices(batches, self._scripted_generate(outputs), n_codebooks=2)
        assert all_indices.shape == (4, 2)
        np.testing.assert_array_equal(all_indices, np.array([[0, 3], [1, 17], [2, 0], [0, 1]]))
        assert all_indices_str.shape == (4, 2)
        assert all_indices_str.tolist() == [
            ["<a_0>", "<b_3>"],
            ["<a_1>", "<b_17>"],
            ["<a_2>", "<b_0>"],
            ["<a_0>", "<b_1>"],
        ]

    def test_custom_prefixes_used_in_string_buffer(self):
        batches = [(None, np.zeros((2, 2)))]
        outputs = [torch.tensor([[0, 3], [1, 17]])]
        prefixes = ["<x_{}>", "<y_{}>"]
        _, all_indices_str = collect_indices(
            batches, self._scripted_generate(outputs), n_codebooks=2, prefixes=prefixes
        )
        assert all_indices_str.tolist() == [["<x_0>", "<y_3>"], ["<x_1>", "<y_17>"]]

    def test_collision_fix_pass_overwrites_rows_and_only_last_token_str(self):
        gen = self._scripted_generate([torch.tensor([[0, 3], [1, 17]]), torch.tensor([[2, 0], [0, 1]])])
        all_indices, all_indices_str = collect_indices(
            [(None, np.zeros((2, 2))), (None, np.zeros((2, 2)))], gen, n_codebooks=2
        )

        fix_gen = self._scripted_generate([torch.tensor([[1, 99], [3, 42]])])
        all_indices, all_indices_str = collect_indices(
            [(np.array([1, 2]), np.zeros((2, 2)))],
            fix_gen,
            n_codebooks=2,
            all_indices=all_indices,
            all_indices_str=all_indices_str,
        )

        assert all_indices is not all_indices_str  # in-place pass, no new buffers
        np.testing.assert_array_equal(all_indices[1], np.array([1, 99]))
        np.testing.assert_array_equal(all_indices[2], np.array([3, 42]))
        np.testing.assert_array_equal(all_indices[0], np.array([0, 3]))
        np.testing.assert_array_equal(all_indices[3], np.array([0, 1]))
        # only the last-column token is rewritten in the string buffer for fixed rows
        assert all_indices_str[1].tolist() == ["<a_1>", "<b_99>"]
        assert all_indices_str[2].tolist() == ["<a_2>", "<b_42>"]
        assert all_indices_str[0].tolist() == ["<a_0>", "<b_3>"]
        assert all_indices_str[3].tolist() == ["<a_0>", "<b_1>"]


class TestReportAndSave:
    def test_writes_mapping_and_passes_distribution_check(self, tmp_path):
        indices = np.array([[0], [1], [2], [0]])
        indices_str = np.array(
            [[format_prefix(0, 0)], [format_prefix(0, 1)], [format_prefix(0, 2)], [format_prefix(0, 0)]]
        )
        output_file = str(tmp_path / "index.json")
        report_and_save(indices, indices_str, n_codebooks=1, codebook_sizes=[3], output_file=output_file)
        with open(output_file) as fp:
            mapping = json.load(fp)
        assert mapping == {"0": [0], "1": [1], "2": [2], "3": [0]}
