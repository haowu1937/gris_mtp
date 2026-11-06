import numpy as np
import pytest

from gensid.generation.utils import (
    build_codes,
    collision_stats,
    format_prefix,
    process_cluster_label_levels,
    make_contiguous_per_level,
    prefix_letter_from_position,
)


class TestPrefixLetterFromPosition:
    @pytest.mark.parametrize(
        ("pos", "letter"),
        [
            (0, "a"),
            (1, "b"),
            (2, "c"),
            (25, "z"),  # last pure letter; chr() based mapping
        ],
    )
    def test_position_to_letter(self, pos, letter):
        assert prefix_letter_from_position(pos) == letter


class TestFormatPrefix:
    @pytest.mark.parametrize(
        ("pos", "index", "expected"),
        [
            (0, 3, "<a_3>"),
            (1, 17, "<b_17>"),
            (2, 0, "<c_0>"),
        ],
    )
    def test_renders_letter_style_token(self, pos, index, expected):
        assert format_prefix(pos=pos, index=index) == expected


class TestLoadClusterLabelLevels:
    def test_single_vector_root(self):
        names, arrays = process_cluster_label_levels([0, 1, 0, 2])
        assert names == ["level_1"]
        np.testing.assert_array_equal(arrays[0], np.array([0, 1, 0, 2]))

    def test_list_of_vectors_root(self):
        names, arrays = process_cluster_label_levels([[0, 1], [2, 3], [4, 5]])
        assert names == ["level_1", "level_2", "level_3"]
        np.testing.assert_array_equal(arrays[1], np.array([2, 3]))

    def test_flat_dict_auto_detected(self):
        names, _ = process_cluster_label_levels({"level_1": [0, 1], "level_2": [2, 3]})
        assert names == ["level_1", "level_2"]

    def test_grouped_dict_selects_requested_group(self):
        payload = {
            "coarse_to_fine_item_labels": {"level_1": [0, 1], "level_2": [2, 3]},
            "other_group": {"level_1": [9, 9]},
        }
        names, arrays = process_cluster_label_levels(payload)
        assert names == ["level_1", "level_2"]
        np.testing.assert_array_equal(arrays[0], np.array([0, 1]))

    def test_auto_selects_only_candidate_group(self):
        names, _ = process_cluster_label_levels(
            {"fine_labels": {"level_1": [0, 1], "level_2": [2, 3]}}, label_group="nonexistent"
        )
        assert names == ["level_1", "level_2"]

    def test_ambiguous_group_raises(self):
        payload = {"g1": {"level_1": [0, 1]}, "g2": {"level_1": [2, 3]}}
        with pytest.raises(ValueError, match="Could not select label group"):
            process_cluster_label_levels(payload, label_group="nope")

    def test_levels_subset_with_int_normalization(self):
        payload = {"level_1": [0, 1], "level_2": [2, 3], "level_10": [4, 5]}
        names, arrays = process_cluster_label_levels(payload, levels=["level_10", 1])
        assert names == ["level_10", "level_1"]
        np.testing.assert_array_equal(arrays[0], np.array([4, 5]))

    def test_requested_level_missing_raises(self):
        with pytest.raises(ValueError, match="Requested level 'level_2' not found"):
            process_cluster_label_levels({"level_1": [0, 1]}, levels=["level_2"])

    def test_non_1d_level_raises(self):
        with pytest.raises(ValueError, match="must be 1-D"):
            process_cluster_label_levels({"level_1": [[0, 1], [2, 3]]})

    def test_mismatched_level_lengths_raise(self):
        with pytest.raises(ValueError, match="must have the same length"):
            process_cluster_label_levels({"level_1": [0, 1], "level_2": [2, 3, 4]})


class TestMakeContiguousPerLevel:
    def test_remaps_ids_to_compact_range(self):
        labels = [np.array([5, 1, 1, 5]), np.array([0, 2, 3, 2])]
        remapped, maps = make_contiguous_per_level(labels)
        np.testing.assert_array_equal(remapped[0], np.array([1, 0, 0, 1]))
        np.testing.assert_array_equal(remapped[1], np.array([0, 1, 2, 1]))
        assert maps[0] == {1: 0, 5: 1}

    def test_keeps_negative_labels_as_minus_one(self):
        labels = [np.array([0, -1, 2, -1])]
        remapped, _ = make_contiguous_per_level(labels, keep_negative=True)
        np.testing.assert_array_equal(remapped[0], np.array([0, -1, 1, -1]))

    def test_relocates_negative_labels_after_valid_when_not_kept(self):
        labels = [np.array([0, -1, 2])]
        remapped, maps = make_contiguous_per_level(labels, keep_negative=False)
        np.testing.assert_array_equal(remapped[0], np.array([1, 3, 2]))
        assert maps[0] == {0: 1, 2: 2, -1: 3}


class TestBuildCodes:
    def test_builds_letter_style_tokens_per_item(self):
        labels = [np.array([0, 1, 0]), np.array([2, 3, 2])]
        codes = build_codes(labels, unclustered_strategy="special", append_unique_leaf=False, unique_prefix="<x_{}>")
        assert codes == {
            "0": ["<a_0>", "<b_2>"],
            "1": ["<a_1>", "<b_3>"],
            "2": ["<a_0>", "<b_2>"],
        }

    def test_unclustered_special_uses_minus_one(self):
        labels = [np.array([0, -1])]
        codes = build_codes(labels, unclustered_strategy="special", append_unique_leaf=False, unique_prefix="<x_{}>")
        assert codes["1"] == ["<a_-1>"]

    def test_unclustered_unique_uses_item_id(self):
        labels = [np.array([0, -1])]
        codes = build_codes(labels, unclustered_strategy="unique", append_unique_leaf=False, unique_prefix="<x_{}>")
        assert codes["1"] == ["<a_1>"]

    def test_unclustered_error_raises(self):
        labels = [np.array([0, -1])]
        with pytest.raises(ValueError, match="unclustered label"):
            build_codes(labels, unclustered_strategy="error", append_unique_leaf=False, unique_prefix="<x_{}>")

    def test_unknown_strategy_raises(self):
        # the strategy branch is only reached when an unclustered (-1) label is present
        with pytest.raises(ValueError, match="Unknown unclustered_strategy"):
            build_codes([np.array([-1])], "bogus", append_unique_leaf=False, unique_prefix="<x_{}>")

    def test_empty_levels_raises(self):
        with pytest.raises(ValueError, match="No label levels were loaded"):
            build_codes([], "special", append_unique_leaf=False, unique_prefix="<x_{}>")

    def test_append_unique_leaf_ranks_duplicate_paths(self):
        labels = [np.array([0, 1, 0])]
        codes = build_codes(labels, unclustered_strategy="special", append_unique_leaf=True, unique_prefix="<x_{}>")
        assert codes == {"0": ["<a_0>", "<x_0>"], "1": ["<a_1>", "<x_0>"], "2": ["<a_0>", "<x_1>"]}

    def test_custom_prefixes_override_letter_style(self):
        labels = [np.array([0, 1])]
        codes = build_codes(
            labels,
            unclustered_strategy="special",
            append_unique_leaf=False,
            unique_prefix="<x_{}>",
            prefixes=["<a_{}>"],
        )
        assert codes == {"0": ["<a_0>"], "1": ["<a_1>"]}


class TestCollisionStats:
    def test_no_collisions(self):
        stats = collision_stats({"0": ["<a_0>"], "1": ["<a_1>"]})
        assert stats["num_items"] == 2
        assert stats["num_unique_sids"] == 2
        assert stats["num_colliding_items"] == 0
        assert stats["collision_rate"] == 0.0
        assert stats["max_collision_group_size"] == 1

    def test_counts_colliding_items(self):
        stats = collision_stats({"0": ["<a_0>"], "1": ["<a_0>"], "2": ["<a_1>"], "3": ["<a_1>"], "4": ["<a_2>"]})
        assert stats["num_unique_sids"] == 3
        assert stats["num_colliding_items"] == 4
        assert stats["collision_rate"] == pytest.approx(2 / 5)
        assert stats["max_collision_group_size"] == 2

    def test_empty_codes(self):
        stats = collision_stats({})
        assert stats == {
            "num_items": 0,
            "num_unique_sids": 0,
            "num_colliding_items": 0,
            "collision_rate": 0.0,
            "max_collision_group_size": 0,
        }
