import numpy as np
import pandas as pd
import pytest

from empanada.mito_morphometrics import (
    MitoMorphometricsConfig,
    analyze_instance,
    analyze_labels,
    analyze_labels_stack,
)
from empanada_napari._mito_metrics_widget import mito_metrics_widget


def _draw_line(shape, row, col_start, col_end, width=3):
    mask = np.zeros(shape, dtype=bool)
    for c in range(col_start, col_end):
        for dr in range(-(width // 2), width // 2 + 1):
            r = row + dr
            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                mask[r, c] = True
    return mask


def _draw_disk(shape, center, radius):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    cy, cx = center
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2


def _draw_ring(shape, center, outer_radius, inner_radius):
    outer = _draw_disk(shape, center, outer_radius)
    inner = _draw_disk(shape, center, inner_radius)
    return outer & ~inner


class TestMitoMorphometrics:
    def test_widget_constructs_with_help_label(self):
        widget = mito_metrics_widget()

        assert "Quantifies mitochondrial shape" in widget.help_head.label

    def test_analyze_tubular_instance(self):
        mask = _draw_line((40, 60), row=20, col_start=5, col_end=55, width=5)
        config = MitoMorphometricsConfig(pixel_size_nm=2.0)
        metrics = analyze_instance(mask, label_id=1, config=config)

        assert metrics["label_id"] == 1
        assert metrics["volume_px"] > 0
        assert metrics["skeleton_length_px"] > 20
        assert metrics["count_tube"] >= 1
        assert metrics["fraction_tube_volume"] > 0.3
        assert metrics["mean_tube_radius_px"] > 0

    def test_analyze_sheet_like_instance(self):
        mask = _draw_disk((50, 50), center=(25, 25), radius=12)
        config = MitoMorphometricsConfig(
            pixel_size_nm=1.0,
            sheet_solidity_threshold=0.8,
            sheet_elongation_threshold=2.0,
        )
        metrics = analyze_instance(mask, label_id=2, config=config)

        assert metrics["volume_px"] > 0
        assert metrics["sphericity"] > 0.5
        assert metrics["fraction_sheet_volume"] + metrics["fraction_tube_volume"] <= 1.01

    def test_analyze_donut_instance(self):
        mask = _draw_ring((60, 60), center=(30, 30), outer_radius=18, inner_radius=10)
        config = MitoMorphometricsConfig(pixel_size_nm=1.0)
        metrics = analyze_instance(mask, label_id=3, config=config)

        assert metrics["volume_px"] > 0
        assert metrics["count_donut"] >= 1 or metrics["fraction_donut_volume"] > 0.05

    def test_analyze_labels_table_and_neighbors(self):
        labels = np.zeros((50, 50), dtype=np.uint32)
        labels[_draw_disk((50, 50), (15, 15), 8)] = 1
        labels[_draw_disk((50, 50), (15, 17), 8)] = 2
        labels[_draw_line((50, 50), 40, 10, 40, width=3)] = 3

        config = MitoMorphometricsConfig(pixel_size_nm=1.0, neighbor_contact_radius_nm=5.0)
        df = analyze_labels(labels, config=config)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert set(df["label_id"]) == {1, 2, 3}
        touching = df.set_index("label_id").loc[[1, 2], "num_neighbors"]
        assert touching.min() >= 1

    def test_analyze_labels_stack(self):
        stack = np.zeros((2, 30, 30), dtype=np.uint32)
        stack[0][_draw_disk((30, 30), (15, 15), 6)] = 1
        stack[1][_draw_line((30, 30), 15, 5, 25, width=3)] = 2

        df = analyze_labels_stack(stack, config=MitoMorphometricsConfig())
        assert len(df) == 2
        assert set(df["slice"]) == {0, 1}

    def test_empty_labels_returns_empty_dataframe(self):
        labels = np.zeros((20, 20), dtype=np.uint32)
        df = analyze_labels(labels)
        assert df.empty

    def test_nanotunnel_heuristic_on_thin_segment(self):
        # Very thin but long segment
        mask = _draw_line((20, 80), row=10, col_start=5, col_end=75, width=1)
        config = MitoMorphometricsConfig(
            pixel_size_nm=1.0,
            nanotunnel_max_radius_nm=2.0,
            nanotunnel_min_length_nm=10.0,
        )
        metrics = analyze_instance(mask, label_id=4, config=config)
        assert metrics["count_nanotunnel"] >= 1 or metrics["fraction_nanotunnel_volume"] > 0.1
