import os

import dask.array as da
import napari
import numpy as np
import pandas as pd
from magicgui import magicgui
from napari.layers import Labels
from napari.qt.threading import thread_worker
from napari_plugin_engine import napari_hook_implementation

from empanada.mito_morphometrics import MitoMorphometricsConfig, analyze_labels, analyze_labels_stack
from empanada_napari.utils import enable_layer_rename_refresh


def _as_numpy(array):
    if isinstance(array, da.Array):
        return np.asarray(array.compute())
    return np.asarray(array)


def _get_current_slice(viewer, labels_layer):
    labels = labels_layer.data
    cursor_pos = viewer.cursor.position

    if labels.ndim == 4:
        axis = tuple(viewer.dims.order[:2])
        plane = (
            int(labels_layer.world_to_data(cursor_pos)[axis[0]]),
            int(labels_layer.world_to_data(cursor_pos)[axis[1]]),
        )
        slices = [slice(None), slice(None), slice(None), slice(None)]
        slices[axis[0]] = plane[0]
        slices[axis[1]] = plane[1]
    elif labels.ndim == 3:
        axis = viewer.dims.order[0]
        plane = int(labels_layer.world_to_data(cursor_pos)[axis])
        slices = [slice(None), slice(None), slice(None)]
        slices[axis] = plane
    else:
        slices = [slice(None), slice(None)]
        plane = None

    return _as_numpy(labels[tuple(slices)]), plane


def _save_metrics(df: pd.DataFrame, save_dir: str, filename: str, export_csv: bool, export_xlsx: bool):
    os.makedirs(save_dir, exist_ok=True)
    saved = []
    if export_csv:
        csv_path = os.path.join(save_dir, f"{filename}.csv")
        df.to_csv(csv_path, index=False)
        saved.append(csv_path)
    if export_xlsx:
        xlsx_path = os.path.join(save_dir, f"{filename}.xlsx")
        df.to_excel(xlsx_path, index=False)
        saved.append(xlsx_path)
    return saved


def _preview_and_save(df, labels_name, filename, export_csv, export_xlsx, save_dir, folder_name):
    if df.empty:
        print("No labeled mitochondria found to analyze.")
        return

    print(f"Computed morphometrics for {len(df)} mitochondria.")
    preview_cols = [
        c
        for c in [
            "slice",
            "label_id",
            "volume_nm3",
            "volume_px",
            "skeleton_length_nm",
            "count_tube",
            "count_junction",
            "count_bulb",
            "count_nanotunnel",
            "count_donut",
            "fraction_tube_volume",
            "fraction_sheet_volume",
            "num_neighbors",
        ]
        if c in df.columns
    ]
    print(df[preview_cols].head(min(10, len(df))).to_string(index=False))

    if export_csv or export_xlsx:
        if not save_dir:
            print("Save directory is required to export metrics.")
            return
        out_dir = os.path.join(str(save_dir), folder_name)
        saved = _save_metrics(df, out_dir, filename, export_csv, export_xlsx)
        for path in saved:
            print(f"Saved metrics to {path}")


def mito_metrics_widget():
    apply_options = {
        "Current image": "Current image",
        "2D patches": "2D patches",
        "3D volume or z-stack": "3D volume or z-stack",
    }

    @magicgui(
        call_button="Compute MitoMetrics (see terminal)",
        layout="vertical",
        help_head=dict(
            widget_type="Label",
            label=(
                '<p>Quantifies mitochondrial shape, structure fractions (tube, sheet, '
                'junction, fusion, bulb, nanotunnel, donut), tube geometry, and local '
                'features per label. Exports a metrics table (CSV/XLSX).</p>'
                '<p><b>Tip:</b> start with "Current image" on one slice. Full 3D can take '
                'a long time on large volumes.</p>'
            ),
        ),
        apply_to=dict(
            widget_type="RadioButtons",
            choices=list(apply_options.keys()),
            value=list(apply_options.keys())[0],
            label="Apply to:",
            tooltip="Analyze the current slice, each patch in a stack, or the full 3D volume.",
        ),
        pixel_size_nm=dict(
            widget_type="FloatSpinBox",
            label="Pixel size (nm)",
            min=0.1,
            max=1000.0,
            value=1.0,
            step=0.1,
        ),
        z_step_nm=dict(
            widget_type="FloatSpinBox",
            label="Z step (nm, 3D only)",
            min=0.1,
            max=1000.0,
            value=1.0,
            step=0.1,
        ),
        nanotunnel_max_radius_nm=dict(
            widget_type="FloatSpinBox",
            label="Nanotunnel max radius (nm)",
            min=1.0,
            max=500.0,
            value=100.0,
            step=1.0,
        ),
        nanotunnel_min_length_nm=dict(
            widget_type="FloatSpinBox",
            label="Nanotunnel min length (nm)",
            min=1.0,
            max=5000.0,
            value=150.0,
            step=1.0,
        ),
        export_csv=dict(
            widget_type="CheckBox",
            value=True,
            label="Export metrics (.csv)",
        ),
        export_xlsx=dict(
            widget_type="CheckBox",
            value=False,
            label="Export metrics (.xlsx)",
        ),
        folder_name=dict(widget_type="LineEdit", value="mito_metrics", label="Folder name"),
        save_dir=dict(
            widget_type="FileEdit",
            value="",
            label="Save directory",
            mode="d",
            tooltip="Directory in which to save morphometrics tables.",
        ),
    )
    def widget(
        viewer: napari.viewer.Viewer,
        labels_layer: Labels,
        help_head: str,
        apply_to: str,
        pixel_size_nm: float,
        z_step_nm: float,
        nanotunnel_max_radius_nm: float,
        nanotunnel_min_length_nm: float,
        export_csv: bool,
        export_xlsx: bool,
        folder_name: str,
        save_dir: str,
    ):
        if labels_layer is None:
            print("Select a Labels layer first.")
            return

        if (export_csv or export_xlsx) and not save_dir:
            print("Choose a Save directory before computing (required for export).")
            return

        config = MitoMorphometricsConfig(
            pixel_size_nm=pixel_size_nm,
            z_step_nm=z_step_nm,
            nanotunnel_max_radius_nm=nanotunnel_max_radius_nm,
            nanotunnel_min_length_nm=nanotunnel_min_length_nm,
        )
        labels_name = labels_layer.name

        @thread_worker
        def _run():
            if apply_to == "Current image":
                if labels_layer.data.ndim > 2:
                    labels_data, plane = _get_current_slice(viewer, labels_layer)
                    print(f"Analyzing current slice/plane: {plane}, shape={labels_data.shape}")
                else:
                    labels_data = _as_numpy(labels_layer.data)
                    plane = "2d"
                    print(f"Analyzing 2D labels, shape={labels_data.shape}")
                df = analyze_labels(labels_data, config=config)
                filename = f"{labels_name}_slice_{plane}_mito_metrics"
                return df, filename

            print("Loading labels into memory (this can take a while for large volumes)...")
            labels_data = _as_numpy(labels_layer.data)
            print(f"Loaded labels with shape={labels_data.shape}")

            if apply_to == "2D patches":
                if labels_data.ndim != 3:
                    raise ValueError("2D patches mode expects a 3D labels stack (N, Y, X).")
                df = analyze_labels_stack(labels_data, config=config)
                filename = f"{labels_name}_patches_mito_metrics"
            else:
                if labels_data.ndim == 2:
                    df = analyze_labels(labels_data, config=config, compute_network_context=True)
                else:
                    print(
                        "Running full 3D morphometrics. This is slow for many labels; "
                        "consider 'Current image' first."
                    )
                    df = analyze_labels(labels_data, config=config, compute_network_context=False)
                filename = f"{labels_name}_volume_mito_metrics"
            return df, filename

        def _on_return(result):
            df, filename = result
            _preview_and_save(df, labels_name, filename, export_csv, export_xlsx, save_dir, folder_name)

        def _on_error(exc):
            print(f"MitoMetrics failed: {exc}")

        worker = _run()
        worker.returned.connect(_on_return)
        worker.errored.connect(_on_error)
        print("MitoMetrics started in background (watch this terminal for progress)...")
        worker.start()

    enable_layer_rename_refresh(widget)
    return widget


@napari_hook_implementation(specname="napari_experimental_provide_dock_widget")
def mito_metrics_dock_widget():
    return mito_metrics_widget, {"name": "MitoMetrics"}
