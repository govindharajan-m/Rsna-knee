"""I render saved montage images for manual preprocessing inspection."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .preprocessing import ProcessedVolume
from .volume import VolumeReconstruction


def save_preprocessing_montage(
    reconstruction: VolumeReconstruction,
    processed: ProcessedVolume,
    output_path: str | Path,
    *,
    representative_slices: int = 5,
) -> Path:
    """I save a montage of raw, normalized, resampled, global, and local MRI views."""

    if reconstruction.raw_volume is None:
        raise ValueError("I cannot render QC for an unusable reconstruction")
    if representative_slices < 1:
        raise ValueError("representative_slices must be positive")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = reconstruction.raw_volume
    normalized = processed.normalization.volume
    resampled = processed.global_volume
    local = processed.local_volume
    raw_indices = _slice_indices(raw.shape[0], representative_slices)
    processed_indices = _slice_indices(resampled.shape[0], representative_slices)
    figure, axes = plt.subplots(representative_slices, 5, figsize=(15, 3 * representative_slices))
    axes = np.atleast_2d(axes)
    titles = ("original", "normalized", "resampled", "candidate ROI", "global + ROI")
    for row_index, (raw_index, processed_index) in enumerate(zip(raw_indices, processed_indices, strict=True)):
        images = (
            raw[raw_index],
            normalized[min(raw_index, normalized.shape[0] - 1)],
            resampled[processed_index],
            local[min(processed_index, local.shape[0] - 1)],
            resampled[processed_index],
        )
        for column_index, image in enumerate(images):
            axis = axes[row_index, column_index]
            axis.imshow(image, cmap="gray")
            axis.set_axis_off()
            if row_index == 0:
                axis.set_title(titles[column_index])
        y_start, y_end, x_start, x_end = processed.roi.bounding_box_zyx[2:]
        axes[row_index, 4].plot(
            [x_start, x_end, x_end, x_start, x_start],
            [y_start, y_start, y_end, y_end, y_start],
            color="lime",
            linewidth=1,
        )
    figure.suptitle(
        "KneeScope-12 preprocessing QC | "
        f"ROI={processed.roi.quality_flag} | ordering={reconstruction.ordering.strategy}"
    )
    figure.tight_layout()
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return destination


def _slice_indices(depth: int, count: int) -> tuple[int, ...]:
    """I choose evenly spaced slice indices, including valid behavior for shallow volumes."""

    if depth < 1:
        raise ValueError("I cannot select montage slices from an empty volume")
    return tuple(int(index) for index in np.linspace(0, depth - 1, min(depth, count))) + tuple(
        depth - 1 for _ in range(max(0, count - depth))
    )
