from __future__ import annotations

from collections import deque

import numpy as np

from ml3.models import AffineTransform, ConnectedRegion


def extract_connected_regions(
    mask: np.ndarray,
    transform: AffineTransform,
    min_pixels: int = 1,
) -> list[ConnectedRegion]:
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    regions: list[ConnectedRegion] = []

    for row, col in np.argwhere(mask):
        if visited[row, col]:
            continue

        queue: deque[tuple[int, int]] = deque([(int(row), int(col))])
        visited[row, col] = True
        pixels: list[tuple[int, int]] = []

        while queue:
            current_row, current_col = queue.popleft()
            pixels.append((current_row, current_col))

            for row_offset in (-1, 0, 1):
                for col_offset in (-1, 0, 1):
                    if row_offset == 0 and col_offset == 0:
                        continue

                    next_row = current_row + row_offset
                    next_col = current_col + col_offset

                    if not (0 <= next_row < height and 0 <= next_col < width):
                        continue
                    if visited[next_row, next_col] or not mask[next_row, next_col]:
                        continue

                    visited[next_row, next_col] = True
                    queue.append((next_row, next_col))

        if len(pixels) < min_pixels:
            continue

        rows = np.array([pixel[0] for pixel in pixels], dtype=float)
        cols = np.array([pixel[1] for pixel in pixels], dtype=float)
        centroid_row = float(rows.mean())
        centroid_col = float(cols.mean())

        regions.append(
            ConnectedRegion(
                pixel_count=len(pixels),
                bbox_pixels=(
                    int(rows.min()),
                    int(cols.min()),
                    int(rows.max()),
                    int(cols.max()),
                ),
                centroid_pixel=(centroid_row, centroid_col),
                centroid_geo=transform.pixel_center_to_geo(centroid_row, centroid_col),
                area_sq_m=len(pixels) * transform.pixel_area_sq_m,
            )
        )

    return regions


def mask_outline(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return mask & ~interior
