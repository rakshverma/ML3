from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ml3.geometry import mask_outline
from ml3.models import ComplianceReport, PremisesBoundary, RasterScene
from ml3.monitoring import MonitoringRunResult


def render_report_bundle(
    output_dir: str | Path,
    premises: PremisesBoundary,
    before_scene: RasterScene,
    after_scene: RasterScene,
    run_result: MonitoringRunResult,
) -> dict[str, str]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    before_image = _scene_to_image(before_scene)
    after_image = _scene_to_image(after_scene)

    before_overlay = _overlay_masks(
        before_image,
        premises.mask,
        vegetation_mask=run_result.before_vegetation_mask,
    )
    after_overlay = _overlay_masks(
        after_image,
        premises.mask,
        vegetation_mask=run_result.after_vegetation_mask,
        green_loss_mask=run_result.green_loss_mask,
        new_construction_mask=run_result.new_construction_mask,
    )
    _draw_alert_regions(after_overlay, run_result.report)

    before_path = target_dir / "before_annotated.png"
    after_path = target_dir / "after_annotated.png"
    comparison_path = target_dir / "comparison_panel.png"
    before_overlay.save(before_path)
    after_overlay.save(after_path)
    _compose_comparison_panel(before_overlay, after_overlay, run_result.report).save(comparison_path)

    return {
        "before_annotated_image": str(before_path),
        "after_annotated_image": str(after_path),
        "comparison_panel": str(comparison_path),
    }


def write_report_documents(report: ComplianceReport, output_dir: str | Path) -> dict[str, str]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    json_path = target_dir / "compliance_report.json"
    markdown_path = target_dir / "compliance_report.md"
    json_path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    markdown_path.write_text(_report_to_markdown(report), encoding="utf-8")

    return {
        "json_report": str(json_path),
        "markdown_report": str(markdown_path),
    }


def _scene_to_image(scene: RasterScene) -> Image.Image:
    stacked = np.dstack([scene.red, scene.green, scene.blue])
    stretched = _percentile_stretch(stacked)
    array = np.clip(stretched * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB").resize(
        (array.shape[1] * 2, array.shape[0] * 2),
        resample=Image.Resampling.NEAREST,
    )


def _percentile_stretch(rgb: np.ndarray) -> np.ndarray:
    low = np.percentile(rgb, 2, axis=(0, 1))
    high = np.percentile(rgb, 98, axis=(0, 1))
    stretched = (rgb - low) / np.maximum(high - low, 1e-6)
    return np.clip(stretched, 0.0, 1.0)


def _overlay_masks(
    base_image: Image.Image,
    premises_mask: np.ndarray,
    vegetation_mask: np.ndarray | None = None,
    green_loss_mask: np.ndarray | None = None,
    new_construction_mask: np.ndarray | None = None,
) -> Image.Image:
    scale = 2
    rgba = base_image.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    overlay_pixels = overlay.load()

    outline = mask_outline(premises_mask)
    vegetation_mask = vegetation_mask if vegetation_mask is not None else np.zeros_like(premises_mask)
    green_loss_mask = green_loss_mask if green_loss_mask is not None else np.zeros_like(premises_mask)
    new_construction_mask = (
        new_construction_mask if new_construction_mask is not None else np.zeros_like(premises_mask)
    )

    for row, col in np.argwhere(vegetation_mask):
        _paint_block(overlay_pixels, row, col, scale, (35, 180, 75, 70))
    for row, col in np.argwhere(green_loss_mask):
        _paint_block(overlay_pixels, row, col, scale, (255, 196, 61, 130))
    for row, col in np.argwhere(new_construction_mask):
        _paint_block(overlay_pixels, row, col, scale, (236, 74, 66, 150))
    for row, col in np.argwhere(outline):
        _paint_block(overlay_pixels, row, col, scale, (73, 200, 255, 255))

    return Image.alpha_composite(rgba, overlay)


def _paint_block(pixels, row: int, col: int, scale: int, color: tuple[int, int, int, int]) -> None:
    top = row * scale
    left = col * scale
    for y in range(top, top + scale):
        for x in range(left, left + scale):
            pixels[x, y] = color


def _draw_alert_regions(image: Image.Image, report: ComplianceReport) -> None:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for index, region in enumerate(report.new_construction_regions, start=1):
        row_min, col_min, row_max, col_max = region.bbox_pixels
        scale = 2
        draw.rectangle(
            [col_min * scale, row_min * scale, (col_max + 1) * scale, (row_max + 1) * scale],
            outline=(255, 72, 72, 255),
            width=2,
        )
        draw.text(
            (col_min * scale + 4, row_min * scale + 4),
            f"Alert {index}",
            font=font,
            fill=(255, 255, 255, 255),
        )


def _compose_comparison_panel(
    before_image: Image.Image,
    after_image: Image.Image,
    report: ComplianceReport,
) -> Image.Image:
    font = ImageFont.load_default()
    width = before_image.width + after_image.width
    height = max(before_image.height, after_image.height) + 70
    canvas = Image.new("RGB", (width, height), (15, 22, 33))
    canvas.paste(before_image.convert("RGB"), (0, 40))
    canvas.paste(after_image.convert("RGB"), (before_image.width, 40))

    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), f"{report.site_name} ({report.site_id})", font=font, fill=(255, 255, 255))
    draw.text((12, 24), f"Before: {report.before_date}", font=font, fill=(190, 220, 255))
    draw.text(
        (before_image.width + 12, 24),
        f"After: {report.after_date}",
        font=font,
        fill=(255, 220, 190),
    )
    return canvas


def _report_to_markdown(report: ComplianceReport) -> str:
    lines = [
        f"# Compliance Report: {report.site_name}",
        "",
        f"- Site ID: `{report.site_id}`",
        f"- Rule: {report.rule_name}",
        f"- Required green cover: {report.required_green_cover_pct:.2f}%",
        f"- Time window: {report.before_date} to {report.after_date}",
        "",
        "## Temporal Analysis",
        "",
        f"- Before green cover: {report.before_metrics.green_cover_pct:.2f}%",
        f"- After green cover: {report.after_metrics.green_cover_pct:.2f}%",
        f"- Delta: {report.green_cover_delta_pct_points:.2f} percentage points",
        f"- Green-loss area: {report.green_loss_area_sq_m:.2f} sq m",
        f"- New-construction area: {report.new_construction_area_sq_m:.2f} sq m",
        "",
        "## Alerts",
        "",
    ]

    if report.alerts:
        for alert in report.alerts:
            coordinate = (
                f" @ {alert.geo_coordinate[0]:.3f}, {alert.geo_coordinate[1]:.3f}"
                if alert.geo_coordinate
                else ""
            )
            lines.append(f"- [{alert.severity}] {alert.message}{coordinate}")
    else:
        lines.append("- No violations triggered in this run.")

    lines.extend(
        [
            "",
            "## Evidence",
            "",
        ]
    )
    for label, path in report.evidence_paths.items():
        lines.append(f"- {label}: `{path}`")

    return "\n".join(lines) + "\n"
