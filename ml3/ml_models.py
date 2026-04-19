from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ml3.models import RasterScene


@dataclass
class PixelLogisticModel:
    label_name: str
    feature_names: list[str]
    weights: np.ndarray
    bias: float
    threshold: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_name": self.label_name,
            "feature_names": self.feature_names,
            "weights": self.weights.tolist(),
            "bias": float(self.bias),
            "threshold": float(self.threshold),
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "PixelLogisticModel":
        return PixelLogisticModel(
            label_name=str(payload["label_name"]),
            feature_names=[str(name) for name in payload["feature_names"]],
            weights=np.asarray(payload["weights"], dtype=float),
            bias=float(payload["bias"]),
            threshold=float(payload.get("threshold", 0.5)),
        )


@dataclass
class MonitoringModelBundle:
    vegetation_model: PixelLogisticModel
    built_up_model: PixelLogisticModel

    def to_dict(self) -> dict[str, Any]:
        return {
            "vegetation_model": self.vegetation_model.to_dict(),
            "built_up_model": self.built_up_model.to_dict(),
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "MonitoringModelBundle":
        return MonitoringModelBundle(
            vegetation_model=PixelLogisticModel.from_dict(payload["vegetation_model"]),
            built_up_model=PixelLogisticModel.from_dict(payload["built_up_model"]),
        )


def train_monitoring_model_bundle(
    *,
    before_scene: RasterScene,
    after_scene: RasterScene,
    ndvi_threshold: float,
    ndbi_threshold: float,
    built_red_threshold: float,
    built_swir_threshold: float,
    max_samples_per_scene: int = 30000,
) -> MonitoringModelBundle:
    vegetation_model = _train_binary_model(
        label_name="vegetation",
        scenes=[before_scene, after_scene],
        target_builder=lambda scene: _heuristic_vegetation_target(scene, ndvi_threshold),
        max_samples_per_scene=max_samples_per_scene,
    )
    built_up_model = _train_binary_model(
        label_name="built_up",
        scenes=[before_scene, after_scene],
        target_builder=lambda scene: _heuristic_built_target(
            scene,
            ndvi_threshold=ndvi_threshold,
            ndbi_threshold=ndbi_threshold,
            built_red_threshold=built_red_threshold,
            built_swir_threshold=built_swir_threshold,
        ),
        max_samples_per_scene=max_samples_per_scene,
    )
    return MonitoringModelBundle(vegetation_model=vegetation_model, built_up_model=built_up_model)


def classify_with_model(scene: RasterScene, model: PixelLogisticModel) -> np.ndarray:
    x = _scene_features(scene)
    probabilities = _sigmoid(np.dot(x, model.weights) + model.bias)
    return probabilities.reshape(scene.shape) >= model.threshold


def save_monitoring_model_bundle(bundle: MonitoringModelBundle, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(bundle.to_dict(), indent=2), encoding="utf-8")
    return target


def load_monitoring_model_bundle(path: str | Path) -> MonitoringModelBundle:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return MonitoringModelBundle.from_dict(payload)


def _train_binary_model(
    *,
    label_name: str,
    scenes: list[RasterScene],
    target_builder,
    max_samples_per_scene: int,
) -> PixelLogisticModel:
    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []

    for scene in scenes:
        scene_features = _scene_features(scene)
        scene_target = target_builder(scene).reshape(-1).astype(float)
        valid = _scene_valid_mask(scene).reshape(-1)

        valid_indices = np.where(valid)[0]
        if valid_indices.size == 0:
            continue

        if valid_indices.size > max_samples_per_scene:
            sampled_indices = np.random.default_rng(42).choice(
                valid_indices,
                size=max_samples_per_scene,
                replace=False,
            )
        else:
            sampled_indices = valid_indices

        feature_parts.append(scene_features[sampled_indices])
        target_parts.append(scene_target[sampled_indices])

    if not feature_parts:
        raise ValueError(f"Unable to train {label_name} model because no valid samples were found.")

    x = np.vstack(feature_parts)
    y = np.concatenate(target_parts)
    weights, bias = _fit_logistic_regression(x, y)
    return PixelLogisticModel(
        label_name=label_name,
        feature_names=["blue", "green", "red", "nir", "swir", "ndvi", "ndbi", "brightness"],
        weights=weights,
        bias=bias,
        threshold=0.5,
    )


def _fit_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.2,
    epochs: int = 350,
    l2: float = 1e-3,
) -> tuple[np.ndarray, float]:
    mu = x.mean(axis=0)
    sigma = np.maximum(x.std(axis=0), 1e-6)
    x_normalized = (x - mu) / sigma

    weights = np.zeros(x.shape[1], dtype=float)
    bias = 0.0

    for _ in range(epochs):
        logits = np.dot(x_normalized, weights) + bias
        probs = _sigmoid(logits)
        error = probs - y

        grad_w = (x_normalized.T @ error) / x_normalized.shape[0] + l2 * weights
        grad_b = float(np.mean(error))

        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    # Fold normalization into the final weights so inference can run on raw features.
    folded_weights = weights / sigma
    folded_bias = bias - float(np.dot(mu / sigma, weights))
    return (folded_weights, folded_bias)


def _scene_features(scene: RasterScene) -> np.ndarray:
    blue = scene.blue
    green = scene.green
    red = scene.red
    nir = scene.nir
    swir = scene.swir if scene.swir is not None else np.zeros(scene.shape, dtype=float)
    ndvi = _normalized_difference(nir, red)
    ndbi = _normalized_difference(swir, nir)
    brightness = (blue + green + red + nir + swir) / 5.0

    stacked = np.stack([blue, green, red, nir, swir, ndvi, ndbi, brightness], axis=-1)
    return stacked.reshape(-1, stacked.shape[-1])


def _scene_valid_mask(scene: RasterScene) -> np.ndarray:
    valid = np.ones(scene.shape, dtype=bool)
    valid &= np.isfinite(scene.blue)
    valid &= np.isfinite(scene.green)
    valid &= np.isfinite(scene.red)
    valid &= np.isfinite(scene.nir)
    if scene.swir is not None:
        valid &= np.isfinite(scene.swir)
    if scene.valid_mask is not None:
        valid &= np.asarray(scene.valid_mask, dtype=bool)
    return valid


def _heuristic_vegetation_target(scene: RasterScene, ndvi_threshold: float) -> np.ndarray:
    ndvi = _normalized_difference(scene.nir, scene.red)
    return ndvi >= ndvi_threshold


def _heuristic_built_target(
    scene: RasterScene,
    *,
    ndvi_threshold: float,
    ndbi_threshold: float,
    built_red_threshold: float,
    built_swir_threshold: float,
) -> np.ndarray:
    ndvi = _normalized_difference(scene.nir, scene.red)
    if scene.swir is not None:
        ndbi = _normalized_difference(scene.swir, scene.nir)
        return (
            (ndbi >= ndbi_threshold)
            & (ndvi < ndvi_threshold * 0.55)
            & (scene.red >= built_red_threshold)
            & (scene.swir >= built_swir_threshold)
        )

    brightness = (scene.red + scene.green + scene.blue) / 3.0
    ndbi = _normalized_difference(brightness, scene.nir)
    brightness = (scene.red + scene.green + scene.blue + scene.nir) / 4.0
    return (ndbi >= ndbi_threshold) & (ndvi < ndvi_threshold * 0.55) & (brightness > 0.18)


def _normalized_difference(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return (numerator - denominator) / (numerator + denominator + 1e-6)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -40, 40)
    return 1.0 / (1.0 + np.exp(-clipped))
