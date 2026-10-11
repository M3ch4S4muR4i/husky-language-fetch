"""Detector hierarchy: one interface, swappable models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

from fetchbot.types import Detection


class Detector(ABC):
    """Detects objects matching text queries in an RGB image."""

    def __init__(self, score_threshold: float = 0.3) -> None:
        self.score_threshold = score_threshold

    @abstractmethod
    def detect(self, image: Any, queries: Sequence[str]) -> list[Detection]:
        """Return detections for ``queries`` above ``score_threshold``."""


class OpenVocabDetector(Detector):
    """Open-vocabulary detector (OWL-ViT, Grounding DINO, or an NVIDIA NIM endpoint).

    Kept as the planned GPU backend; not implemented in the PyBullet build.
    """

    def __init__(self, model_name: str, score_threshold: float = 0.3) -> None:
        super().__init__(score_threshold)
        self.model_name = model_name

    def load(self) -> None:
        """Load model weights or connect to the inference endpoint."""
        raise NotImplementedError

    def detect(self, image: Any, queries: Sequence[str]) -> list[Detection]:
        """Run text-conditioned detection."""
        raise NotImplementedError


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorized RGB (uint8, HxWx3) -> HSV with H in degrees [0, 360), S and V in [0, 1]."""
    x = rgb.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(axis=-1), x.min(axis=-1)
    delta = mx - mn
    safe = np.where(delta > 1e-6, delta, 1.0)
    h = (
        np.where(
            mx == r,
            ((g - b) / safe) % 6.0,
            np.where(mx == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0),
        )
        * 60.0
    )
    h = np.where(delta > 1e-6, h, 0.0)
    s = np.where(mx > 1e-6, delta / np.where(mx > 1e-6, mx, 1.0), 0.0)
    return np.stack([h, s, mx], axis=-1)


class ColorDetector(Detector):
    """HSV color-threshold detector.

    A query like ``"red box"`` is matched on its color word. All pixels whose
    hue lies in the color's range (with enough saturation and brightness) form
    the blob; its score grows with pixel count. Only colors in ``HUE_RANGES``
    are supported. Each connected blob is one detection, and its shape (box,
    cylinder, ball) is classified by :mod:`fetchbot.perception.shape`, so two
    objects of the same color are told apart.
    """

    #: Hue intervals in degrees (a color may wrap around 0, e.g. red).
    HUE_RANGES: dict[str, tuple[tuple[float, float], ...]] = {
        "red": ((0.0, 12.0), (345.0, 360.0)),
        "yellow": ((45.0, 70.0),),
        "green": ((90.0, 160.0),),
        "blue": ((205.0, 250.0),),
    }

    def __init__(
        self,
        score_threshold: float = 0.3,
        min_saturation: float = 0.55,
        min_value: float = 0.25,
        min_pixels: int = 12,
        classify_shapes: bool = True,
    ) -> None:
        super().__init__(score_threshold)
        self.min_saturation = min_saturation
        self.min_value = min_value
        self.min_pixels = min_pixels
        self.last_masks: dict[str, np.ndarray] = {}
        self.last_component_masks: list[np.ndarray] = []
        self.classify_shapes = classify_shapes
        self.shape_classifier = None

    @staticmethod
    def color_of(query: str) -> str | None:
        for word in query.lower().split():
            if word in ColorDetector.HUE_RANGES:
                return word
        return None

    def mask(self, hsv: np.ndarray, color: str) -> np.ndarray:
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        m = np.zeros(h.shape, dtype=bool)
        for lo, hi in self.HUE_RANGES[color]:
            m |= (h >= lo) & (h < hi)
        return m & (s >= self.min_saturation) & (v >= self.min_value)

    def detect(self, image: Any, queries: Sequence[str], frame=None) -> list[Detection]:
        """Detect blobs whose color matches a word in each query (e.g. 'red').

        Each 4-connected component of a color mask is a separate detection (two red
        objects in view give two detections), with ``shape`` classified from its
        silhouette and, if ``frame`` (the RGB-D :class:`CameraFrame`) is given, its
        depth profile. ``last_masks[label]`` is the union mask of a color and
        ``last_component_masks[i]`` the mask of the i-th returned detection.
        """
        from fetchbot.perception.shape import (
            ShapeClassifier,
            connected_components,
            shape_features,
            touches_border,
        )

        if self.shape_classifier is None:
            self.shape_classifier = ShapeClassifier()
        hsv = rgb_to_hsv(np.asarray(image))
        out: list[Detection] = []
        self.last_masks = {}
        self.last_component_masks = []
        for q in queries:
            color = self.color_of(q)
            if color is None:
                continue
            m = self.mask(hsv, color)
            if int(m.sum()) < self.min_pixels:
                continue
            for comp in connected_components(m, min_pixels=self.min_pixels):
                n = int(comp.sum())
                score = min(1.0, n / (4.0 * self.min_pixels))
                if score < self.score_threshold:
                    continue
                shape = None
                if self.classify_shapes and not touches_border(comp):
                    shape = self.shape_classifier.predict(shape_features(comp, frame))
                ys, xs = np.nonzero(comp)
                self.last_masks[q] = self.last_masks.get(q, np.zeros_like(m)) | comp
                self.last_component_masks.append(comp)
                out.append(
                    Detection(
                        label=q,
                        score=score,
                        bbox_xyxy=(
                            float(xs.min()),
                            float(ys.min()),
                            float(xs.max()),
                            float(ys.max()),
                        ),
                        shape=shape,
                    )
                )
        return out
