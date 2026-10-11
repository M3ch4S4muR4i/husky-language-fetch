"""Shape classification of color blobs: box vs cylinder vs ball.

The color detector finds *which* pixels are red; this module splits a color mask
into connected components (one per object) and classifies each component's shape
from its silhouette and its depth profile:

* silhouette: aspect ratio (height/width in pixels), extent (area / bbox area) and
  how much narrower the top and bottom rows are than the middle (a ball tapers, a
  box or cylinder does not);
* depth (back-projected 3D points of the mask): metric height of the top above the
  floor, metric width across the view, and the curvature of the depth profile along
  the middle row (a box face is flat or a V at a corner; a cylinder or ball bulges
  toward the camera).

The classifier is a nearest-centroid model on standardized features. Its
parameters (``SHAPE_MODEL``) were fit by ``scripts/eval_language.py shapes`` on rendered
views from training seeds only and are evaluated on separate seeds.
Components touching the image border or with too few pixels are not classified
(``None``): their silhouette is truncated.
"""

from __future__ import annotations

from collections import deque

import numpy as np

SHAPES = ("box", "cylinder", "ball")
FEATURES = ("aspect", "extent", "taper", "height_m", "width_m", "curv_mm", "flat_top")


def connected_components(mask: np.ndarray, min_pixels: int = 1) -> list[np.ndarray]:
    """4-connected components of a boolean mask (largest first), as boolean masks."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    comps: list[np.ndarray] = []
    ys, xs = np.nonzero(mask)
    n = 0
    for y0, x0 in zip(ys.tolist(), xs.tolist(), strict=True):
        if labels[y0, x0]:
            continue
        n += 1
        labels[y0, x0] = n
        q = deque([(y0, x0)])
        pix = []
        while q:
            y, x = q.popleft()
            pix.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                v, u = y + dy, x + dx
                if 0 <= v < h and 0 <= u < w and mask[v, u] and not labels[v, u]:
                    labels[v, u] = n
                    q.append((v, u))
        if len(pix) >= min_pixels:
            comps.append(labels == n)
    comps.sort(key=lambda m: -int(m.sum()))
    return comps


def touches_border(mask: np.ndarray, margin: int = 1) -> bool:
    return bool(
        mask[:margin].any()
        or mask[-margin:].any()
        or mask[:, :margin].any()
        or mask[:, -margin:].any()
    )


def shape_features(mask: np.ndarray, frame=None) -> dict[str, float] | None:
    """Feature dict for one component, or None if it is too small/truncated."""
    ys, xs = np.nonzero(mask)
    if ys.size < 20:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    hpx, wpx = y1 - y0 + 1, x1 - x0 + 1
    if hpx < 5 or wpx < 4:
        return None
    rows = mask[y0 : y1 + 1, x0 : x1 + 1].sum(axis=1).astype(np.float32)
    mid = float(np.median(rows[len(rows) // 3 : 2 * len(rows) // 3 + 1]))
    k = max(1, len(rows) // 6)
    ends = float(np.mean(np.concatenate([rows[:k], rows[-k:]])))
    f = {"aspect": hpx / wpx, "extent": float(ys.size) / (hpx * wpx), "taper": ends / max(mid, 1.0)}
    if frame is None:
        f |= {"height_m": 0.0, "width_m": 0.0, "curv_mm": 0.0, "flat_top": 0.0}
        return f
    from fetchbot.perception.projection import backproject

    pts = backproject(frame, xs, ys)
    z = pts[:, 2]
    f["height_m"] = float(np.percentile(z, 98))
    lateral = pts @ frame.right
    f["width_m"] = float(np.percentile(lateral, 98) - np.percentile(lateral, 2))
    # Fraction of points within 1.5 cm of the top: large for flat tops (box, cylinder).
    f["flat_top"] = float(np.mean(z > f["height_m"] - 0.015))
    # Depth profile along the middle third of rows: quadratic vs linear fit residual.
    band = (ys >= y0 + hpx // 3) & (ys <= y0 + 2 * hpx // 3)
    d = pts[band] @ frame.forward  # distance along the optical axis
    lat = lateral[band]
    curv = 0.0
    if d.size >= 6 and np.ptp(lat) > 1e-3:
        lc = lat - lat.mean()
        a1 = np.polyfit(lc, d, 1)
        a2 = np.polyfit(lc, d, 2)
        r1 = np.sqrt(np.mean((np.polyval(a1, lc) - d) ** 2))
        r2 = np.sqrt(np.mean((np.polyval(a2, lc) - d) ** 2))
        # Signed: positive if the middle bulges toward the camera (cylinder/ball);
        # a box corner (V) also bulges, but a V is fit worse by the parabola.
        curv = 1000.0 * (r1 - r2) * (1.0 if a2[0] > 0 else -1.0)
    f["curv_mm"] = float(curv)
    return f


#: Nearest-centroid parameters (feature means/stds per class); filled in from the
#: training run of scripts/eval_language.py shapes (train seeds 1000-1029, 1264 views).
#: Held-out seeds 5000-5019: 95.9% of 869 single views.
SHAPE_MODEL: dict | None = {
    "features": ["aspect", "extent", "taper", "height_m", "width_m", "curv_mm", "flat_top"],
    "mean": [1.06947, 0.83855, 0.61914, 0.15102, 0.13887, 10.1409, 0.22731],
    "std": [0.20577, 0.07598, 0.20219, 0.00759, 0.02704, 5.53603, 0.07535],
    "centroids": {
        "box": [-0.317, 0.2782, 0.148, -0.5127, 0.8148, 0.1967, 0.6118],
        "cylinder": [1.8596, 0.7424, 1.0416, 2.1918, -1.173, -0.528, -0.0542],
        "ball": [-0.3822, -0.7499, -0.6945, -0.2452, -0.6662, -0.049, -0.878],
    },
}


class ShapeClassifier:
    """Nearest-centroid classifier on standardized :data:`FEATURES`."""

    def __init__(self, model: dict | None = None, max_dist: float = 3.5) -> None:
        self.model = model if model is not None else SHAPE_MODEL
        self.max_dist = max_dist

    @staticmethod
    def fit(samples: list[tuple[dict, str]]) -> dict:
        X = np.array([[f[k] for k in FEATURES] for f, _ in samples], dtype=np.float64)
        y = [s for _, s in samples]
        mu, sd = X.mean(axis=0), np.maximum(X.std(axis=0), 1e-3)
        Z = (X - mu) / sd
        cents = {}
        for s in SHAPES:
            idx = [i for i, lab in enumerate(y) if lab == s]
            if idx:
                cents[s] = Z[idx].mean(axis=0).round(4).tolist()
        return {
            "features": list(FEATURES),
            "mean": mu.round(5).tolist(),
            "std": sd.round(5).tolist(),
            "centroids": cents,
        }

    def predict(self, feats: dict | None) -> str | None:
        if feats is None or self.model is None:
            return None
        m = self.model
        z = (np.array([feats[k] for k in m["features"]]) - np.array(m["mean"])) / np.array(m["std"])
        best, bd = None, float("inf")
        for s, c in m["centroids"].items():
            d = float(np.linalg.norm(z - np.array(c)))
            if d < bd:
                best, bd = s, d
        return best if bd <= self.max_dist else None
