"""ColorDetector on synthetic images and pixel back-projection."""

import unittest

import numpy as np

from fetchbot.data import CameraFrame
from fetchbot.perception import ColorDetector, backproject, localize, rgb_to_hsv
from fetchbot.perception.shape import (
    FEATURES,
    ShapeClassifier,
    connected_components,
    shape_features,
)


class TestColorDetector(unittest.TestCase):
    def test_hsv(self):
        hsv = rgb_to_hsv(np.array([[[255, 0, 0], [0, 0, 255], [128, 128, 128]]], np.uint8))
        self.assertAlmostEqual(float(hsv[0, 0, 0]), 0.0)
        self.assertAlmostEqual(float(hsv[0, 1, 0]), 240.0)
        self.assertAlmostEqual(float(hsv[0, 2, 1]), 0.0)

    def test_detects_colored_blob(self):
        img = np.full((60, 80, 3), 200, np.uint8)
        img[20:30, 40:50] = (210, 20, 20)  # red
        img[5:8, 5:8] = (20, 20, 220)  # tiny blue blob below min_pixels
        dets = ColorDetector(min_pixels=12).detect(img, ["red box", "blue", "green ball"])
        self.assertEqual([d.label for d in dets], ["red box"])
        self.assertEqual(dets[0].bbox_xyxy, (40.0, 20.0, 49.0, 29.0))

    def test_backproject_center_pixel(self):
        depth = np.full((4, 4), 2.0, np.float32)
        frame = CameraFrame(
            np.zeros((4, 4, 3), np.uint8),
            depth,
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            2.0,
            2.0,
            0.0,
        )
        p = backproject(frame, np.array([1]), np.array([1]))[0]
        self.assertAlmostEqual(p[0], 2.0)
        self.assertAlmostEqual(p[1], 0.5)  # left of center -> +y
        self.assertAlmostEqual(p[2], 1.5)
        mask = np.ones((4, 4), bool)
        from fetchbot.types import Detection

        det = localize(Detection("red", 1.0, (0, 0, 3, 3)), mask, frame)
        self.assertAlmostEqual(det.map_pose.x, 2.0)


class TestShape(unittest.TestCase):
    def test_connected_components_split_instances(self):
        m = np.zeros((40, 40), bool)
        m[2:10, 2:10] = True
        m[20:35, 25:30] = True
        m[0, 39] = True  # single noise pixel
        comps = connected_components(m, min_pixels=5)
        self.assertEqual(sorted(int(c.sum()) for c in comps), [64, 75])

    def test_two_same_color_objects_give_two_detections(self):
        img = np.full((60, 80, 3), 200, np.uint8)
        img[20:32, 5:17] = (210, 20, 20)
        img[20:32, 50:62] = (210, 20, 20)
        dets = ColorDetector(min_pixels=12).detect(img, ["red"])
        self.assertEqual(len(dets), 2)

    def test_silhouette_features(self):
        box = np.zeros((40, 40), bool)
        box[5:30, 10:30] = True
        yy, xx = np.mgrid[:40, :40]
        disc = (yy - 20) ** 2 + (xx - 20) ** 2 <= 12**2
        fb, fd = shape_features(box), shape_features(disc)
        self.assertAlmostEqual(fb["extent"], 1.0, places=3)
        self.assertLess(fd["extent"], 0.85)  # a circle fills ~pi/4 of its box
        self.assertGreater(fb["taper"], fd["taper"])  # a disc is narrow at top and bottom
        self.assertIsNone(shape_features(np.zeros((40, 40), bool)))

    def test_nearest_centroid_classifier(self):
        def f(**kw):
            return {k: kw.get(k, 0.0) for k in FEATURES}

        rng = np.random.default_rng(0)
        samples = []
        for _ in range(20):
            samples.append((f(extent=1.0 + 0.01 * rng.normal(), taper=1.0), "box"))
            samples.append((f(extent=0.78 + 0.01 * rng.normal(), taper=0.4), "ball"))
        clf = ShapeClassifier(ShapeClassifier.fit(samples))
        self.assertEqual(clf.predict(f(extent=0.99, taper=0.95)), "box")
        self.assertEqual(clf.predict(f(extent=0.79, taper=0.45)), "ball")
        self.assertIsNone(clf.predict(None))


if __name__ == "__main__":
    unittest.main()
