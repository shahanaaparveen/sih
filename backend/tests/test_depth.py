import os
import sys
import unittest

import numpy as np

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (os.path.join(BACKEND, "services"), os.path.join(BACKEND, "stage04")):
    if p not in sys.path:
        sys.path.insert(0, p)

import depth_pipeline as dp  # noqa: E402


def _make_case(depth_like: bool, n: int = 160, seed: int = 0):
    """A synthetic pred map + sparse (uv, z). depth_like: pred ~ z (larger=farther);
    else pred ~ 1/z (disparity, larger=nearer)."""
    rng = np.random.default_rng(seed)
    coords = rng.integers(5, 250, size=(n, 2))          # integer (u, v) -> bilinear sample is exact
    z = rng.uniform(2.0, 25.0, size=n)
    val = z.copy() if depth_like else (1.0 / z)
    val = val * (1 + rng.normal(0, 0.01, n))            # mild noise
    pred = np.zeros((256, 256), np.float32)
    for (u, v), x in zip(coords, val):
        pred[v, u] = x
    return pred, coords.astype(float), z


class TestDepthAlignment(unittest.TestCase):
    def test_detects_depth_like_output(self):
        pred, uv, z = _make_case(depth_like=True)
        fit = dp.align_relative(pred, uv, z)
        self.assertIsNotNone(fit)
        self.assertEqual(fit["mode"], "depth")
        self.assertLess(fit["holdout_rel_err_median"], 0.1)

    def test_detects_disparity_like_output(self):
        pred, uv, z = _make_case(depth_like=False)
        fit = dp.align_relative(pred, uv, z)
        self.assertIsNotNone(fit)
        self.assertEqual(fit["mode"], "disp")
        self.assertLess(fit["holdout_rel_err_median"], 0.1)


if __name__ == "__main__":
    unittest.main()
