"""
Tests for the SfM keyframe selector (backend/services/keyframe_sfm.py). Pure numpy: no video needed.

Run from the repo root:  python -m unittest discover -s backend/tests -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from services.keyframe_sfm import (detect_scene_cuts, overlap_fraction, relative_sharpness_gate,  # noqa: E402
                                   select_sfm_keyframes)
from services.video_processor import DEFAULT_KEYFRAME_MODE, KEYFRAME_MODES, SFM_PRESETS, sfm_overlaps  # noqa: E402

W, H = 1000.0, 500.0


def pan(n, dx_per_frame, fail=()):
    """n frames of a camera panning by dx pixels/frame: transform i-1 -> i moves content by -dx."""
    t = np.array([[1, 0, -dx_per_frame], [0, 1, 0]], dtype=np.float64)
    return [None if (i in fail) else t.copy() for i in range(n)]


class OverlapTests(unittest.TestCase):
    def test_identity_is_full_overlap_and_disjoint_is_zero(self):
        self.assertAlmostEqual(overlap_fraction(np.eye(3), W, H), 1.0)
        shift = np.array([[1, 0, -2 * W], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        self.assertAlmostEqual(overlap_fraction(shift, W, H), 0.0)

    def test_overlap_of_a_horizontal_shift(self):
        shift = np.array([[1, 0, -250], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        self.assertAlmostEqual(overlap_fraction(shift, W, H), 0.75, places=4)

    def test_forward_flight_zoom_reduces_overlap_in_both_directions(self):
        # regression: flying forward zooms the scene in; the old "share of the current frame that the earlier
        # frame covers" stayed ~100% for any zoom-in, so no keyframe was ever due on a forward-flying FPV clip
        cx, cy = W / 2, H / 2
        def zoom(s):
            return np.array([[s, 0, cx - s * cx], [0, s, cy - s * cy], [0, 0, 1]], dtype=np.float64)
        self.assertAlmostEqual(overlap_fraction(zoom(1.0), W, H), 1.0)
        self.assertAlmostEqual(overlap_fraction(zoom(1.25), W, H), 1 / 1.25 ** 2, places=3)   # zoom in
        self.assertAlmostEqual(overlap_fraction(zoom(0.8), W, H), 0.8 ** 2, places=3)          # zoom out
        self.assertLess(overlap_fraction(zoom(1.25), W, H), 0.7)

    def test_degenerate_transform_counts_as_no_overlap(self):
        broken = np.array([[np.inf, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        self.assertEqual(overlap_fraction(broken, W, H), 0.0)

    def test_zoom_and_rotation_are_handled(self):
        c, s = np.cos(0.1), np.sin(0.1)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        ov = overlap_fraction(rot, W, H)
        self.assertTrue(0.6 < ov < 1.0)


class SelectionTests(unittest.TestCase):
    def test_spacing_follows_the_overlap_band(self):
        # 2% of the width per frame: overlap with the previous keyframe = 1 - 0.02*k
        # -> a keyframe is due at k=8 (0.84) and must be chosen by k=17 (0.66)
        n = 200
        res = select_sfm_keyframes(pan(n, 0.02 * W), [1.0] * n, W, H)
        kfs = res["keyframes"]
        gaps = np.diff(kfs)
        self.assertEqual(kfs[0], 0)
        self.assertTrue(((gaps >= 8) & (gaps <= 17)).all(), gaps)
        for ov in res["overlaps"][1:]:
            self.assertTrue(0.65 <= ov <= 0.85, ov)

    def test_faster_motion_gives_denser_keyframes(self):
        n = 120
        slow = select_sfm_keyframes(pan(n, 0.01 * W), [1.0] * n, W, H)["keyframes"]
        fast = select_sfm_keyframes(pan(n, 0.03 * W), [1.0] * n, W, H)["keyframes"]
        self.assertGreater(len(fast), 2 * len(slow) - 2)

    def test_picks_the_sharpest_frame_in_the_window(self):
        n = 60
        sharp = [1.0] * n
        sharp[13] = 50.0                      # inside the first window (frames 8..17)
        kfs = select_sfm_keyframes(pan(n, 0.02 * W), sharp, W, H)["keyframes"]
        self.assertEqual(kfs[1], 13)

    def test_blur_run_does_not_open_a_gap(self):
        # a long blurry stretch: the old absolute cutoff would drop all of it; overlap-based selection
        # still yields a keyframe every <= 17 frames (the sharpest of the blurry ones)
        n = 150
        sharp = [100.0] * n
        for i in range(40, 100):
            sharp[i] = 5.0 + (i % 5)
        kfs = select_sfm_keyframes(pan(n, 0.02 * W), sharp, W, H)["keyframes"]
        self.assertLessEqual(np.diff(kfs).max(), 17)

    def test_static_camera_yields_a_single_keyframe(self):
        n = 50
        ident = [np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64) for _ in range(n)]
        self.assertEqual(select_sfm_keyframes(ident, [1.0] * n, W, H)["keyframes"], [0])

    def test_tracking_failures_force_a_keyframe(self):
        n = 40
        res = select_sfm_keyframes(pan(n, 0.001 * W, fail={10, 11, 12}), [1.0] * n, W, H)
        self.assertGreater(len(res["keyframes"]), 1)
        self.assertLessEqual(res["keyframes"][1], 12)

    def test_extreme_motion_terminates_and_stays_ordered(self):
        n = 30
        kfs = select_sfm_keyframes(pan(n, 0.9 * W), [1.0] * n, W, H)["keyframes"]
        self.assertEqual(kfs, sorted(set(kfs)))
        self.assertEqual(kfs[-1], n - 1)      # every frame is needed: they barely overlap

    def test_forward_flight_zoom_yields_keyframes(self):
        # 1% zoom per frame and almost no lateral motion, like flying straight into a canyon
        n = 200
        z = np.array([[1.01, 0, W / 2 * (1 - 1.01)], [0, 1.01, H / 2 * (1 - 1.01)]], dtype=np.float64)
        kfs = select_sfm_keyframes([z.copy() for _ in range(n)], [1.0] * n, W, H)["keyframes"]
        self.assertGreater(len(kfs), 8)
        self.assertLessEqual(np.diff(kfs).max(), 40)

    def test_black_fade_in_is_skipped(self):
        n = 100
        sharp = [0.0, 0.5, 1.0, 1.5, 2.0] + [100.0] * (n - 5)
        tr = pan(n, 0.02 * W, fail=set(range(0, 6)))            # no features until frame 6
        kfs = select_sfm_keyframes(tr, sharp, W, H)["keyframes"]
        self.assertGreaterEqual(kfs[0], 5)
        self.assertTrue(all(k >= 5 for k in kfs))
        self.assertEqual(select_sfm_keyframes(pan(20, 0.02 * W), [100.0] * 20, W, H)["keyframes"][0], 0)

    def test_empty_and_invalid_arguments(self):
        self.assertEqual(select_sfm_keyframes([], [], W, H)["keyframes"], [])
        with self.assertRaises(ValueError):
            select_sfm_keyframes(pan(5, 1), [1.0] * 5, W, H, target_overlap=0.5, min_overlap=0.7)


class RelativeSharpnessTests(unittest.TestCase):
    def test_blurry_frame_is_dropped_regardless_of_absolute_scale(self):
        base = [100.0] * 40
        base[20] = 20.0
        for scale in (0.05, 1.0, 40.0):    # tiny, normal and huge Laplacian variances (other resolution / content)
            kept = relative_sharpness_gate([v * scale for v in base])
            self.assertNotIn(20, kept)
            self.assertEqual(len(kept), 39)


class SceneCutTests(unittest.TestCase):
    def test_finds_hard_cuts_but_not_fades_or_fast_motion(self):
        rng = np.random.default_rng(0)
        diffs = list(3.6 + rng.normal(0, 0.8, 300).clip(-2, 4))
        diffs[0] = 0.0
        for i, v in {50: 71.0, 120: 30.0, 200: 24.0}.items():   # the cut strengths measured on a real FPV clip
            diffs[i] = v
        diffs[150] = 11.3                                        # fade to black / whip pan: not a cut
        self.assertEqual(detect_scene_cuts(diffs), [50, 120, 200])

    def test_no_cuts_in_a_calm_clip_and_degenerate_inputs(self):
        self.assertEqual(detect_scene_cuts([0.0] + [3.0] * 100), [])
        self.assertEqual(detect_scene_cuts([]), [])
        self.assertEqual(detect_scene_cuts([0.0, 99.0]), [])     # too short to judge

    def test_threshold_scales_with_a_busy_clip(self):
        busy = [0.0] + [12.0] * 200                              # very shaky footage: normal change is 12
        busy[100] = 40.0                                         # above 15 but not 5x the median
        self.assertEqual(detect_scene_cuts(busy), [])
        busy[100] = 90.0
        self.assertEqual(detect_scene_cuts(busy), [100])


class PresetTests(unittest.TestCase):
    def test_presets_are_valid_overlap_bands(self):
        for mode, (target, minimum) in SFM_PRESETS.items():
            self.assertTrue(0 < minimum < target <= 1, mode)
        self.assertIn(DEFAULT_KEYFRAME_MODE, KEYFRAME_MODES)
        self.assertIn("notebook", KEYFRAME_MODES)

    def test_dense_preset_is_denser_than_light(self):
        n = 400
        dense = select_sfm_keyframes(pan(n, 0.01 * W), [1.0] * n, W, H, *SFM_PRESETS["sfm"])["keyframes"]
        light = select_sfm_keyframes(pan(n, 0.01 * W), [1.0] * n, W, H, *SFM_PRESETS["sfm_light"])["keyframes"]
        self.assertGreater(len(dense), 2.5 * len(light))

    def test_explicit_overlaps_override_the_preset(self):
        self.assertEqual(sfm_overlaps("sfm"), SFM_PRESETS["sfm"])
        self.assertEqual(sfm_overlaps("sfm", target_overlap=0.9), (0.9, SFM_PRESETS["sfm"][1]))
        self.assertEqual(sfm_overlaps("sfm_light", min_overlap=0.5), (SFM_PRESETS["sfm_light"][0], 0.5))


if __name__ == "__main__":
    unittest.main()
