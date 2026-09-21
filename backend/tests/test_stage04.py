"""
Tests for the Stage 04 building blocks that do not need COLMAP, a GPU or a video.

Run from the repo root:  python -m unittest discover -s backend/tests -v
"""
import os
import sys
import tempfile
import unittest
import zipfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from services import colmap_io, stage04  # noqa: E402


def write_model(directory, n_images=3):
    """A tiny hand-written COLMAP text model: cameras translate along +X, all looking down +Z."""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "cameras.txt"), "w") as f:
        f.write("# cameras\n1 SIMPLE_RADIAL 800 450 650 400 225 0.0\n")
    with open(os.path.join(directory, "images.txt"), "w") as f:
        f.write("# images\n")
        for i in range(n_images):  # identity rotation, camera centre at x = i  ->  t = -C
            f.write(f"{i + 1} 1 0 0 0 {-float(i)} 0 0 1 keyframe_{i:04d}_original_{i * 3:04d}.jpg\n")
            f.write("100 100 1 500 200 -1\n" if i == 0 else "\n")
    with open(os.path.join(directory, "points3D.txt"), "w") as f:
        f.write("# points\n1 0 0 10 255 128 0 0.5 1 0\n")


class ColmapIoTests(unittest.TestCase):
    def test_parses_model_and_camera_centres(self):
        with tempfile.TemporaryDirectory() as d:
            write_model(d)
            m = colmap_io.load_model(d)
            self.assertEqual(len(m.images), 3)
            self.assertEqual(m.cameras[1].model, "SIMPLE_RADIAL")
            centres = [colmap_io.camera_center(im)[0] for im in sorted(m.images.values(), key=lambda i: i.name)]
            np.testing.assert_allclose(centres, [0, 1, 2])
            self.assertEqual(m.points3d[1].track_length, 1)

    def test_image_without_observations_does_not_desync_parsing(self):
        with tempfile.TemporaryDirectory() as d:
            write_model(d)
            m = colmap_io.load_model(d)
            self.assertEqual(len(m.images[1].xys), 2)
            self.assertEqual(len(m.images[2].xys), 0)
            self.assertEqual(m.images[3].name, "keyframe_0002_original_0006.jpg")

    def test_name_helpers(self):
        self.assertEqual(colmap_io.keyframe_index_from_name("images/keyframe_0007_original_0123.jpg"), 7)
        self.assertEqual(colmap_io.frame_index_from_name("keyframe_0007_original_0123.jpg"), 123)
        self.assertEqual(colmap_io.frame_index_from_name("fill_0002_original_0042.jpg"), 42)
        self.assertIsNone(colmap_io.keyframe_index_from_name("fill_0002_original_0042.jpg"))

    def test_sparse_depth_is_depth_in_camera_frame(self):
        with tempfile.TemporaryDirectory() as d:
            write_model(d)
            m = colmap_io.load_model(d)
            uv, z = colmap_io.sparse_depth_samples(m, m.images[1])   # point (0,0,10), camera at origin
            np.testing.assert_allclose(z, [10.0])
            np.testing.assert_allclose(uv, [[100, 100]])

    def test_similarity_align_recovers_scale_rotation_translation(self):
        rng = np.random.default_rng(1)
        src = rng.normal(size=(30, 3))
        ang = 0.7
        R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
        dst = 2.5 * src @ R.T + np.array([1.0, -2.0, 3.0])
        s, R2, t = colmap_io.similarity_align(src, dst)
        self.assertAlmostEqual(s, 2.5, places=6)
        np.testing.assert_allclose(R2, R, atol=1e-6)
        np.testing.assert_allclose((s * src @ R2.T) + t, dst, atol=1e-6)


class GapFillTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.frames_dir = self.tmp.name
        # one sharp frame (100) every 4th frame, everything else blurry (1): each search window
        # (about 4 frames wide) contains a sharp one, so the sharpest pick is unambiguous
        self.frames = [{"frame_index": i, "sharpness": 100.0 if i % 4 == 2 else 1.0} for i in range(200)]
        for f in self.frames:
            with open(os.path.join(self.frames_dir, f"frame_{f['frame_index']:04d}.jpg"), "wb"):
                pass

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def kfs(indices):
        return [{"keyframe_index": n, "frame_index": i} for n, i in enumerate(indices)]

    def test_no_fill_when_gaps_are_normal(self):
        self.assertEqual(stage04.plan_gap_fills(self.kfs(range(0, 100, 2)), self.frames, self.frames_dir), [])

    def test_fills_only_the_abnormal_gap_and_stays_inside_it(self):
        keyframes = self.kfs([0, 1, 2, 3, 4, 40, 41, 42, 43])           # hole between 4 and 40
        fills = stage04.plan_gap_fills(keyframes, self.frames, self.frames_dir)
        got = [f["frame_index"] for f in fills]
        self.assertTrue(len(got) >= 5)
        self.assertTrue(all(4 < f < 40 for f in got))
        self.assertEqual(len(got), len(set(got)))
        gaps = np.diff(sorted([0, 1, 2, 3, 4, 40, 41, 42, 43] + got))
        self.assertLessEqual(gaps.max(), 9)                              # hole is now bridged

    def test_picks_the_sharpest_frame_in_each_window(self):
        keyframes = self.kfs([0, 1, 2, 3, 4, 40, 41, 42, 43])
        fills = stage04.plan_gap_fills(keyframes, self.frames, self.frames_dir)
        self.assertTrue(fills)
        for fill in fills:
            self.assertEqual(fill["sharpness"], 100.0)
            self.assertEqual(fill["frame_index"] % 4, 2)

    def test_skips_frames_missing_on_disk_and_without_frame_data(self):
        keyframes = self.kfs([0, 1, 2, 3, 4, 40, 41])
        self.assertEqual(stage04.plan_gap_fills(keyframes, [], self.frames_dir), [])
        for name in os.listdir(self.frames_dir):
            os.remove(os.path.join(self.frames_dir, name))
        self.assertEqual(stage04.plan_gap_fills(keyframes, self.frames, self.frames_dir), [])


class MultiModelSummaryTests(unittest.TestCase):
    """An edited video reconstructs as several separate models; the report must say so, not blame the keyframes."""

    def summary_for(self, colmap_report, manifest):
        with tempfile.TemporaryDirectory() as d:
            results = os.path.join(d, "results")
            model = os.path.join(results, "model")
            write_model(model)
            import json
            with open(os.path.join(results, "colmap_report.json"), "w") as f:
                json.dump(colmap_report, f)
            return stage04.summarize(model, colmap_io.load_model(model), manifest, 3)

    def test_several_reconstructions_with_cuts_are_explained_and_judged_fairly(self):
        report = {"num_models": 5, "registered_any_fraction": 0.986, "mean_reprojection_error_px": 0.56,
                  "num_images_registered_any_model": 142, "warnings": ["COLMAP produced 5 disconnected reconstructions"]}
        manifest = {"keyframes": [{"keyframe_index": i} for i in range(3)], "scene_cuts": [440, 1027, 1140]}
        s = self.summary_for(report, manifest)
        self.assertEqual(s["verdict"], "fair")                    # not "poor": 98.6% registered somewhere
        self.assertEqual(s["images"]["registered_any_model"], 142)
        self.assertEqual(s["scene_cuts"], [440, 1027, 1140])
        self.assertTrue(any("expected" in w for w in s["warnings"]))

    def test_a_single_clean_reconstruction_is_good(self):
        report = {"num_models": 1, "mean_reprojection_error_px": 0.4, "warnings": []}
        manifest = {"keyframes": [{"keyframe_index": i} for i in range(3)]}
        self.assertEqual(self.summary_for(report, manifest)["verdict"], "good")

    def test_mostly_unregistered_is_poor(self):
        report = {"num_models": 3, "registered_any_fraction": 0.4, "mean_reprojection_error_px": 0.5, "warnings": []}
        manifest = {"keyframes": [{"keyframe_index": i} for i in range(3)]}
        self.assertEqual(self.summary_for(report, manifest)["verdict"], "poor")


class ImportSafetyTests(unittest.TestCase):
    def make_zip(self, path, files):
        with zipfile.ZipFile(path, "w") as z:
            for name, data in files.items():
                z.writestr(name, data)

    def test_rejects_zip_slip(self):
        with tempfile.TemporaryDirectory() as d:
            bad = os.path.join(d, "bad.zip")
            self.make_zip(bad, {"model/cameras.txt": "x", "../../escape.txt": "pwned"})
            with self.assertRaisesRegex(ValueError, "Unsafe path"):
                stage04.import_results(os.path.join(d, "stage"), "p", 3, bad)
            self.assertFalse(os.path.exists(os.path.join(d, "escape.txt")))

    def test_rejects_results_of_another_project_and_keeps_previous_results(self):
        with tempfile.TemporaryDirectory() as d:
            model = os.path.join(d, "m")
            write_model(model)
            files = {}
            for n in os.listdir(model):
                with open(os.path.join(model, n)) as fh:
                    files[f"model/{n}"] = fh.read()
            stage_dir = os.path.join(d, "stage")

            good = os.path.join(d, "good.zip")
            self.make_zip(good, {**files, "manifest.json": '{"project_slug": "p", "project": "p", "keyframes": [{"keyframe_index": 0}, {"keyframe_index": 1}, {"keyframe_index": 2}]}'})
            summary = stage04.import_results(stage_dir, "p", 3, good)
            self.assertEqual(summary["images"]["keyframes_registered"], 3)

            other = os.path.join(d, "other.zip")
            self.make_zip(other, {**files, "manifest.json": '{"project_slug": "q", "project": "q", "keyframes": []}'})
            with self.assertRaisesRegex(ValueError, "belong to project"):
                stage04.import_results(stage_dir, "p", 3, other)
            self.assertTrue(os.path.isfile(os.path.join(stage_dir, "results", "model", "images.txt")))

    def test_rejects_non_zip_and_missing_model(self):
        with tempfile.TemporaryDirectory() as d:
            junk = os.path.join(d, "junk.zip")
            with open(junk, "wb") as fh:
                fh.write(b"not a zip")
            with self.assertRaisesRegex(ValueError, "not a valid zip"):
                stage04.import_results(os.path.join(d, "s"), "p", 1, junk)
            empty = os.path.join(d, "empty.zip")
            self.make_zip(empty, {"readme.txt": "hi"})
            with self.assertRaisesRegex(ValueError, "no model/ folder"):
                stage04.import_results(os.path.join(d, "s"), "p", 1, empty)


if __name__ == "__main__":
    unittest.main()
