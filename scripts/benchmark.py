#!/usr/bin/env python3
"""
Phase 10.1 - end-to-end CPU benchmark.

Runs the full single-pass pipeline on one clip and records honest wall-clock timing per stage:
  process (frames + keyframes + trajectory) -> COLMAP pose -> Depth Anything V2 -> dense cloud ->
  georeference -> exports. Writes benchmark.json and prints a Markdown table.

    python scripts/benchmark.py [--clip PATH] [--mode sfm_light] [--budget 120] [--matcher exhaustive]

These are the numbers to quote for the "processing speed" criterion - measured, not claimed.
"""
import argparse
import glob
import json
import os
import platform
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from config import settings                                   # noqa: E402
from services.video_processor import process_video_pipeline   # noqa: E402
from services.sfm_local import run_local_stage04              # noqa: E402
from services.fusion import build_dense_cloud                 # noqa: E402
from services.georef import georeference                      # noqa: E402
from services.exporter import export_las, export_camera_track_geojson  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default=None, help="video path (default: first uploads/*.mp4)")
    ap.add_argument("--mode", default="sfm_light", choices=["sfm", "sfm_light", "notebook"])
    ap.add_argument("--budget", type=int, default=120, help="MAX_FRAMES_ON_DISK for this run")
    ap.add_argument("--matcher", default="exhaustive", choices=["auto", "exhaustive", "sequential"])
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmark.json"))
    args = ap.parse_args()

    clip = args.clip or (sorted(glob.glob(os.path.join(ROOT, "uploads", "*.mp4"))) or [None])[0]
    if not clip or not os.path.isfile(clip):
        sys.exit("No clip found. Put a video in uploads/ or pass --clip.")
    settings.MAX_FRAMES_ON_DISK = args.budget

    work = tempfile.mkdtemp(prefix="aero3d_bench_")
    dirs = {k: os.path.join(work, k) for k in ("frames", "keyframes", "trajectories")}
    stage_dir = os.path.join(work, "stage04")
    timings, t_all = {}, time.time()

    def step(name, fn):
        t0 = time.time()
        r = fn()
        timings[name] = round(time.time() - t0, 1)
        print(f"[bench] {name}: {timings[name]}s", flush=True)
        return r

    results = step("process_video", lambda: process_video_pipeline(
        clip, "benchmark_clip.mp4", dirs, keyframe_mode=args.mode))
    slug = results["project_slug"]
    summary = step("stage04_pose_depth", lambda: run_local_stage04(
        results, os.path.join(dirs["keyframes"], slug), os.path.join(dirs["frames"], slug),
        stage_dir, slug, device="cpu", matcher=args.matcher, num_threads=4))
    dense = step("dense_cloud", lambda: build_dense_cloud(
        stage_dir, os.path.join(dirs["keyframes"], slug), os.path.join(dirs["frames"], slug),
        os.path.join(stage_dir, "dense.ply")))
    geo = step("georeference", lambda: georeference(stage_dir, results))
    step("export_las", lambda: export_las(os.path.join(stage_dir, "dense.ply"), os.path.join(stage_dir, "dense.las")))
    step("export_geojson", lambda: export_camera_track_geojson(stage_dir, os.path.join(stage_dir, "track.geojson")))

    report = {
        "hardware": {"platform": platform.platform(), "processor": platform.processor() or "unknown",
                     "cpu_count": os.cpu_count(), "device": "cpu"},
        "clip": {"file": os.path.basename(clip),
                 "resolution": results.get("resolution"), "fps": results.get("fps"),
                 "duration_s": results.get("duration"), "total_frames": results.get("total_frames")},
        "settings": {"keyframe_mode": args.mode, "max_frames_on_disk": args.budget, "matcher": args.matcher},
        "result": {"keyframes": results.get("number_of_final_keyframes"),
                   "registered": summary["images"]["registered"], "verdict": summary["verdict"],
                   "reprojection_error_px": summary["reprojection_error_px"],
                   "dense_points": dense["points"], "depth_maps": summary["depth"]["images_with_depth"],
                   "rmse_horizontal_m": geo["rmse_horizontal_m"], "gps_source": geo["source"]},
        "colmap_timings_s": (summary.get("colmap") or {}).get("timings_s"),
        "stage_timings_s": timings,
        "total_seconds": round(time.time() - t_all, 1),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n| Stage | Seconds |\n|---|---|")
    for k, v in timings.items():
        print(f"| {k} | {v} |")
    print(f"| **TOTAL** | **{report['total_seconds']}** |")
    print(f"\nClip: {report['clip']['file']} {report['clip']['resolution']} "
          f"{report['clip']['duration_s']}s -> {report['result']['dense_points']:,} dense points, "
          f"verdict {report['result']['verdict']}. Wrote {args.out}")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
