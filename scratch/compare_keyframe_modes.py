"""
Compares Stage 02 keyframe modes end to end against ground truth:

    video -> Stage 02 (mode) -> Stage 04 export (with gap-fill) -> COLMAP -> pose error vs the true camera path

Needs pycolmap (CPU is fine) and clips with a ground-truth json written by make_synthetic_flight.py.

    PYCOLMAP_PATH=D:/aero3d_tools/pkgs python scratch/compare_keyframe_modes.py \
        --work D:/aero3d_tools/compare  flight,D:/data/flight.mp4,D:/data/flight_groundtruth.json  ...
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
sys.path.insert(0, os.path.join(ROOT, "scratch"))

from services import stage04  # noqa: E402
from services.video_processor import project_slug, process_video_pipeline  # noqa: E402
from eval_stage04_accuracy import evaluate  # noqa: E402


def run_case(clip, video, gt, mode, work, **mode_kwargs):
    case = os.path.join(work, f"{clip}_{mode}")
    shutil.rmtree(case, ignore_errors=True)
    dirs = {k: os.path.join(case, k) for k in ("frames", "keyframes", "trajectories")}
    for d in dirs.values():
        os.makedirs(d)

    name = os.path.basename(video)
    t0 = time.time()
    results = process_video_pipeline(video, name, dirs, keyframe_mode=mode, **mode_kwargs)
    t_stage02 = time.time() - t0

    slug = project_slug(name)
    pkg_zip = os.path.join(case, "pkg.zip")
    manifest = stage04.export_package(results, os.path.join(dirs["keyframes"], slug),
                                      os.path.join(dirs["frames"], slug), pkg_zip, slug)
    with zipfile.ZipFile(pkg_zip) as z:
        z.extractall(os.path.join(case, "pkg"))

    env = dict(os.environ, PYTHONPATH=os.environ.get("PYCOLMAP_PATH", ""))
    t0 = time.time()
    proc = subprocess.run([sys.executable, os.path.join(case, "pkg", "colmap_pipeline.py"), "--images",
                           os.path.join(case, "pkg", "images"), "--out", os.path.join(case, "work"),
                           "--device", "cpu"], env=env, capture_output=True, text=True)
    t_colmap = time.time() - t0

    row = {"clip": clip, "mode": mode, "frames": results["total_frames"], "keyframes": len(results["keyframes"]),
           "max_kf_gap": manifest["keyframe_gaps"]["max_gap_frames"], "fill": len(manifest["fill_frames"]),
           "stage02_s": round(t_stage02, 1), "colmap_s": round(t_colmap, 1)}
    report_path = os.path.join(case, "work", "colmap_report.json")
    if proc.returncode != 0 or not os.path.isfile(report_path):
        row.update(images=row["keyframes"] + row["fill"], registered=0, models=0, failed=True)
        return row
    rep = json.load(open(report_path))
    acc = evaluate(os.path.join(case, "work", "model"), gt) or {}
    row.update(images=rep["num_images_input"], registered=rep["num_images_registered"], models=rep["num_models"],
               reproj_px=rep["mean_reprojection_error_px"], points=rep["num_points3D"],
               pos_err_m=round(acc.get("pos_err_mean_m", float("nan")), 3),
               pos_err_pct=round(acc.get("pos_err_pct_of_path", float("nan")), 3),
               ang_err_deg=round(acc.get("ang_err_mean_deg", float("nan")), 3),
               path_len_m=round(acc.get("path_len_m", 0.0), 1), focal_err_pct=round(acc.get("focal_err_pct", float("nan")), 2))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--modes", default="notebook,sfm")
    ap.add_argument("--target-overlap", type=float, default=None)
    ap.add_argument("--min-overlap", type=float, default=None)
    ap.add_argument("clips", nargs="+", help="name,video.mp4,groundtruth.json")
    args = ap.parse_args()
    kw = {}
    if args.target_overlap is not None:
        kw["target_overlap"] = args.target_overlap
    if args.min_overlap is not None:
        kw["min_overlap"] = args.min_overlap

    rows = []
    for spec in args.clips:
        clip, video, gt = spec.split(",")
        for mode in args.modes.split(","):
            print(f"--- {clip} / {mode}", flush=True)
            rows.append(run_case(clip, video, gt, mode, args.work, **(kw if mode == "sfm" else {})))
            print(rows[-1], flush=True)

    cols = ["clip", "mode", "frames", "keyframes", "max_kf_gap", "fill", "images", "registered", "models",
            "reproj_px", "pos_err_m", "pos_err_pct", "ang_err_deg", "focal_err_pct", "path_len_m", "colmap_s"]
    print("\n" + " | ".join(f"{c:>11}" for c in cols))
    for r in rows:
        print(" | ".join(f"{str(r.get(c, '-')):>11}" for c in cols))
    json.dump(rows, open(os.path.join(args.work, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
