#!/usr/bin/env python3
"""
Stage 04a - Structure-from-Motion with COLMAP (pycolmap).

Keyframe images -> camera intrinsics + a 6-DoF pose for every registered image + a sparse 3D
point cloud. Runs unchanged on CPU (pycolmap) or GPU (pycolmap-cuda12, e.g. on Colab).

    python colmap_pipeline.py --images images --out work [--camera-model SIMPLE_RADIAL]
                              [--matcher auto|exhaustive|sequential] [--device auto|cpu|cuda]

Writes to --out:
    database.db                    COLMAP feature/match database
    sparse/<n>/                    every reconstruction COLMAP produced (binary)
    model/{cameras,images,points3D}.txt      the BEST reconstruction (most registered images), text
    colmap_report.json             accuracy / health numbers, timings, warnings

Poses and scale are in COLMAP's own arbitrary world frame. Metric scale and georeferencing are
a later stage (they need real GPS/known distances).
"""
import argparse
import json
import os
import re
import shutil
import sys
import time

import pycolmap

# Below this many images, all pairs are matched (most accurate). Above it, video-style
# sequential matching keeps the run time sane.
EXHAUSTIVE_MAX_IMAGES = 300


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, help="folder with the keyframe JPEGs")
    p.add_argument("--out", required=True, help="work/output folder")
    p.add_argument("--camera-model", default="SIMPLE_RADIAL",
                   help="SIMPLE_RADIAL (default, robust) | OPENCV (more distortion terms, needs many well-spread views) | PINHOLE ...")
    p.add_argument("--matcher", default="auto", choices=["auto", "exhaustive", "sequential"])
    p.add_argument("--sequential-overlap", type=int, default=25)
    p.add_argument("--max-image-size", type=int, default=3200, help="longest side used for SIFT (px)")
    p.add_argument("--max-features", type=int, default=8192)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num-threads", type=int, default=-1,
                   help="CPU threads for COLMAP (-1 = all cores). Each SIFT thread decodes an image at the same time, "
                        "so on a machine with little free RAM a smaller number avoids an out-of-memory abort")
    p.add_argument("--reuse-sparse", action="store_true",
                   help="skip feature extraction / matching / mapping and rebuild model/ and the report from the "
                        "reconstructions already in --out/sparse (e.g. after a session restart)")
    return p.parse_args()


def frame_of(name):
    """Video frame a package image came from ('keyframe_0007_original_0123.jpg' -> 123), or None."""
    m = re.search(r"original_(\d+)", os.path.basename(name))
    return int(m.group(1)) if m else None


def describe_models(recs):
    """One entry per reconstruction COLMAP produced, largest first, with the video frames it covers."""
    out = []
    for idx, rec in recs.items():
        frames = [f for f in (frame_of(im.name) for im in rec.images.values()) if f is not None]
        out.append({"index": int(idx), "images": int(rec.num_reg_images()), "points": int(rec.num_points3D()),
                    "frame_range": [min(frames), max(frames)] if frames else None,
                    "mean_reprojection_error_px": round(float(rec.compute_mean_reprojection_error()), 4)})
    return sorted(out, key=lambda m: -m["images"])


def registered_in_any(recs):
    return len({im.name for rec in recs.values() for im in rec.images.values()})


def timed(label, timings, fn, *a, **kw):
    t0 = time.time()
    print(f"[colmap] {label} ...", flush=True)
    result = fn(*a, **kw)
    timings[label] = round(time.time() - t0, 1)
    print(f"[colmap] {label} done in {timings[label]}s", flush=True)
    return result


def main():
    args = parse_args()
    images_dir = os.path.abspath(args.images)
    out = os.path.abspath(args.out)
    names = sorted(f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if len(names) < 3:
        sys.exit(f"Need at least 3 images, found {len(names)} in {images_dir}")

    if not args.reuse_sparse:
        shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.join(out, "sparse"), exist_ok=True)
    db = os.path.join(out, "database.db")
    device = {"auto": pycolmap.Device.auto, "cpu": pycolmap.Device.cpu, "cuda": pycolmap.Device.cuda}[args.device]
    has_cuda = bool(getattr(pycolmap, "has_cuda", False))
    print(f"[colmap] pycolmap {pycolmap.__version__} | CUDA build: {has_cuda} | {len(names)} images")

    warnings, timings = [], {}
    if args.device == "cuda" and not has_cuda:
        sys.exit("--device cuda requested but this pycolmap build has no CUDA support")

    def run_sfm():
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = args.camera_model
        extraction = pycolmap.FeatureExtractionOptions()
        extraction.max_image_size = args.max_image_size
        extraction.sift.max_num_features = args.max_features
        extraction.num_threads = args.num_threads
        matching = pycolmap.FeatureMatchingOptions()
        matching.num_threads = args.num_threads

        # One physical camera filmed the whole video -> share a single set of intrinsics.
        # That is the main accuracy lever: intrinsics are estimated from every image at once.
        timed("feature extraction", timings, pycolmap.extract_features, db, images_dir,
              camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
              extraction_options=extraction, device=device)

        matcher = args.matcher
        if matcher == "auto":
            matcher = "exhaustive" if len(names) <= EXHAUSTIVE_MAX_IMAGES else "sequential"
        if matcher == "exhaustive":
            timed("exhaustive matching", timings, pycolmap.match_exhaustive, db, matching_options=matching, device=device)
        else:
            pairing = pycolmap.SequentialPairingOptions()
            pairing.overlap = args.sequential_overlap
            pairing.quadratic_overlap = True
            pairing.num_threads = args.num_threads
            timed("sequential matching", timings, pycolmap.match_sequential, db,
                  matching_options=matching, pairing_options=pairing, device=device)

        opts = pycolmap.IncrementalPipelineOptions()
        opts.multiple_models = True          # keep every disconnected piece so we can report splits
        opts.ba_refine_focal_length = True
        opts.ba_refine_extra_params = True
        opts.ba_refine_principal_point = False   # principal point stays at the image centre (more stable)
        opts.min_model_size = 3
        opts.num_threads = args.num_threads
        recs = timed("incremental mapping + bundle adjustment", timings, pycolmap.incremental_mapping,
                     db, images_dir, os.path.join(out, "sparse"), opts)
        return recs, matcher

    if args.reuse_sparse:
        sparse_dir = os.path.join(out, "sparse")
        recs = {int(n): pycolmap.Reconstruction(os.path.join(sparse_dir, n))
                for n in sorted(os.listdir(sparse_dir)) if n.isdigit()}
        matcher = "reused"
        print(f"[colmap] reusing {len(recs)} existing reconstruction(s) from {sparse_dir}")
    else:
        recs, matcher = run_sfm()

    if not recs:
        sys.exit("COLMAP could not build any reconstruction. Typical causes: too little overlap between "
                 "keyframes (large gaps), or too few textured features (water, sky, uniform fields).")

    best_id = max(recs, key=lambda k: recs[k].num_reg_images())
    best = recs[best_id]
    model_dir = os.path.join(out, "model")
    os.makedirs(model_dir, exist_ok=True)
    best.write_text(model_dir)

    n_reg, n_total = best.num_reg_images(), len(names)
    mean_err = float(best.compute_mean_reprojection_error())
    track = float(best.compute_mean_track_length())
    cam = next(iter(best.cameras.values()))
    models = describe_models(recs)
    n_any = registered_in_any(recs)
    if len(recs) > 1:
        warnings.append(f"COLMAP produced {len(recs)} disconnected reconstructions "
                        f"({', '.join(str(m['images']) for m in models)} images); {n_any}/{n_total} images registered "
                        "in some model, and only the largest is imported. Separate reconstructions mean the video "
                        "has hard cuts or passages that are too fast/blurred to link.")
    elif n_reg < 0.8 * n_total:
        warnings.append(f"Only {n_reg}/{n_total} images registered ({100 * n_reg // n_total}%). Typical causes: too "
                        "little overlap between neighbouring keyframes, heavy motion blur, or featureless surfaces "
                        "(water, sky, uniform fields).")
    if mean_err > 1.0:
        warnings.append(f"Mean reprojection error {mean_err:.2f}px is high (good is < 0.8px).")

    report = {
        "stage": "colmap",
        "colmap_version": pycolmap.__version__,
        "cuda_build": has_cuda,
        "device_requested": args.device,
        "matcher": matcher,
        "camera_model": cam.model.name if hasattr(cam.model, "name") else str(cam.model),
        "camera_params": [float(x) for x in cam.params],
        "image_size": [int(cam.width), int(cam.height)],
        "num_images_input": n_total,
        "num_images_registered": n_reg,
        "registered_fraction": round(n_reg / n_total, 4),
        "num_models": len(recs),
        "models": models,
        "num_images_registered_any_model": n_any,
        "registered_any_fraction": round(n_any / n_total, 4),
        "num_points3D": int(best.num_points3D()),
        "mean_track_length": round(track, 3),
        "mean_observations_per_image": round(float(best.compute_mean_observations_per_reg_image()), 1),
        "mean_reprojection_error_px": round(mean_err, 4),
        "timings_s": timings,
        "warnings": warnings,
    }
    with open(os.path.join(out, "colmap_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
