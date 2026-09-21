#!/usr/bin/env python3
"""
Stage 04b - Depth Anything V2 depth maps, aligned to the COLMAP reconstruction.

Depth Anything V2 (relative variants) predicts *relative inverse depth* per image: it is right up
to an unknown scale and shift, different for every image. COLMAP's sparse points give true
(SfM-consistent) depth at a few hundred pixels per image, so for each image we fit

        1/z_colmap  ~=  a * prediction + b        (robust least squares)

and store  z = 1 / (a * prediction + b).  All depth maps then live in COLMAP's world units, so
they agree with each other and with the camera poses (needed by the later fusion / meshing stages).

The alignment is validated on HELD-OUT sparse points (20% of points never used for the fit),
so the reported error is an honest estimate rather than a fit residual.

    python depth_pipeline.py --images images --work work \
        --model depth-anything/Depth-Anything-V2-Large-hf

Needs: torch, transformers, opencv, numpy. Input: work/model/*.txt from colmap_pipeline.py.
Output: work/depth/<image>.npy (float16, COLMAP units), work/depth_preview/<image>.jpg,
        work/depth_report.json
Units are COLMAP's arbitrary scale, not metres (metric scale needs georeferencing).
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colmap_io import frame_index_from_name, keyframe_index_from_name, load_model, sparse_depth_samples  # noqa: E402

MIN_POINTS = 12          # fewer aligned sparse points than this -> the image's depth is skipped
HOLDOUT_EVERY = 5        # every 5th sparse point is held out for validation
FLOAT16_MAX = 65000.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True)
    p.add_argument("--work", required=True, help="folder that holds model/ from colmap_pipeline.py")
    p.add_argument("--model", default="depth-anything/Depth-Anything-V2-Large-hf")
    p.add_argument("--kind", default="auto", choices=["auto", "relative", "metric"],
                   help="auto: 'metric' if the model id contains 'Metric', else relative")
    p.add_argument("--input-size", type=int, default=1036, help="longest side fed to the network (multiple of 14)")
    p.add_argument("--store-size", type=int, default=1024, help="longest side of the saved depth maps")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--limit", type=int, default=0, help="only process the first N images (testing)")
    return p.parse_args()


def load_network(model_id, device_arg):
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    device = "cuda" if (device_arg == "auto" and torch.cuda.is_available()) or device_arg == "cuda" else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[depth] loading {model_id} on {device} ({dtype})", flush=True)
    processor = AutoImageProcessor.from_pretrained(model_id)
    # convert after loading: the dtype kwarg is spelled differently across transformers versions
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device=device, dtype=dtype).eval()
    return torch, processor, model, device, dtype


def predict(net, rgb, input_long_side):
    torch, processor, model, device, dtype = net
    h, w = rgb.shape[:2]
    s = input_long_side / max(h, w)
    nh, nw = max(14, round(h * s / 14) * 14), max(14, round(w * s / 14) * 14)
    small = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    pixel_values = processor(images=small, return_tensors="pt", do_resize=False)["pixel_values"]
    with torch.no_grad():
        out = model(pixel_values=pixel_values.to(device, dtype)).predicted_depth
    return out[0].float().cpu().numpy()


def sample(map_2d, uv):
    """Bilinear samples of a (H, W) float map at pixel coordinates uv (M, 2) given in map pixels."""
    mx = uv[:, 0].astype(np.float32).reshape(1, -1)
    my = uv[:, 1].astype(np.float32).reshape(1, -1)
    return cv2.remap(map_2d.astype(np.float32), mx, my, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)[0]


def robust_linear_fit(x, y, iters=4):
    """y ~= a*x + b with iterative 2.5-MAD outlier rejection. Returns (a, b, inlier_mask)."""
    mask = np.ones(len(x), bool)
    a = b = 0.0
    for _ in range(iters):
        A = np.stack([x[mask], np.ones(mask.sum())], axis=1)
        (a, b), *_ = np.linalg.lstsq(A, y[mask], rcond=None)
        res = y - (a * x + b)
        mad = 1.4826 * np.median(np.abs(res[mask] - np.median(res[mask])))
        new = np.abs(res) < 2.5 * max(mad, 1e-12)
        if new.sum() < MIN_POINTS or (new == mask).all():
            break
        mask = new
    return float(a), float(b), mask


def align_relative(pred, uv, z):
    """Fit 1/z ~ a*pred + b on 80% of the sparse points, validate on the other 20%."""
    d = sample(pred, uv)
    inv_z = 1.0 / z
    ok = np.isfinite(d) & np.isfinite(inv_z)
    d, inv_z, z = d[ok], inv_z[ok], z[ok]
    hold = (np.arange(len(d)) % HOLDOUT_EVERY) == 0
    train = ~hold
    if train.sum() < MIN_POINTS:
        return None
    a, b, inl = robust_linear_fit(d[train], inv_z[train])
    if a <= 0:
        return None
    inv_hat = a * d[hold] + b
    z_hat = 1.0 / np.maximum(inv_hat, 1e-9)
    rel_err = np.abs(z_hat - z[hold]) / z[hold]
    return {"a": a, "b": b, "n_fit": int(train.sum()), "n_inliers": int(inl.sum()),
            "n_holdout": int(hold.sum()), "holdout_rel_err_median": float(np.median(rel_err)),
            "z_far": float(np.percentile(z, 99) * 3.0)}


def align_metric(pred, uv, z):
    d = sample(pred, uv)
    ok = np.isfinite(d) & (d > 1e-6)
    d, z = d[ok], z[ok]
    hold = (np.arange(len(d)) % HOLDOUT_EVERY) == 0
    if (~hold).sum() < MIN_POINTS:
        return None
    s = float(np.median(z[~hold] / d[~hold]))
    rel_err = np.abs(s * d[hold] - z[hold]) / z[hold]
    return {"scale": s, "n_fit": int((~hold).sum()), "n_holdout": int(hold.sum()),
            "holdout_rel_err_median": float(np.median(rel_err)), "z_far": float(np.percentile(z, 99) * 3.0)}


def colorize(depth):
    inv = 1.0 / np.maximum(depth, 1e-9)
    lo, hi = np.percentile(inv, [2, 98])
    norm = np.clip((inv - lo) / max(hi - lo, 1e-12), 0, 1)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)  # warm = near


def main():
    args = parse_args()
    model_dir = os.path.join(args.work, "model")
    if not os.path.isfile(os.path.join(model_dir, "images.txt")):
        sys.exit(f"{model_dir} not found - run colmap_pipeline.py first")
    kind = args.kind if args.kind != "auto" else ("metric" if "metric" in args.model.lower() else "relative")

    recon = load_model(model_dir)
    images = sorted(recon.images.values(), key=lambda im: im.name)
    if args.limit:
        images = images[:args.limit]
    depth_dir, prev_dir = os.path.join(args.work, "depth"), os.path.join(args.work, "depth_preview")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(prev_dir, exist_ok=True)

    net = load_network(args.model, args.device)
    per_image, t0 = [], time.time()
    for n, im in enumerate(images, 1):
        stem = os.path.splitext(os.path.basename(im.name))[0]
        entry = {"name": im.name, "frame_index": frame_index_from_name(im.name),
                 "keyframe_index": keyframe_index_from_name(im.name), "stem": stem}
        bgr = cv2.imread(os.path.join(args.images, im.name))
        if bgr is None:
            entry.update(status="image_missing")
            per_image.append(entry)
            continue
        cam = recon.cameras[im.camera_id]
        h0, w0 = bgr.shape[:2]
        sc = min(1.0, args.store_size / max(h0, w0))
        sw, sh = max(1, round(w0 * sc)), max(1, round(h0 * sc))

        pred = cv2.resize(predict(net, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), args.input_size), (sw, sh),
                          interpolation=cv2.INTER_LINEAR)
        uv, z = sparse_depth_samples(recon, im)
        uv_store = uv * np.array([sw / cam.width, sh / cam.height])
        fit = (align_relative if kind == "relative" else align_metric)(pred, uv_store, z) if len(z) else None
        if fit is None:
            entry.update(status="skipped_too_few_sparse_points", n_sparse_points=int(len(z)))
            per_image.append(entry)
            continue

        if kind == "relative":
            depth = 1.0 / np.maximum(fit["a"] * pred + fit["b"], 1.0 / fit["z_far"])
        else:
            depth = np.minimum(fit["scale"] * np.maximum(pred, 1e-6), fit["z_far"])
        np.save(os.path.join(depth_dir, stem + ".npy"), np.minimum(depth, FLOAT16_MAX).astype(np.float16))
        cv2.imwrite(os.path.join(prev_dir, stem + ".jpg"), colorize(depth), [cv2.IMWRITE_JPEG_QUALITY, 88])
        entry.update(status="ok", n_sparse_points=int(len(z)), store_size=[sw, sh],
                     **{k: round(v, 6) if isinstance(v, float) else v for k, v in fit.items()})
        per_image.append(entry)
        if n % 5 == 0 or n == len(images):
            print(f"[depth] {n}/{len(images)}  ({(time.time() - t0) / n:.2f}s/img)  "
                  f"held-out rel. err {fit['holdout_rel_err_median'] * 100:.1f}%", flush=True)

    ok = [e for e in per_image if e["status"] == "ok"]
    errs = [e["holdout_rel_err_median"] for e in ok]
    warnings = []
    if len(ok) < len(per_image):
        warnings.append(f"{len(per_image) - len(ok)} image(s) got no depth map (too few sparse points).")
    if errs and float(np.median(errs)) > 0.15:
        warnings.append(f"Median held-out depth error {100 * np.median(errs):.0f}% is high; depth maps are "
                        "only roughly consistent with the reconstruction.")
    report = {
        "stage": "depth",
        "model": args.model,
        "kind": kind,
        "input_size": args.input_size,
        "device": net[3],
        "images_total": len(per_image),
        "images_with_depth": len(ok),
        "holdout_rel_err_median": round(float(np.median(errs)), 4) if errs else None,
        "holdout_rel_err_p90": round(float(np.percentile(errs, 90)), 4) if errs else None,
        "seconds": round(time.time() - t0, 1),
        "units": "COLMAP world units (arbitrary scale, not metres)",
        "warnings": warnings,
        "images": per_image,
    }
    with open(os.path.join(args.work, "depth_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[depth] done. {len(ok)}/{len(per_image)} depth maps; median held-out error "
          f"{report['holdout_rel_err_median']}")


if __name__ == "__main__":
    main()
