"""
Stage 04 run locally on CPU (no Colab round-trip).

The Colab path was: export_package (images + gap-fills + manifest + the scripts) -> upload -> run
colmap_pipeline.py + depth_pipeline.py in the notebook -> download results zip -> import_results.

This module does exactly that, on this machine: it reuses export_package and import_results unchanged
and just runs the two bundled scripts in between with the project's own Python. That keeps a single
code path for pose/depth and reuses all the existing validation, summarising and storage.

    summary = run_local_stage04(results, keyframes_dir, frames_dir, stage_dir, slug, progress=cb)
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from typing import Any, Callable, Dict, List, Optional

from config import settings
from services import stage04 as stage04_service
from services import masking

# Files produced by the scripts that make up the "results" package import_results expects.
_RESULT_MEMBERS = ("model", "depth", "depth_preview", "masks", "colmap_report.json", "depth_report.json", "manifest.json")

ProgressCB = Optional[Callable[[str, str], None]]   # (stage, message)


def _emit(cb: ProgressCB, stage: str, message: str):
    print(f"[stage04-local] {stage}: {message}", flush=True)
    if cb:
        try:
            cb(stage, message)
        except Exception:
            pass


def _run_script(script_path: str, args: List[str], cb: ProgressCB, label: str) -> None:
    """Run one bundled script with this project's interpreter, streaming its output to the log/callback."""
    cmd = [sys.executable, script_path, *args]
    _emit(cb, label, "starting " + " ".join(os.path.basename(c) for c in cmd[:2]))
    proc = subprocess.Popen(cmd, cwd=os.path.dirname(script_path), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        tail.append(line)
        del tail[:-40]
        if line:
            _emit(cb, label, line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed (exit {proc.returncode}). Last output:\n" + "\n".join(tail[-15:]))


def _zip_results(pkg_dir: str, out_zip: str) -> None:
    """Bundle only the members import_results needs (not the input images) into a results zip."""
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for member in _RESULT_MEMBERS:
            path = os.path.join(pkg_dir, member)
            if os.path.isfile(path):
                z.write(path, member)
            elif os.path.isdir(path):
                for dp, _, files in os.walk(path):
                    for f in files:
                        full = os.path.join(dp, f)
                        z.write(full, os.path.relpath(full, pkg_dir))


def run_local_stage04(results: Dict[str, Any], keyframes_dir: str, frames_dir: str, stage_dir: str,
                      slug: str, device: str = "cpu", matcher: str = "auto",
                      depth_model: Optional[str] = None, max_image_size: int = 1600,
                      num_threads: Optional[int] = None, run_depth: bool = True,
                      run_masking: Optional[bool] = None, progress: ProgressCB = None) -> Dict[str, Any]:
    """
    End-to-end local pose (+ optional depth) for one processed project. Returns the same summary dict
    the Colab import produced, and writes it to <stage_dir>/summary.json.
    """
    depth_model = depth_model or settings.DEPTH_MODEL
    run_masking = settings.MASK_DYNAMIC if run_masking is None else run_masking
    work = tempfile.mkdtemp(prefix="aero3d_stage04_")
    t0 = time.time()
    try:
        # 1. Reuse export_package to assemble images/ (keyframes + gap-fills) + manifest + the scripts.
        pkg_zip = os.path.join(work, "package.zip")
        _emit(progress, "prepare", "selecting keyframes and gap-fill frames")
        stage04_service.export_package(results, keyframes_dir, frames_dir, pkg_zip, slug)

        pkg = os.path.join(work, "pkg")
        with zipfile.ZipFile(pkg_zip) as z:
            z.extractall(pkg)
        images = os.path.join(pkg, "images")
        # COLMAP/depth write here. It MUST be separate from `images`: colmap_pipeline clears its --out
        # at startup, which would otherwise delete the input images.
        out_dir = os.path.join(work, "sfm")
        n_imgs = len([f for f in os.listdir(images) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        _emit(progress, "prepare", f"{n_imgs} images ready")

        # 2a. Dynamic-object masks (people/vehicles/animals) so they don't become false geometry.
        masks_dir = None
        if run_masking:
            masks_dir = os.path.join(pkg, "masks")
            _emit(progress, "mask", "detecting dynamic objects (people/vehicles/animals)")
            try:
                mstats = masking.generate_masks(images, masks_dir, model_path=settings.YOLO_MODEL, progress=progress)
                _emit(progress, "mask", f"{mstats['images_with_dynamic']}/{mstats['images']} images had dynamic objects")
            except Exception as e:
                _emit(progress, "mask", f"masking skipped ({e})")
                masks_dir = None

        # 2b. Camera pose + sparse cloud (COLMAP), masking out dynamic pixels if available.
        colmap_args = ["--images", images, "--out", out_dir, "--device", device, "--matcher", matcher,
                       "--max-image-size", str(max_image_size)]
        if masks_dir:
            colmap_args += ["--masks", masks_dir]
        if num_threads:
            colmap_args += ["--num-threads", str(num_threads)]
        _run_script(os.path.join(pkg, "colmap_pipeline.py"), colmap_args, progress, "COLMAP")

        # 3. Aligned depth (Depth Anything V2), optional.
        if run_depth:
            _run_script(os.path.join(pkg, "depth_pipeline.py"),
                        ["--images", images, "--work", out_dir, "--model", depth_model, "--device", device],
                        progress, "Depth")

        # 4. Repackage the outputs and reuse import_results (validation + summarise + storage).
        #    The manifest lives with the package images; bring it next to the model for import.
        shutil.copyfile(os.path.join(pkg, "manifest.json"), os.path.join(out_dir, "manifest.json"))
        if masks_dir and os.path.isdir(masks_dir):
            shutil.copytree(masks_dir, os.path.join(out_dir, "masks"), dirs_exist_ok=True)
        res_zip = os.path.join(work, "stage04_results.zip")
        _zip_results(out_dir, res_zip)
        _emit(progress, "import", "validating and summarising results")
        summary = stage04_service.import_results(stage_dir, slug, len(results.get("keyframes") or []), res_zip)
        summary["local_run_seconds"] = round(time.time() - t0, 1)
        summary["ran_locally"] = True
        # persist the updated summary (import_results already wrote it; rewrite with the timing fields)
        with open(os.path.join(stage_dir, "summary.json"), "w", encoding="utf-8") as f:
            import json
            json.dump(summary, f, indent=2)
        _emit(progress, "done", f"verdict={summary.get('verdict')} in {summary['local_run_seconds']}s")
        return summary
    finally:
        shutil.rmtree(work, ignore_errors=True)
