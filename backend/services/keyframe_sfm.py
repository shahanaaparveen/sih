"""
Keyframe selection tuned for Structure-from-Motion (mode "sfm").

The original "notebook" filter keeps frames that pass an absolute sharpness cutoff and whose
equalised-histogram correlation to the previous keyframe is low. Neither number relates to what SfM
needs: consecutive images must OVERLAP enough to be matched (roughly 65-85%), and each image
should be as sharp as it can be. This selector works on exactly those two things.

Inputs are per-frame quantities the pipeline already computes in its single pass:
  * transforms[i]  2x3 similarity mapping frame i-1 pixel coordinates -> frame i (None = estimation failed)
  * sharpness[i]   Laplacian variance of frame i

Walking forward from the last keyframe R, the overlap between R and frame j is the area of R's frame
rectangle, moved into frame j by the chained transforms, that still lies inside frame j. Once the
overlap falls to `target_overlap` a new keyframe is due; among the frames from there until the
overlap reaches `min_overlap` the SHARPEST one is taken. Frames in between are redundant. Motion
blur therefore never costs coverage: a blurry frame is only skipped in favour of a sharper neighbour
that overlaps just as well.
"""
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

DEFAULT_TARGET_OVERLAP = 0.85   # a keyframe becomes due when overlap with the previous one drops to this
DEFAULT_MIN_OVERLAP = 0.65      # ...and must be chosen before it drops below this
MAX_CONSECUTIVE_FAILURES = 2    # motion could not be estimated for this many frames in a row -> force a keyframe


def _to3(m: np.ndarray) -> np.ndarray:
    out = np.eye(3)
    out[:2, :] = m
    return out


def overlap_fraction(chain: np.ndarray, width: float, height: float) -> float:
    """
    Shared area of two views as a fraction of the LARGER of the two footprints (intersection over the
    bigger one), where `chain` (3x3) maps the earlier frame into the current one.
    1.0 = identical view, 0.0 = nothing in common.

    Dividing by the larger footprint matters: when the camera flies forward the scene zooms in, the earlier
    frame's footprint grows beyond the current frame, and "how much of the current frame does the old one
    still cover" would stay at ~100% however far the drone has travelled. This form drops for zooming out
    or in, translating and rotating alike.
    """
    rect = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    moved = cv2.perspectiveTransform(rect[None], chain)[0].astype(np.float32)
    if not np.isfinite(moved).all():
        return 0.0
    moved_area = abs(float(cv2.contourArea(moved)))
    inter, _ = cv2.intersectConvexConvex(moved, rect)
    union_ref = max(moved_area, width * height)
    return float(max(0.0, min(1.0, inter / union_ref))) if union_ref > 0 else 0.0


def first_usable_frame(transforms: Sequence[Optional[np.ndarray]], sharpness: Sequence[float],
                       min_relative_sharpness: float = 0.3) -> int:
    """
    Index of the first frame that can start a reconstruction: motion to the next frame could be
    estimated (so it has features) and it is not a near-black / fading frame (sharpness at least
    `min_relative_sharpness` x the video's median). Fade-ins, black leaders and title cards are skipped.
    """
    s = np.asarray(sharpness, dtype=np.float64)
    floor = min_relative_sharpness * float(np.median(s)) if s.size else 0.0
    for i in range(len(s) - 1):
        if transforms[i + 1] is not None and s[i] >= floor:
            return i
    return 0


def select_sfm_keyframes(
    transforms: Sequence[Optional[np.ndarray]],
    sharpness: Sequence[float],
    width: float,
    height: float,
    target_overlap: float = DEFAULT_TARGET_OVERLAP,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
) -> Dict[str, object]:
    """
    Returns {"keyframes": [frame_index, ...], "overlaps": [overlap with the previous keyframe or None]}.
    The first keyframe is the first usable frame (see first_usable_frame), normally frame 0.
    transforms[0] is ignored.
    """
    n = len(sharpness)
    if n == 0:
        return {"keyframes": [], "overlaps": []}
    if not 0.0 < min_overlap < target_overlap <= 1.0:
        raise ValueError("need 0 < min_overlap < target_overlap <= 1")

    start = first_usable_frame(transforms, sharpness)
    keyframes: List[int] = [start]
    overlaps: List[Optional[float]] = [None]
    ref = start
    chain = np.eye(3)
    window: List[int] = []            # frames whose overlap with `ref` is within [min_overlap, target_overlap]
    seen: Dict[int, float] = {}       # measured overlap with `ref` of every frame walked so far
    failures = 0
    last_ok = ref                     # last frame that still overlapped `ref` >= min_overlap

    def commit(frame: int):
        nonlocal ref, window, seen, failures, last_ok
        keyframes.append(frame)
        overlaps.append(seen.get(frame))
        ref = frame
        window, seen, failures, last_ok = [], {}, 0, frame

    j = start + 1
    while j < n:
        t = transforms[j]
        if t is None:
            failures += 1
            chain = _to3(np.eye(2, 3)) @ chain      # unknown motion: assume none, but count it
            ov = None
        else:
            failures = 0
            chain = _to3(np.asarray(t, dtype=np.float64)) @ chain
            ov = overlap_fraction(chain, width, height)
            seen[j] = ov

        must_decide = failures >= MAX_CONSECUTIVE_FAILURES
        if ov is not None:
            if ov >= min_overlap:
                last_ok = j
                if ov <= target_overlap:
                    window.append(j)
            else:
                must_decide = True

        if must_decide:
            if window:
                pick = max(window, key=lambda f: sharpness[f])
            elif last_ok > ref:
                pick = last_ok                        # never reached the "due" band: take the last frame that still matched
            else:
                pick = j                              # motion too fast to bridge: forced, will show up as a gap
            commit(pick)
            # frames after the pick were measured against the old reference: restart the walk from the pick
            chain = np.eye(3)
            j = pick + 1
            continue
        j += 1

    return {"keyframes": keyframes, "overlaps": overlaps}


def detect_scene_cuts(frame_diffs: Sequence[float], min_abs: float = 15.0, factor: float = 5.0) -> List[int]:
    """
    Frame indices where the video hard-cuts to a different shot.

    frame_diffs[i] is the mean absolute pixel difference between tiny greyscale versions of frame i-1 and
    frame i (frame_diffs[0] is ignored). A cut is a difference far above the clip's normal frame-to-frame
    change: both above `min_abs` grey levels and `factor` x the median. Gradual fades stay below that, and
    fast camera motion normally does too. On a real FPV clip the five cuts scored 24-71 while the rest of
    the 2,000 frames stayed under 12 (median 3.6).
    """
    d = np.asarray(frame_diffs, dtype=np.float64)
    if d.size < 3:
        return []
    threshold = max(min_abs, factor * float(np.median(d[1:])))
    return [int(i) for i in np.where(d > threshold)[0] if i > 0]


def relative_sharpness_gate(sharpness: Sequence[float], window: int = 31, fraction: float = 0.5) -> List[int]:
    """
    Frame indices that are at least `fraction` as sharp as the median of their neighbourhood.
    Used for reporting ("frames surviving blur filtering") - a relative measure, so it does not depend
    on resolution or scene content the way the fixed cutoff of 100 does.
    """
    s = np.asarray(sharpness, dtype=np.float64)
    if s.size == 0:
        return []
    half = window // 2
    keep = []
    for i in range(s.size):
        lo, hi = max(0, i - half), min(s.size, i + half + 1)
        if s[i] >= fraction * np.median(s[lo:hi]):
            keep.append(i)
    return keep
