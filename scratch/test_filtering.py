import json, cv2, os
import numpy as np

with open('backend/storage/latest_pipeline_results.json', 'r') as f:
    d = json.load(f)

frames = d.get('frames', [])
candidate_541 = [f for f in frames if f['sharpness'] >= 300.0]
print(f"Candidate frames (threshold >= 300.0): {len(candidate_541)}")

COMPARE_WIDTH = 320
COMPARE_HEIGHT = 180

# Preload small grayscale images directly
preloaded_raw = []
preloaded_eq = []
metadata = []

for f in candidate_541:
    idx = f['frame_index']
    p = os.path.join('backend/storage/frames', f'frame_{idx:04d}.jpg')
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    small = cv2.resize(img, (COMPARE_WIDTH, COMPARE_HEIGHT))
    small_eq = cv2.equalizeHist(small)
    preloaded_raw.append(small)
    preloaded_eq.append(small_eq)
    metadata.append((idx, f['sharpness']))

print(f"Successfully loaded {len(preloaded_raw)} frames into memory.")

# Precompute histograms
hists_raw = [cv2.calcHist([img], [0], None, [256], [0, 256]) for img in preloaded_raw]
hists_eq = [cv2.calcHist([img], [0], None, [256], [0, 256]) for img in preloaded_eq]

# Test 1: with equalizeHist at SIMILARITY_THRESHOLD = 0.95
kf_eq = [metadata[0]]
prev_h = hists_eq[0]
correls_eq = []
for i in range(1, len(hists_eq)):
    c = cv2.compareHist(prev_h, hists_eq[i], cv2.HISTCMP_CORREL)
    correls_eq.append(c)
    if c < 0.95:
        kf_eq.append(metadata[i])
        prev_h = hists_eq[i]

print(f"\n[Test 1] With equalizeHist, SIMILARITY_THRESHOLD = 0.95:")
print(f"  Keyframes: {len(kf_eq)}")
print(f"  Correlations: min={min(correls_eq):.4f}, max={max(correls_eq):.4f}, mean={np.mean(correls_eq):.4f}")

# Test 2: without equalizeHist at SIMILARITY_THRESHOLD = 0.95
kf_raw = [metadata[0]]
prev_h = hists_raw[0]
correls_raw = []
for i in range(1, len(hists_raw)):
    c = cv2.compareHist(prev_h, hists_raw[i], cv2.HISTCMP_CORREL)
    correls_raw.append(c)
    if c < 0.95:
        kf_raw.append(metadata[i])
        prev_h = hists_raw[i]

print(f"\n[Test 2] WITHOUT equalizeHist, SIMILARITY_THRESHOLD = 0.95:")
print(f"  Keyframes: {len(kf_raw)}")
print(f"  Correlations: min={min(correls_raw):.4f}, max={max(correls_raw):.4f}, mean={np.mean(correls_raw):.4f}")

# Test 3: Range of thresholds on equalizeHist
print("\n[Test 3] Threshold sweep WITH equalizeHist:")
for th in [0.99, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]:
    cnt = 1
    prev_h = hists_eq[0]
    for i in range(1, len(hists_eq)):
        c = cv2.compareHist(prev_h, hists_eq[i], cv2.HISTCMP_CORREL)
        if c < th:
            cnt += 1
            prev_h = hists_eq[i]
    print(f"  Threshold {th:.2f} -> {cnt} keyframes")

# Test 4: Range of thresholds WITHOUT equalizeHist
print("\n[Test 4] Threshold sweep WITHOUT equalizeHist:")
for th in [0.9999, 0.9995, 0.999, 0.998, 0.995, 0.99, 0.98, 0.97, 0.96, 0.95, 0.90]:
    cnt = 1
    prev_h = hists_raw[0]
    for i in range(1, len(hists_raw)):
        c = cv2.compareHist(prev_h, hists_raw[i], cv2.HISTCMP_CORREL)
        if c < th:
            cnt += 1
            prev_h = hists_raw[i]
    print(f"  Threshold {th:.4f} -> {cnt} keyframes")
