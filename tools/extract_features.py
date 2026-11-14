# tools/extract_features.py
# Usage (from project root, venv active):
#   python tools/extract_features.py tools/raw/frame_01.jpg
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------- settings ----------------
N_COLS = 128          # number of longitude slices (the "melody" steps)
WIDTH = 1024         # work at this width for speed

# Thresholds (tune using data/debug_masks.png)
FLOW_S_MIN = 0.35    # min saturation for a colored flow pixel
FLOW_V_MIN = 0.18    # min brightness for a colored flow pixel
LAND_S_MAX = 0.20    # max saturation for gray land
LAND_V_MIN = 0.15    # land is mid-dark gray...
LAND_V_MAX = 0.55    # ...but not bright

IMG_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tools/raw/frame_01.jpg")
OUT_JSON = Path("data/features.json")
DEBUG_PNG = Path("data/debug_masks.png")
OUT_JSON.parent.mkdir(exist_ok=True)

# ---------------- load ----------------
bgr = cv2.imread(str(IMG_PATH))
if bgr is None:
    raise FileNotFoundError(f"Could not read image: {IMG_PATH}")
h0, w0 = bgr.shape[:2]
bgr = cv2.resize(bgr, (WIDTH, round(h0 * WIDTH / w0)), interpolation=cv2.INTER_AREA)
H, W = bgr.shape[:2]

hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
hue = hsv[..., 0] * 2.0            # degrees 0-360
sat = hsv[..., 1] / 255.0
val = hsv[..., 2] / 255.0
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

# ---------------- masks ----------------
flow = (sat >= FLOW_S_MIN) & (val >= FLOW_V_MIN)
land = (sat <= LAND_S_MAX) & (val >= LAND_V_MIN) & (val <= LAND_V_MAX) & ~flow
land = cv2.medianBlur(land.astype(np.uint8) * 255, 5) > 0   # remove speckle

# ---------------- intensity from hue ----------------
# blue (~220 deg) = low  ->  green -> yellow -> orange/red (~0 deg) = high
h = np.where(hue > 300, hue - 360, hue)          # wrap reds near 360 to negative
intensity = np.clip((220.0 - h) / 220.0, 0.0, 1.0)

edges = cv2.Canny(gray, 60, 160) > 0             # texture / turbulence proxy

# ---------------- global features ----------------
n_px = H * W
fi = intensity[flow]
hist, _ = np.histogram(fi, bins=5, range=(0, 1))
global_feats = {
    "land_pct": float(land.sum() / n_px),
    "flow_pct": float(flow.sum() / n_px),
    "background_pct": float(1 - land.sum() / n_px - flow.sum() / n_px),
    "mean_intensity": float(fi.mean()) if fi.size else 0.0,
    "intensity_hist_5bins": (hist / max(hist.sum(), 1)).round(4).tolist(),
    "mean_brightness": float(val.mean()),
    "contrast": float(gray.std() / 255.0),
    "edge_density": float(edges[~land].mean()),
}

# ---------------- per-column features ----------------
bounds = np.linspace(0, W, N_COLS + 1, dtype=int)
rows = np.arange(H)[:, None]
columns = []
for i in range(N_COLS):
    x0, x1 = bounds[i], bounds[i + 1]
    f, l = flow[:, x0:x1], land[:, x0:x1]
    s, e, v = intensity[:, x0:x1], edges[:, x0:x1], val[:, x0:x1]

    w = (s ** 2) * f                              # emphasise strong flow
    lat_frac = float((w * rows).sum() / w.sum() / H) if w.sum() > 0 else 0.5
    ocean = ~l

    columns.append({
        "i": i,
        "lon_deg": float(-180 + 360 * (i + 0.5) / N_COLS),
        "land_frac": float(l.mean()),
        "flow_cov": float(f.mean()),
        "mean_intensity": float(s[f].mean()) if f.any() else 0.0,
        "peak_intensity": float(np.percentile(s[f], 95)) if f.sum() > 20 else 0.0,
        "strong_lat_deg": 90.0 - 180.0 * lat_frac,   # latitude of strongest flow
        "turbulence": float(e[ocean].mean()) if ocean.any() else 0.0,
        "brightness": float(v.mean()),
    })

# ---------------- save ----------------
out = {
    "meta": {
        "source": str(IMG_PATH), "work_size": [W, H], "n_cols": N_COLS,
        "projection": "equirectangular (x = longitude, y = latitude)",
        "thresholds": {"FLOW_S_MIN": FLOW_S_MIN, "FLOW_V_MIN": FLOW_V_MIN,
                       "LAND_S_MAX": LAND_S_MAX, "LAND_V_MIN": LAND_V_MIN,
                       "LAND_V_MAX": LAND_V_MAX},
    },
    "global": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in global_feats.items()},
    "columns": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()} for c in columns],
}
OUT_JSON.write_text(json.dumps(out, indent=2))

# ---------------- debug image: original | masks ----------------
mask_img = np.zeros_like(bgr)
mask_img[land] = (255, 255, 255)       # land = white
mask_img[flow] = (0, 200, 0)           # flow = green
for x in bounds[1:-1]:
    cv2.line(mask_img, (int(x), 0), (int(x), H - 1), (80, 80, 80), 1)
cv2.imwrite(str(DEBUG_PNG), np.hstack([bgr, mask_img]))

print("Saved", OUT_JSON, "and", DEBUG_PNG)
for k, v in out["global"].items():
    print(f"  {k}: {v}")