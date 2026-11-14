import cv2
import json
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
VIDEO_PATH = "assets/video/clip_01.mp4"
OUTPUT_JSON = "data/video_features.json"
BAND_NAMES = ["North", "Upper-mid", "Middle", "Lower-mid", "South"]
CROP_FRAC = [0.0, 0.0, 1.0, 1.0]
SAMPLE_FPS = 4.0

# Bounding box [x1, y1, x2, y2] isolating the Strait of Gibraltar bottleneck
STRAIT_BOX = [0.405, 0.545, 0.465, 0.605]

def get_ocean_current_mask(hsv_frame):
    H, S, V = cv2.split(hsv_frame)
    mask = ((S > 35) & (V > 30)).astype(np.uint8) * 255
    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

def hue_to_color_index(H_channel):
    h_clipped = np.clip(H_channel.astype(np.float32), 0, 120)
    return ((120.0 - h_clipped) / 120.0) * 100.0

def compute_swirl_index(prev_gray, curr_gray, ocean_mask):
    """
    Uses dense optical flow to measure rotational curl (vorticity / eddies)
    inside the active ocean currents.
    """
    # Downscale by 2x for fast optical flow calculation
    small_prev = cv2.resize(prev_gray, (0, 0), fx=0.5, fy=0.5)
    small_curr = cv2.resize(curr_gray, (0, 0), fx=0.5, fy=0.5)
    small_mask = cv2.resize(ocean_mask, (0, 0), fx=0.5, fy=0.5) > 0

    flow = cv2.calcOpticalFlowFarneback(
        small_prev, small_curr, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    u = flow[..., 0]  # horizontal velocity
    v = flow[..., 1]  # vertical velocity

    # Curl = dv/dx - du/dy (measures circular spinning / eddies)
    dv_dx = np.gradient(v, axis=1)
    du_dy = np.gradient(u, axis=0)
    curl = np.abs(dv_dx - du_dy)

    masked_curl = curl[small_mask]
    return float(np.mean(masked_curl)) if len(masked_curl) > 0 else 0.0

def extract():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / orig_fps
    step = max(1, int(round(orig_fps / SAMPLE_FPS)))
    actual_fps = orig_fps / step

    frames_data = []
    prev_gray = None
    prev_bands = None
    frame_idx = 0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            h, w, _ = bgr.shape
            x1, y1 = int(w * CROP_FRAC[0]), int(h * CROP_FRAC[1])
            x2, y2 = int(w * CROP_FRAC[2]), int(h * CROP_FRAC[3])
            crop = bgr[y1:y2, x1:x2]
            ch, cw, _ = crop.shape

            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            ocean_mask = get_ocean_current_mask(hsv)
            color_idx = hue_to_color_index(hsv[:, :, 0])

            # 1. Extract 5 latitudinal bands (ocean-only)
            nb = len(BAND_NAMES)
            band_h = ch // nb
            band_vals = []
            for b in range(nb):
                r_start = b * band_h
                r_end = (b + 1) * band_h if b < nb - 1 else ch
                b_mask = ocean_mask[r_start:r_end, :] > 0
                b_vals = color_idx[r_start:r_end, :][b_mask]
                if len(b_vals) > 0:
                    band_vals.append(float(np.mean(b_vals)))
                else:
                    band_vals.append(prev_bands[b] if prev_bands else 0.0)

            # 2. Extract Strait of Gibraltar jet intensity
            sx1, sy1 = int(cw * STRAIT_BOX[0]), int(ch * STRAIT_BOX[1])
            sx2, sy2 = int(cw * STRAIT_BOX[2]), int(ch * STRAIT_BOX[3])
            strait_mask = ocean_mask[sy1:sy2, sx1:sx2] > 0
            strait_vals = color_idx[sy1:sy2, sx1:sx2][strait_mask]
            strait_idx = float(np.mean(strait_vals)) if len(strait_vals) > 0 else 0.0

            # 3. Calculate frame-to-frame activity & eddy swirl curl
            if prev_bands is None:
                d_bands = [0.0] * nb
                activity = 0.0
                swirl = 0.0
            else:
                d_bands = [band_vals[b] - prev_bands[b] for b in range(nb)]
                diff = cv2.absdiff(gray, prev_gray)
                masked_diff = diff[ocean_mask > 0]
                activity = float(np.mean(masked_diff)) if len(masked_diff) > 0 else 0.0
                swirl = compute_swirl_index(prev_gray, gray, ocean_mask)

            frames_data.append({
                "bands": [round(v, 3) for v in band_vals],
                "d_bands": [round(d, 3) for d in d_bands],
                "mean": round(float(np.mean(band_vals)), 3),
                "activity": round(activity, 3),
                "swirl": round(swirl, 4),
                "strait": round(strait_idx, 3)
            })

            prev_bands = band_vals
            prev_gray = gray

        frame_idx += 1

    cap.release()

    output = {
        "meta": {
            "band_names": BAND_NAMES,
            "crop_frac": CROP_FRAC,
            "strait_box": STRAIT_BOX,
            "feature_fps": round(actual_fps, 4),
            "duration": round(duration, 3)
        },
        "frames": frames_data
    }

    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Done! Extracted {len(frames_data)} frames with swirl & strait metrics to {OUTPUT_JSON}")

if __name__ == "__main__":
    extract()