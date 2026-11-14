import cv2
import json
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
VIDEO_PATH = "assets/video/clip_01.mp4"
OUTPUT_JSON = "data/video_features.json"
BAND_NAMES = ["North", "Upper-mid", "Middle", "Lower-mid", "South"]
# Crop fraction [x1, y1, x2, y2] matching your UI overlay
CROP_FRAC = [0.0, 0.0, 1.0, 1.0]
SAMPLE_FPS = 4.0  # How many samples per second to extract

def get_ocean_current_mask(hsv_frame):
    """
    Returns a binary mask (255 = active colored ocean current, 0 = grey land or black background).
    """
    H, S, V = cv2.split(hsv_frame)
    # Grey land has low saturation (S < 35). Empty black ocean has low value (V < 30).
    mask = ((S > 35) & (V > 30)).astype(np.uint8) * 255
    
    # Clean up tiny 1-pixel noise specks
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def hue_to_color_index(H_channel):
    """
    Maps OpenCV Hue (0-179) from Blue (~110) -> Green (~60) -> Yellow (~30) -> Red (0)
    into a 0 to 100 scale (0 = coldest/slowest blue, 100 = hottest/fastest red).
    """
    # Clip hue between 0 (red) and 120 (deep blue)
    h_clipped = np.clip(H_channel.astype(np.float32), 0, 120)
    # Invert so 120 (blue) becomes 0, and 0 (red) becomes 100
    index = ((120.0 - h_clipped) / 120.0) * 100.0
    return index

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

            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            # 1. Create the Ocean-Only Mask (ignores grey land!)
            ocean_mask = get_ocean_current_mask(hsv)

            # Save debug images on the very first frame to verify the mask
            if frame_idx == 0:
                cv2.imwrite("debug_mask.png", ocean_mask)
                masked_preview = cv2.bitwise_and(crop, crop, mask=ocean_mask)
                cv2.imwrite("debug_masked_ocean.png", masked_preview)
                print("Saved debug_mask.png and debug_masked_ocean.png")
            
            # 2. Convert Hue to 0-100 temperature/speed index
            color_idx = hue_to_color_index(hsv[:, :, 0])

            # 3. Slice into 5 horizontal latitudinal bands
            nb = len(BAND_NAMES)
            band_h = crop.shape[0] // nb
            band_vals = []

            for b in range(nb):
                r_start = b * band_h
                r_end = (b + 1) * band_h if b < nb - 1 else crop.shape[0]

                b_mask = ocean_mask[r_start:r_end, :] > 0
                b_vals = color_idx[r_start:r_end, :][b_mask]

                if len(b_vals) > 0:
                    band_vals.append(float(np.mean(b_vals)))
                else:
                    # Fallback if a band has zero colored currents
                    band_vals.append(prev_bands[b] if prev_bands else 0.0)

            # 4. Calculate frame-to-frame change (d_bands) & ocean activity
            if prev_bands is None:
                d_bands = [0.0] * nb
                activity = 0.0
            else:
                d_bands = [band_vals[b] - prev_bands[b] for b in range(nb)]
                # Measure pixel motion ONLY inside the ocean mask
                diff = cv2.absdiff(gray, prev_gray)
                masked_diff = diff[ocean_mask > 0]
                activity = float(np.mean(masked_diff)) if len(masked_diff) > 0 else 0.0

            frames_data.append({
                "bands": [round(v, 3) for v in band_vals],
                "d_bands": [round(d, 3) for d in d_bands],
                "mean": round(float(np.mean(band_vals)), 3),
                "activity": round(activity, 3)
            })

            prev_bands = band_vals
            prev_gray = gray

        frame_idx += 1

    cap.release()

    output = {
        "meta": {
            "band_names": BAND_NAMES,
            "crop_frac": CROP_FRAC,
            "feature_fps": round(actual_fps, 4),
            "duration": round(duration, 3)
        },
        "frames": frames_data
    }

    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Done! Extracted {len(frames_data)} ocean-masked frames to {OUTPUT_JSON}")

if __name__ == "__main__":
    extract()