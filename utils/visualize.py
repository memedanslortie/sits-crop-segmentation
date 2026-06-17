import cv2
import numpy as np

def make_video(patch, out_path="s2_rgb.mp4", fps=5, gamma=1):
    T = patch.shape[0]

    frames = []
    for t in range(T):
        rgb = to_rgb(patch[t])
        rgb_uint8 = (rgb**gamma * 255).astype(np.uint8)
        frames.append(rgb_uint8)

    h, w, _ = frames[0].shape

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'avc1'),
        fps,
        (w, h),
        True
    )

    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

    writer.release()

def sentinel2_normalize(x):
    x = np.clip(x, 0, 10000)  # standard S2 reflectance scaling
    return x / 10000.0

def build_rgb(frame):
    # frame: (10, H, W)

    r = frame[3]
    g = frame[2]
    b = frame[1]

    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)

    # normalisation par canal ou globale
    rgb = sentinel2_normalize(rgb)

    return rgb