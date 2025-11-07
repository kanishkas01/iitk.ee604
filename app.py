# refactored_depth_measurement.py
# Works in Colab or local Python (with display fallbacks).
# Usage: either call run_pipeline(...) with appropriate args, or run interactively.

import os
import sys
import math
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
import torch

# Optional Colab-friendly display
def in_colab():
    return 'google.colab' in sys.modules

if in_colab():
    from google.colab.patches import cv2_imshow
    def show(img, title=None):
        cv2_imshow(img)
else:
    def show(img, title=None):
        # Use matplotlib for consistent display in notebooks / scripts
        if img.ndim == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray')
        if title:
            plt.title(title)
        plt.axis('off')
        plt.show()

# --- Utilities and safer helpers ---
def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path)
    if img is None:
        raise IOError(f"cv2 failed to read image: {path}")
    return img

def safe_good_features(mask, max_corners=10, quality=0.05, min_dist=50):
    """Return Nx1x2 int32 corners or None safely."""
    if mask is None or mask.size == 0:
        return None
    corners = cv2.goodFeaturesToTrack(mask, maxCorners=max_corners,
                                      qualityLevel=quality, minDistance=min_dist)
    if corners is None:
        return None
    return np.int32(corners)

def small_area_remover(binary):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    output = np.zeros_like(binary)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_component_label = np.argmax(areas) + 1
        output[labels == largest_component_label] = 255
    return output

def merge_colinear_lines(lines, angle_threshold=5, distance_threshold=20):
    """Merge and extend colinear line segments. Returns list of [x1,y1,x2,y2]."""
    if lines is None:
        return []
    # normalize to list of 4-tuples
    lines_flat = []
    for l in lines:
        if isinstance(l, np.ndarray) and l.shape[-1] == 4:
            x1,y1,x2,y2 = l.reshape(-1)
        elif isinstance(l, (list,tuple)) and len(l) == 4:
            x1,y1,x2,y2 = l
        else:
            # Hough returns [[x1,y1,x2,y2]] maybe
            arr = np.array(l).reshape(-1)
            if arr.size >= 4:
                x1,y1,x2,y2 = arr[:4]
            else:
                continue
        lines_flat.append([int(x1),int(y1),int(x2),int(y2)])

    merged = []
    def angle(line):
        x1,y1,x2,y2 = line
        return math.degrees(math.atan2(y2-y1, x2-x1))

    def endpoint_min_dist(l1, l2):
        pts1 = [(l1[0],l1[1]),(l1[2],l1[3])]
        pts2 = [(l2[0],l2[1]),(l2[2],l2[3])]
        dmin = min(math.hypot(x1-x2,y1-y2) for (x1,y1) in pts1 for (x2,y2) in pts2)
        return dmin

    for ln in lines_flat:
        merged_flag = False
        a1 = angle(ln)
        for i, m in enumerate(merged):
            a2 = angle(m)
            if abs((a1 - a2)) < angle_threshold and endpoint_min_dist(ln, m) < distance_threshold:
                # combine endpoints
                xs = [ln[0], ln[2], m[0], m[2]]
                ys = [ln[1], ln[3], m[1], m[3]]
                # choose extremal endpoints along main direction
                if abs(max(xs) - min(xs)) >= abs(max(ys) - min(ys)):
                    idx_min = xs.index(min(xs))
                    idx_max = xs.index(max(xs))
                else:
                    idx_min = ys.index(min(ys))
                    idx_max = ys.index(max(ys))
                merged[i] = [xs[idx_min], ys[idx_min], xs[idx_max], ys[idx_max]]
                merged_flag = True
                break
        if not merged_flag:
            merged.append(ln)
    return merged

# --- SAD / view / measurement helpers (kept from your original logic, but safer) ---
def sad(camheight, depthmap_bgr, mask, viewport=[3.4,3.6], f=6.5):
    """Return bounding dx,dy and bounding box from mask area using features+Hough lines.
       depthmap_bgr: colorized depth image (BGR)
       mask: single-channel binary mask (0/255)
    """
    if mask is None:
        raise ValueError("Mask is None in sad()")
    # ensure binary
    bin_mask = np.where(mask>0, 255, 0).astype(np.uint8)
    corners = safe_good_features(bin_mask, max_corners=10, quality=0.05, min_dist=30)
    if corners is None or len(corners) == 0:
        # fallback to boundingRect of mask
        ys, xs = np.where(bin_mask>0)
        if len(xs) == 0:
            raise ValueError("Mask empty in sad()")
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
    else:
        x_min = int(np.min(corners[:,:,0
