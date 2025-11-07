import streamlit as st
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans

# ---------------------------
# Helper Functions
# ---------------------------

def small_area_remover(binary):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    output = np.zeros_like(binary)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_component_label = np.argmax(areas) + 1
        output[labels == largest_component_label] = 255
    return output

def merge_colinear_lines(lines, angle_threshold=5, distance_threshold=20):
    if lines is None:
        return []
    merged_lines = []
    def line_angle(l):
        x1, y1, x2, y2 = l
        return np.degrees(np.arctan2(y2 - y1, x2 - x1))
    def endpoint_distance(l1, l2):
        x11, y11, x12, y12 = l1
        x21, y21, x22, y22 = l2
        dists = [
            np.hypot(x11 - x21, y11 - y21),
            np.hypot(x11 - x22, y11 - y22),
            np.hypot(x12 - x21, y12 - y21),
            np.hypot(x12 - x22, y12 - y22)
        ]
        return np.min(dists)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        merged = False
        for i, mline in enumerate(merged_lines):
            if abs(line_angle(line[0]) - line_angle(mline)) < angle_threshold and endpoint_distance(line[0], mline) < distance_threshold:
                pts = np.array([[x1,y1],[x2,y2],[mline[0],mline[1]],[mline[2],mline[3]]])
                x_coords, y_coords = pts[:,0], pts[:,1]
                if abs(x_coords[0]-x_coords[1]) > abs(y_coords[0]-y_coords[1]):
                    idx_min, idx_max = np.argmin(x_coords), np.argmax(x_coords)
                else:
                    idx_min, idx_max = np.argmin(y_coords), np.argmax(y_coords)
                merged_lines[i] = [x_coords[idx_min], y_coords[idx_min], x_coords[idx_max], y_coords[idx_max]]
                merged = True
                break
        if not merged:
            merged_lines.append([x1, y1, x2, y2])
    return merged_lines

def sad(camheight, depthmap, mask, viewport=[3.4, 3.6], f=6.5, imgsize=None):
    gray_img = cv2.cvtColor(depthmap, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(mask, 10, 0.05, 50)
    if corners is None:
        return [0, 0, (0, 0), (0, 0)]
    corners = np.int32(corners)
    x_min = np.min(corners[:, :, 0])
    y_min = np.min(corners[:, :, 1])
    x_max = np.max(corners[:, :, 0])
    y_max = np.max(corners[:, :, 1])
    dx, dy = x_max - x_min, y_max - y_min
    return [dx, dy, (x_min, y_min), (x_max, y_max)]

# ✅ Fixed version
def view(dx, dy, img_width, img_height, camh=300, cx=0.82, cy=0.79, f=6.5, viewport=[3.6, 6.4]):
    """Corrected: uses image width for dx, image height for dy."""
    v_view = viewport[0]  # vertical real span
    h_view = viewport[1]  # horizontal real span
    tx = (dx / float(img_width)) * h_view
    ty = (dy / float(img_height)) * v_view
    x = (camh / f) * tx
    y = (camh / f) * ty
    return [(cx) * x, (cy) * y]

def vertical_text(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX, scale=1, color=(0,255,0),
                  thickness=3, lineType=cv2.LINE_AA, angle=90):
    x, y = org
    img_out = img.copy()
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    text_img = np.zeros((text_h + baseline, text_w, 3), dtype=np.uint8)
    cv2.putText(text_img, text, (0, text_h), font, scale, color, thickness, lineType)
    M = cv2.getRotationMatrix2D((text_w//2, text_h//2), angle, 1.0)
    rotated = cv2.warpAffine(text_img, M, (text_h, text_w), flags=cv2.INTER_LINEAR)
    h, w = rotated.shape[:2]
    if y + h <= img_out.shape[0] and x + w <= img_out.shape[1]:
        img_out[y:y+h, x:x+w] = np.where(rotated>0, rotated, img_out[y:y+h, x:x+w])
    return img_out

# ---------------------------
# Streamlit App
# ---------------------------

st.title("3D Object Measurement (Width, Length, Depth)")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
relative_heigh_ration = st.selectbox("Relative Height Ratio", ["low", "med", "high", "vhigh"])
nom_of_objects = st.number_input("Number of Objects", min_value=1, value=1, step=1)
camh = st.number_input("Camera Height (mm)", min_value=1, value=289, step=1)
ref_h = st.number_input("Reference Object Height (mm)", min_value=1.0, value=100.0, step=1.0)

if uploaded_file and st.button("Run"):
    # --- Load and prepare image ---
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    initial_image = img_bgr.copy()

    # --- Depth estimation ---
    st.write("Running depth estimation model...")
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    result = processor.post_process_depth_estimation(outputs, target_sizes=[(image.height, image.width)])[0]
    depth = result["predicted_depth"].squeeze().cpu().numpy()
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
    magma = plt.cm.get_cmap('magma')
    depth_color = (magma(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR)

    # --- Histogram analysis ---
    gray = cv2.cvtColor(depth_color, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    smoothed_hist = gaussian_filter1d(hist, sigma=1.89)

    # Bounds
    if relative_heigh_ration == "low":
        low_bound = 110
    elif relative_heigh_ration == "med":
        low_bound = 100
    elif relative_heigh_ration == "high":
        low_bound = 80
    else:
        low_bound = 60

    derivative = np.gradient(smoothed_hist[low_bound:])
    zero_crossings = np.where(np.diff(np.sign(derivative)))[0]
    minima = np.array([i for i in zero_crossings if derivative[i-1] < 0 and derivative[i+1] > 0]).astype(int) + low_bound

    kmeans = KMeans(n_clusters=nom_of_objects, random_state=42)
    kmeans.fit(minima.reshape(-1,1))
    centers = np.sort(kmeans.cluster_centers_.reshape(-1))

    # --- Object segmentation ---
    ret, ground = cv2.threshold(gray, minima[0],255,cv2.THRESH_BINARY)
    masks = {}
    if nom_of_objects > 1:
        for i in range(1, nom_of_objects):
            _, thresh = cv2.threshold(gray, centers[i], 255, cv2.THRESH_BINARY)
            binary = cv2.subtract(ground, thresh)
            masks[i] = small_area_remover(binary)
        masks[0] = small_area_remover(ground)
    else:
        masks[0] = small_area_remover(ground)

    # --- Measurement ---
    bounding_boxes = []
    temp = depth_color.copy()
    img_height = initial_image.shape[0]
    img_width = initial_image.shape[1]

    for i in range(nom_of_objects):
        dx, dy, tl_p, br_p = sad(camheight=camh, depthmap=temp, mask=masks[i])
        x, y = view(dx, dy,
                    img_width=img_width,
                    img_height=img_height,
                    camh=camh,
                    f=5.42,
                    viewport=[6.144, 8.6])
        cv2.circle(temp, tl_p, 5, (0,255,0), 2)
        cv2.circle(temp, br_p, 5, (0,255,0), 2)
        cv2.rectangle(temp, tl_p, br_p, (0,255,0), 2)
        bounding_boxes.append([tl_p, br_p])
        cv2.putText(temp, f"<Width {int(x)}mm>", (tl_p[0], br_p[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
        temp = vertical_text(temp, f"<Length {int(y)}mm>", tl_p)

    # --- Depth estimation ---
    ref = np.mean(depth_color[0:bounding_boxes[0][0][1], 0:bounding_boxes[0][0][0]])
    mean_vals = []
    min1 = 255
    for i in range(nom_of_objects):
        dcopy = depth_color.copy()
        m = masks[i] // 255
        meanint = dcopy[m == 1].mean()
        if ref < meanint < min1:
            min1 = meanint
        mean_vals.append(meanint)

    scaler = float(min1 - ref)
    for i in range(nom_of_objects):
        temph = (float(mean_vals[i] - ref) / scaler) * ref_h
        cv2.putText(temp, f"v Depth {int(temph)}mm v", bounding_boxes[i][0],
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3)

    # --- Output final image ---
    st.image(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB), caption="Final Annotated Image", use_container_width=True)
    st.success("Processing Complete!")
