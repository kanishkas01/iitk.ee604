import streamlit as st
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
import io

# -----------------------------------------------------------
# Utility: display via Streamlit instead of cv2_imshow
# -----------------------------------------------------------
def show_cv_image(image, caption=None):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(rgb_image, caption=caption, use_container_width=True)

# -----------------------------------------------------------
# Core functions (exact same as your original code)
# -----------------------------------------------------------
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
        d = np.min([
            np.hypot(x11 - x21, y11 - y21),
            np.hypot(x11 - x22, y11 - y22),
            np.hypot(x12 - x21, y12 - y21),
            np.hypot(x12 - x22, y12 - y22)
        ])
        return d

    for line in lines:
        x1, y1, x2, y2 = line[0]
        merged = False
        for i, mline in enumerate(merged_lines):
            if abs(line_angle(line[0]) - line_angle(mline)) < angle_threshold and endpoint_distance(line[0], mline) < distance_threshold:
                all_points = np.array([[x1, y1], [x2, y2], [mline[0], mline[1]], [mline[2], mline[3]]])
                x_coords = all_points[:, 0]
                y_coords = all_points[:, 1]
                if abs(x_coords[0] - x_coords[1]) > abs(y_coords[0] - y_coords[1]):
                    idx_min = np.argmin(x_coords)
                    idx_max = np.argmax(x_coords)
                else:
                    idx_min = np.argmin(y_coords)
                    idx_max = np.argmax(y_coords)
                merged_lines[i] = [x_coords[idx_min], y_coords[idx_min], x_coords[idx_max], y_coords[idx_max]]
                merged = True
                break
        if not merged:
            merged_lines.append([x1, y1, x2, y2])
    return merged_lines

def sad(camheight, depthmap, mask, viewport=[3.4, 3.6], f=6.5, imgsize=None):
    gray_img = cv2.cvtColor(depthmap, cv2.COLOR_BGR2GRAY)
    binary = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    corners = cv2.goodFeaturesToTrack(mask, 10, 0.05, 50)
    if corners is None:
        return [0, 0, (0, 0), (0, 0)]
    corners = np.int32(corners)

    edges = cv2.Canny(mask, 20, 50, apertureSize=3)
    linesP = cv2.HoughLinesP(edges, 0.5, np.pi / 720, threshold=10, minLineLength=70, maxLineGap=20)

    if linesP is not None:
        merged_lines = merge_colinear_lines(linesP, 15, 400)

    x_min = np.min(corners[:, :, 0])
    y_min = np.min(corners[:, :, 1])
    x_max = np.max(corners[:, :, 0])
    y_max = np.max(corners[:, :, 1])

    dx = x_max - x_min
    dy = y_max - y_min
    return [dx, dy, (x_min, y_min), (x_max, y_max)]

def view(dx, dy, px, py, camh=300, cx=0.82, cy=0.79, f=6.5, viewport=[3.6, 6.4]):
    tx = (dx / px) * viewport[1]
    ty = (dy / py) * viewport[0]
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

def mean_depth(depth, lt_p, rb_p):
    lx, ly = lt_p
    rx, ry = rb_p
    return np.mean(depth[ly:ry, lx:rx])

# -----------------------------------------------------------
# Streamlit App
# -----------------------------------------------------------
st.title("EE604 3D Object Measurement (Width, Length, Depth)")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
col1, col2, col3 = st.columns(3)
with col1:
    relative_heigh_ration = st.selectbox("Relative Height Ratio", ["low", "med", "high", "vhigh"])
with col2:
    nom_of_objects = st.number_input("Number of Objects", min_value=1, max_value=10, value=1, step=1)
with col3:
    camh = st.number_input("Camera Height (mm)", min_value=1, value=289, step=1)

ref_h = st.number_input("Reference Object Height (mm)", min_value=1.0, value=100.0, step=1.0)

if uploaded_file and st.button("Run Measurement"):
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    initial_image = img_bgr.copy()

    st.write("Running depth estimation model...")
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)

    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    post_processed = processor.post_process_depth_estimation(outputs, target_sizes=[(img.height, img.width)])
    depth_result = post_processed[0]
    depth = depth_result["predicted_depth"].squeeze().cpu().numpy()
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
    magma_cmap = plt.cm.get_cmap('magma')
    depth_magma = magma_cmap(depth_norm)
    depth_magma_rgb = (depth_magma[:, :, :3] * 255).astype(np.uint8)
    depth_color = cv2.cvtColor(depth_magma_rgb, cv2.COLOR_RGB2BGR)

    show_cv_image(depth_color, "Depth Map")

    gray = cv2.cvtColor(depth_color, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    smoothed_hist = gaussian_filter1d(hist, sigma=1.89)

    if relative_heigh_ration == "low":
        low_bound = 110; error_rct = 1.08
    elif relative_heigh_ration == "med":
        low_bound = 100; error_rct = 1.23
    elif relative_heigh_ration == "high":
        low_bound = 80; error_rct = 2.91
    else:
        low_bound = 60; error_rct = 3.0

    derivative = np.gradient(smoothed_hist[low_bound:])
    zero_crossings = np.where(np.diff(np.sign(derivative)))[0]
    minima = np.array([i for i in zero_crossings if derivative[i-1] < 0 and derivative[i+1] > 0]).astype(int) + low_bound

    kmeans = KMeans(n_clusters=nom_of_objects, random_state=42)
    kmeans.fit(minima.reshape(-1,1))
    centers = np.sort(kmeans.cluster_centers_.reshape(len(kmeans.cluster_centers_)))

    gray = cv2.cvtColor(depth_color, cv2.COLOR_BGR2GRAY)
    ret, ground = cv2.threshold(gray, minima[0],255,cv2.THRESH_BINARY)

    masks = {}
    if nom_of_objects > 1:
        for i in range(1, nom_of_objects):
            _, thresh = cv2.threshold(gray, centers[i], 255, cv2.THRESH_BINARY)
            binary = ground - thresh
            output = small_area_remover(binary)
            masks[i] = output
        sum_img = np.zeros(gray.shape, dtype=np.uint8)
        for i in range(1, nom_of_objects):
            sum_img = cv2.add(sum_img, masks[i])
        residual = cv2.subtract(ground, sum_img)
        _, residual = cv2.threshold(residual, 1, 255, cv2.THRESH_BINARY)
        masks[0] = small_area_remover(residual)
    else:
        masks = {0: small_area_remover(ground)}

    bounding_boxes = []
    temp = depth_color.copy()
    for i in range(nom_of_objects):
        dx, dy, tl_p, br_p = sad(camheight=camh, depthmap=temp, mask=masks[i])
        x, y = view(dx, dy, px=initial_image.shape[0], py=initial_image.shape[1], f=5.42, viewport=[6.144,8.6], camh=camh)
        cv2.circle(temp, tl_p, 5, (0, 255, 0), 2)
        cv2.circle(temp, br_p, 5, (0, 255, 0), 2)
        cv2.rectangle(temp, tl_p, br_p, (0, 255, 0), 2)
        bounding_boxes.append([tl_p, br_p])
        cv2.putText(temp,f"<Width {int(x)}mm>",(tl_p[0],br_p[1]),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0),3)
        temp = vertical_text(temp,f"<Length {int(y)}>mm",tl_p)

    ref = np.mean(depth_color[0:bounding_boxes[0][0][1], 0:bounding_boxes[0][0][0]])
    mean_val = []
    min1 = 255
    for i in range(nom_of_objects):
        depth_copy = depth_color.copy()
        _01img = masks[i]//255
        meanint = depth_copy[_01img==1].mean()
        if(ref < meanint < min1):
            min1 = meanint
        mean_val.append(meanint)

    scaler = float(min1 - ref)
    for i in range(0, nom_of_objects):
        temph = (float(mean_val[i]-ref)/scaler)*ref_h
        cv2.putText(temp,f"v Depth {int(temph)}mm v", org = bounding_boxes[i][0], fontFace=cv2.FONT_HERSHEY_SIMPLEX , fontScale=1 , thickness=3 , color = (255, 255, 0))

    show_cv_image(temp, "Final Annotated Image")
    st.success("Processing complete. Final output displayed above.")
