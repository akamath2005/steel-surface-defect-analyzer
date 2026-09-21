import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from skimage.feature import hog, local_binary_pattern


# ---------------------------------------------------------
# APPLICATION CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Steel Surface Defect Analyzer",
    page_icon="🔍",
    layout="wide",
)


# ---------------------------------------------------------
# VISUAL DESIGN: STEEL BACKGROUND, ORANGE + BLUE HIGHLIGHTS
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    :root {
        --steel-bg: #141b26;
        --steel-panel: #1d2735;
        --steel-raised: #263345;
        --steel-line: #35465d;
        --steel-text: #edf2f7;
        --steel-muted: #9fb0c5;
        --orange: #f2a541;
        --blue: #69bfe8;
    }

    .stApp {
        background:
            radial-gradient(circle at 85% 5%, rgba(105,191,232,.08), transparent 28rem),
            linear-gradient(180deg, #151e2a 0%, var(--steel-bg) 100%);
        color: var(--steel-text);
    }

    [data-testid="stSidebar"] {
        background: #111925;
        border-right: 1px solid var(--steel-line);
    }

    [data-testid="stHeader"] {
        background: rgba(20, 27, 38, .82);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 5rem;
    }

    h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] {
        color: var(--steel-text);
    }

    .hero {
        padding: 2.2rem 2.4rem;
        margin-bottom: 1.4rem;
        background: linear-gradient(125deg, rgba(38,51,69,.96), rgba(29,39,53,.94));
        border: 1px solid var(--steel-line);
        border-left: 5px solid var(--orange);
        border-radius: 14px;
        box-shadow: 0 20px 55px rgba(0,0,0,.24);
    }

    .hero-kicker {
        color: var(--orange);
        font-weight: 800;
        font-size: .8rem;
        letter-spacing: .14em;
        text-transform: uppercase;
        margin-bottom: .65rem;
    }

    .hero h1 {
        margin: 0;
        font-size: clamp(2.1rem, 5vw, 4rem);
        line-height: 1;
        letter-spacing: -.025em;
    }

    .hero p {
        color: var(--steel-muted);
        max-width: 760px;
        margin: 1rem 0 0;
        font-size: 1.04rem;
    }

    .lab-heading {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin: 3.4rem 0 1.25rem;
        padding-bottom: .9rem;
        border-bottom: 1px solid var(--steel-line);
    }

    .lab-number {
        flex: 0 0 auto;
        padding: .38rem .72rem;
        color: #171108;
        background: var(--orange);
        border-radius: 6px;
        font-size: .78rem;
        font-weight: 900;
        letter-spacing: .08em;
        text-transform: uppercase;
    }

    .lab-heading h2 {
        margin: 0;
        color: var(--steel-text);
        font-size: 1.72rem;
    }

    .lab-heading p {
        margin: .25rem 0 0;
        color: var(--steel-muted);
        font-size: .93rem;
    }

    [data-testid="stFileUploader"] {
        background: rgba(29,39,53,.85);
        border: 1px solid var(--steel-line);
        border-radius: 12px;
        padding: 1rem 1.1rem .4rem;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: var(--steel-raised);
        border: 1px dashed var(--blue);
        border-radius: 9px;
    }

    [data-testid="stMetric"] {
        min-height: 112px;
        background: linear-gradient(145deg, rgba(38,51,69,.95), rgba(29,39,53,.95));
        border: 1px solid var(--steel-line);
        border-top: 3px solid var(--blue);
        border-radius: 10px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 25px rgba(0,0,0,.14);
    }

    [data-testid="stMetricLabel"] {
        color: var(--steel-muted);
    }

    [data-testid="stMetricValue"] {
        color: var(--steel-text);
    }

    [data-testid="stImage"] img {
        border: 1px solid var(--steel-line);
        border-radius: 9px;
        background: var(--steel-panel);
    }

    .comparison-label {
        color: var(--blue);
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin: .3rem 0 .65rem;
    }

    .final-panel {
        margin-top: 3.5rem;
        padding: 1.7rem 1.8rem;
        background: linear-gradient(135deg, #243246, #192330);
        border: 1px solid var(--steel-line);
        border-top: 4px solid var(--orange);
        border-radius: 14px;
        box-shadow: 0 20px 45px rgba(0,0,0,.25);
    }

    .final-panel h2 {
        margin: 0 0 .4rem;
    }

    .final-panel p {
        color: var(--steel-muted);
        margin: 0;
    }

    .result-status {
        display: inline-block;
        margin-top: 1rem;
        padding: .52rem .85rem;
        border-radius: 6px;
        font-weight: 800;
    }

    .status-good { color: #b7f7d0; background: rgba(45,180,105,.16); border: 1px solid #3aae72; }
    .status-warn { color: #ffe0a8; background: rgba(242,165,65,.14); border: 1px solid var(--orange); }
    .status-bad { color: #ffc1c1; background: rgba(229,83,83,.15); border: 1px solid #e35c5c; }

    hr {
        border-color: var(--steel-line);
    }

    .stButton > button, [data-testid="stDownloadButton"] > button {
        border-color: var(--orange);
        color: var(--steel-text);
    }

    .stButton > button:hover, [data-testid="stDownloadButton"] > button:hover {
        border-color: var(--blue);
        color: var(--blue);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def show_lab_heading(lab_number, title, description):
    """Display a consistent heading before each lab topic."""
    st.markdown(
        f"""
        <div class="lab-heading">
            <div class="lab-number">Lab No. {lab_number}</div>
            <div>
                <h2>{title}</h2>
                <p>{description}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <section class="hero">
        <div class="hero-kicker">Foundations of Computer Vision Mini Project</div>
        <h1>Steel Surface Defect Analyzer</h1>
        <p>
            A transparent OpenCV pipeline that enhances a steel image, detects
            boundaries, segments suspicious regions and extracts texture and
            shape features. Every stage corresponds to Labs 1–5.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------

st.sidebar.markdown("## Analysis controls")
st.sidebar.caption("Adjust the pipeline and watch the results update.")

gaussian_kernel = st.sidebar.slider(
    "Gaussian kernel size", 3, 15, 5, step=2
)
median_kernel = st.sidebar.slider(
    "Median kernel size", 3, 15, 5, step=2
)
canny_low = st.sidebar.slider(
    "Canny lower threshold", 0, 255, 50
)
canny_high = st.sidebar.slider(
    "Canny upper threshold", 0, 255, 150
)
segmentation_method = st.sidebar.selectbox(
    "Segmentation method",
    [
        "Otsu - Dark Defects",
        "Otsu - Bright Defects",
        "Adaptive Threshold",
        "K-Means Clustering",
    ],
)
morphology_kernel = st.sidebar.slider(
    "Morphology kernel size", 3, 15, 5, step=2
)
minimum_area = st.sidebar.slider(
    "Minimum defect area", 10, 5000, 100, step=10
)
fast_threshold = st.sidebar.slider(
    "FAST corner threshold", 1, 100, 25
)


# ---------------------------------------------------------
# CLEAN UPLOAD AREA
# ---------------------------------------------------------

st.markdown("### Upload a steel surface image")
st.caption("Supported formats: JPG, JPEG, PNG and BMP")

uploaded_file = st.file_uploader(
    "Drag and drop an image here or browse from your computer",
    type=["jpg", "jpeg", "png", "bmp"],
    label_visibility="collapsed",
)


if uploaded_file is None:
    st.info("Upload an image to begin the Lab 1–5 analysis pipeline.")
    st.stop()


# =========================================================
# LAB NO. 1 TOPIC: IMAGE READING AND RESIZING
# =========================================================

pil_image = ImageOps.exif_transpose(
    Image.open(uploaded_file)
).convert("RGB")
original_rgb = np.array(pil_image)
resized_rgb = cv2.resize(
    original_rgb,
    (500, 500),
    interpolation=cv2.INTER_AREA,
)

show_lab_heading(
    1,
    "Image Input and Resizing",
    "Read the image, standardise its colour format and resize it for processing.",
)

st.markdown('<div class="comparison-label">Before / After</div>', unsafe_allow_html=True)
lab1_left, lab1_right = st.columns(2, gap="large")

with lab1_left:
    st.image(original_rgb, caption="Original image", use_container_width=True)
    st.caption(
        f"Original dimensions: {original_rgb.shape[1]} × {original_rgb.shape[0]} pixels"
    )

with lab1_right:
    st.image(resized_rgb, caption="Resized image", use_container_width=True)
    st.caption("Processing dimensions: 500 × 500 pixels")


# =========================================================
# LAB NO. 2 TOPIC: IMAGE ENHANCEMENT
# =========================================================

gray_image = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2GRAY)
equalized_image = cv2.equalizeHist(gray_image)

original_brightness = float(np.mean(gray_image))
enhanced_brightness = float(np.mean(equalized_image))
original_contrast = float(np.std(gray_image))
enhanced_contrast = float(np.std(equalized_image))

show_lab_heading(
    2,
    "Image Enhancement",
    "Convert to grayscale and improve global contrast using histogram equalisation.",
)

st.markdown('<div class="comparison-label">Before / After</div>', unsafe_allow_html=True)
lab2_left, lab2_right = st.columns(2, gap="large")

with lab2_left:
    st.image(gray_image, caption="Before: grayscale image", use_container_width=True)

with lab2_right:
    st.image(
        equalized_image,
        caption="After: histogram-equalised image",
        use_container_width=True,
    )

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Original brightness", f"{original_brightness:.2f}")
metric2.metric("Enhanced brightness", f"{enhanced_brightness:.2f}")
metric3.metric("Original contrast", f"{original_contrast:.2f}")
metric4.metric(
    "Enhanced contrast",
    f"{enhanced_contrast:.2f}",
    delta=f"{enhanced_contrast - original_contrast:+.2f}",
)

histogram_figure, histogram_axes = plt.subplots(1, 2, figsize=(12, 4))
histogram_figure.patch.set_facecolor("#1d2735")

for axis in histogram_axes:
    axis.set_facecolor("#1d2735")
    axis.tick_params(colors="#9fb0c5")
    axis.xaxis.label.set_color("#9fb0c5")
    axis.yaxis.label.set_color("#9fb0c5")
    for spine in axis.spines.values():
        spine.set_color("#35465d")

histogram_axes[0].hist(
    gray_image.ravel(), bins=256, range=(0, 256), color="#69bfe8"
)
histogram_axes[0].set_title("Original histogram", color="#edf2f7")
histogram_axes[0].set_xlabel("Pixel intensity")
histogram_axes[0].set_ylabel("Pixel count")

histogram_axes[1].hist(
    equalized_image.ravel(), bins=256, range=(0, 256), color="#f2a541"
)
histogram_axes[1].set_title("Equalised histogram", color="#edf2f7")
histogram_axes[1].set_xlabel("Pixel intensity")
histogram_axes[1].set_ylabel("Pixel count")

plt.tight_layout()
st.pyplot(histogram_figure)
plt.close(histogram_figure)


# =========================================================
# LAB NO. 3 TOPIC: FILTERING AND EDGE DETECTION
# =========================================================

gaussian_image = cv2.GaussianBlur(
    equalized_image,
    (gaussian_kernel, gaussian_kernel),
    0,
)
median_image = cv2.medianBlur(equalized_image, median_kernel)
blurred_for_sharpening = cv2.GaussianBlur(equalized_image, (5, 5), 0)
sharpened_image = cv2.addWeighted(
    equalized_image,
    1.5,
    blurred_for_sharpening,
    -0.5,
    0,
)

sobel_x = cv2.Sobel(gaussian_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gaussian_image, cv2.CV_64F, 0, 1, ksize=3)
sobel_magnitude = cv2.convertScaleAbs(cv2.magnitude(sobel_x, sobel_y))

if canny_low < canny_high:
    applied_canny_low = canny_low
    applied_canny_high = canny_high
else:
    applied_canny_low = 50
    applied_canny_high = 150

edge_image = cv2.Canny(
    gaussian_image,
    applied_canny_low,
    applied_canny_high,
)
edge_percentage = (
    np.count_nonzero(edge_image) / edge_image.size
) * 100

show_lab_heading(
    3,
    "Filtering and Edge Detection",
    "Reduce noise, sharpen fine details and expose defect boundaries.",
)

if canny_low >= canny_high:
    st.warning(
        "The lower Canny threshold must be smaller than the upper threshold. "
        "Safe defaults of 50 and 150 were applied."
    )

lab3_left, lab3_right = st.columns(2, gap="large")

with lab3_left:
    st.image(
        gaussian_image,
        caption=f"Gaussian filter ({gaussian_kernel} × {gaussian_kernel})",
        use_container_width=True,
    )

with lab3_right:
    st.image(
        median_image,
        caption=f"Median filter ({median_kernel} × {median_kernel})",
        use_container_width=True,
    )

lab3_left_2, lab3_right_2 = st.columns(2, gap="large")

with lab3_left_2:
    st.image(
        sharpened_image,
        caption="Unsharp masking",
        use_container_width=True,
    )

with lab3_right_2:
    st.image(
        sobel_magnitude,
        caption="Sobel gradient magnitude",
        use_container_width=True,
    )

st.markdown('<div class="comparison-label">Filtered Image / Detected Edges</div>', unsafe_allow_html=True)
edge_left, edge_right = st.columns(2, gap="large")

with edge_left:
    st.image(gaussian_image, caption="Filtered input", use_container_width=True)

with edge_right:
    st.image(edge_image, caption="Canny edges", use_container_width=True)

st.metric("Detected edge percentage", f"{edge_percentage:.2f}%")


# =========================================================
# LAB NO. 4 TOPIC: IMAGE SEGMENTATION
# =========================================================

kmeans_image = None
threshold_value = None

if segmentation_method == "Otsu - Dark Defects":
    threshold_value, initial_mask = cv2.threshold(
        gaussian_image,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
elif segmentation_method == "Otsu - Bright Defects":
    threshold_value, initial_mask = cv2.threshold(
        gaussian_image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
elif segmentation_method == "Adaptive Threshold":
    initial_mask = cv2.adaptiveThreshold(
        gaussian_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        5,
    )
else:
    # LAB NO. 4 TOPIC: K-MEANS CLUSTERING
    pixel_values = np.float32(gaussian_image.reshape((-1, 1)))
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2,
    )
    _, labels, centers = cv2.kmeans(
        pixel_values,
        3,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS,
    )
    centers = np.uint8(centers)
    kmeans_image = centers[labels.flatten()].reshape(gaussian_image.shape)
    cluster_counts = np.bincount(labels.flatten(), minlength=3)
    nonempty_clusters = np.where(cluster_counts > 0)[0]
    defect_cluster = nonempty_clusters[
        np.argmin(cluster_counts[nonempty_clusters])
    ]
    initial_mask = np.where(
        labels.reshape(gaussian_image.shape) == defect_cluster,
        255,
        0,
    ).astype(np.uint8)

# LAB NO. 4 TOPIC: MORPHOLOGICAL CLEANING
morphology_element = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (morphology_kernel, morphology_kernel),
)
cleaned_mask = cv2.morphologyEx(
    initial_mask,
    cv2.MORPH_OPEN,
    morphology_element,
)
cleaned_mask = cv2.morphologyEx(
    cleaned_mask,
    cv2.MORPH_CLOSE,
    morphology_element,
)

# LAB NO. 4 TOPIC: CONTOUR DETECTION
contours, _ = cv2.findContours(
    cleaned_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)
filtered_mask = np.zeros_like(cleaned_mask)
detected_contours = []
maximum_allowed_area = cleaned_mask.size * 0.80

for contour in contours:
    contour_area = cv2.contourArea(contour)
    if minimum_area <= contour_area <= maximum_allowed_area:
        detected_contours.append(contour)
        cv2.drawContours(
            filtered_mask,
            [contour],
            -1,
            255,
            thickness=cv2.FILLED,
        )

# LAB NO. 4 TOPIC: DEFECT VISUALISATION
defect_overlay = resized_rgb.copy()
red_layer = np.zeros_like(resized_rgb)
red_layer[:, :, 0] = 255
defect_location = filtered_mask > 0
defect_overlay[defect_location] = (
    0.55 * resized_rgb[defect_location]
    + 0.45 * red_layer[defect_location]
).astype(np.uint8)

bounded_image = resized_rgb.copy()
for contour in detected_contours:
    x, y, width, height = cv2.boundingRect(contour)
    cv2.rectangle(
        bounded_image,
        (x, y),
        (x + width, y + height),
        (242, 165, 65),
        3,
    )

defect_percentage = (
    np.count_nonzero(filtered_mask) / filtered_mask.size
) * 100
number_of_defects = len(detected_contours)

if defect_percentage < 1:
    severity = "No significant defect detected"
    severity_class = "status-good"
elif defect_percentage < 10:
    severity = "Minor surface defect"
    severity_class = "status-warn"
else:
    severity = "Major surface defect"
    severity_class = "status-bad"

show_lab_heading(
    4,
    "Defect Segmentation",
    "Separate suspicious regions, clean the mask and estimate the affected area.",
)

method_line = f"Selected method: **{segmentation_method}**"
if threshold_value is not None:
    method_line += f" · Otsu threshold: **{threshold_value:.2f}**"
st.markdown(method_line)

st.markdown('<div class="comparison-label">Before / After Morphological Cleaning</div>', unsafe_allow_html=True)
lab4_left, lab4_right = st.columns(2, gap="large")

with lab4_left:
    st.image(initial_mask, caption="Initial segmentation mask", use_container_width=True)

with lab4_right:
    st.image(filtered_mask, caption="Cleaned defect mask", use_container_width=True)

if kmeans_image is not None:
    st.image(kmeans_image, caption="K-means clustered image", use_container_width=True)

st.markdown('<div class="comparison-label">Input / Analysed Result</div>', unsafe_allow_html=True)
lab4_left_2, lab4_right_2 = st.columns(2, gap="large")

with lab4_left_2:
    st.image(resized_rgb, caption="Input image", use_container_width=True)

with lab4_right_2:
    st.image(
        defect_overlay,
        caption="Suspected defects highlighted in red",
        use_container_width=True,
    )

lab4_box_left, lab4_box_right = st.columns([1, 1], gap="large")
with lab4_box_left:
    st.image(
        bounded_image,
        caption="Detected regions outlined in orange",
        use_container_width=True,
    )

with lab4_box_right:
    st.metric("Estimated defective area", f"{defect_percentage:.2f}%")
    st.metric("Detected defect regions", number_of_defects)


# =========================================================
# LAB NO. 5 TOPIC: FEATURE EXTRACTION
# =========================================================

# LAB NO. 5 TOPIC: HARRIS CORNER DETECTION
harris_response = cv2.cornerHarris(
    np.float32(gray_image),
    blockSize=2,
    ksize=3,
    k=0.04,
)
harris_response = cv2.dilate(harris_response, None)

if harris_response.max() > 0:
    harris_mask = harris_response > 0.01 * harris_response.max()
else:
    harris_mask = np.zeros_like(gray_image, dtype=bool)

harris_image = resized_rgb.copy()
harris_image[harris_mask] = [255, 80, 80]
harris_binary = np.uint8(harris_mask) * 255
harris_components, _ = cv2.connectedComponents(harris_binary)
harris_corner_count = max(0, harris_components - 1)

# LAB NO. 5 TOPIC: FAST CORNER DETECTION
fast_detector = cv2.FastFeatureDetector_create(
    threshold=fast_threshold,
    nonmaxSuppression=True,
)
fast_keypoints = fast_detector.detect(gray_image, None)
fast_image = cv2.drawKeypoints(
    resized_rgb,
    fast_keypoints,
    None,
    color=(105, 191, 232),
)

# LAB NO. 5 TOPIC: LOCAL BINARY PATTERN (LBP)
lbp_radius = 3
lbp_points = 8 * lbp_radius
lbp_image = local_binary_pattern(
    gray_image,
    lbp_points,
    lbp_radius,
    method="uniform",
)
lbp_display = cv2.normalize(
    lbp_image,
    None,
    0,
    255,
    cv2.NORM_MINMAX,
).astype(np.uint8)
lbp_bins = int(lbp_image.max() + 1)
lbp_histogram, _ = np.histogram(
    lbp_image.ravel(),
    bins=lbp_bins,
    range=(0, lbp_bins),
)
lbp_histogram = lbp_histogram.astype(float)
lbp_histogram /= lbp_histogram.sum() + 1e-7

# LAB NO. 5 TOPIC: HISTOGRAM OF ORIENTED GRADIENTS (HOG)
hog_features, hog_image = hog(
    gray_image,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    visualize=True,
    feature_vector=True,
)
hog_display = cv2.normalize(
    hog_image,
    None,
    0,
    255,
    cv2.NORM_MINMAX,
).astype(np.uint8)

show_lab_heading(
    5,
    "Feature Extraction",
    "Describe corners, local texture and gradient orientation using Harris, FAST, LBP and HOG.",
)

st.markdown('<div class="comparison-label">Harris / FAST Corners</div>', unsafe_allow_html=True)
lab5_left, lab5_right = st.columns(2, gap="large")

with lab5_left:
    st.image(harris_image, caption="Harris corner regions", use_container_width=True)
    st.metric("Harris corner regions", harris_corner_count)

with lab5_right:
    st.image(fast_image, caption="FAST keypoints", use_container_width=True)
    st.metric("FAST keypoints", len(fast_keypoints))

st.markdown('<div class="comparison-label">LBP Texture / HOG Shape</div>', unsafe_allow_html=True)
lab5_left_2, lab5_right_2 = st.columns(2, gap="large")

with lab5_left_2:
    st.image(lbp_display, caption="LBP texture representation", use_container_width=True)

with lab5_right_2:
    st.image(hog_display, caption="HOG feature visualisation", use_container_width=True)

feature_metric1, feature_metric2 = st.columns(2)
feature_metric1.metric("LBP histogram features", len(lbp_histogram))
feature_metric2.metric("HOG feature count", len(hog_features))


# =========================================================
# PROFESSIONAL FINAL RESULT PANEL - NO LARGE TEXT REPORT
# =========================================================

st.markdown(
    f"""
    <section class="final-panel">
        <h2>Final Inspection Result</h2>
        <p>
            The result combines enhancement, edge evidence, segmented area and
            extracted visual features. Parameter changes can affect this estimate.
        </p>
        <div class="result-status {severity_class}">{severity}</div>
    </section>
    """,
    unsafe_allow_html=True,
)

final_metric1, final_metric2, final_metric3, final_metric4 = st.columns(4)
final_metric1.metric("Defective area", f"{defect_percentage:.2f}%")
final_metric2.metric("Defect regions", number_of_defects)
final_metric3.metric("Edge percentage", f"{edge_percentage:.2f}%")
final_metric4.metric("FAST keypoints", len(fast_keypoints))

final_left, final_right = st.columns(2, gap="large")

with final_left:
    st.image(resized_rgb, caption="Inspected steel surface", use_container_width=True)

with final_right:
    st.image(bounded_image, caption="Final detected regions", use_container_width=True)

st.caption(
    "This is a classical computer-vision estimate, not an industrial certification system."
)
