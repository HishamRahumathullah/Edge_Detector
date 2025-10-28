# streamlit_app.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

st.set_page_config(layout="wide", page_title="Interactive Edge Detection UI")

st.title("Interactive Edge Detection UI")
st.markdown(
    "Upload an image and experiment with Sobel, Laplacian and Canny edge detectors. Compare all algorithms side-by-side.")

# Sidebar: upload and algorithm selector
with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload an image (JPG, PNG, BMP)", type=["jpg", "jpeg", "png", "bmp"])

    st.markdown("### Algorithm Selection")
    compare_all = st.checkbox("Compare All Algorithms Side-by-Side", value=True)

    if not compare_all:
        algo = st.radio("Select single algorithm", ("Sobel", "Laplacian", "Canny"))
    else:
        algo = "Comparison"

    st.markdown("### Algorithm parameters")

    # Common parameters
    st.markdown("#### Common Parameters")
    use_blur = st.checkbox("Apply Gaussian Blur (for noise reduction)", value=False)
    if use_blur:
        blur_ksize = st.slider("Blur kernel size", 1, 31, 5, step=2)
        blur_sigma = st.slider("Blur sigma", 0.1, 10.0, 1.0, step=0.1)

    # Canny params
    if compare_all or algo == "Canny":
        st.markdown("#### Canny Parameters")
        canny_low = st.slider("Lower threshold", 0, 255, 50)
        canny_high = st.slider("Upper threshold", 0, 255, 150)

    # Sobel params
    if compare_all or algo == "Sobel":
        st.markdown("#### Sobel Parameters")
        sobel_ksize = st.slider("Sobel kernel size", 1, 31, 3, step=2)
        sobel_scale = st.slider("Sobel scale", 1, 10, 1)
        sobel_delta = st.slider("Sobel delta", 0, 255, 0)

    # Laplacian params
    if compare_all or algo == "Laplacian":
        st.markdown("#### Laplacian Parameters")
        lap_ksize = st.slider("Laplacian kernel size", 1, 31, 3, step=2)
        lap_scale = st.slider("Laplacian scale", 1, 10, 1)
        lap_delta = st.slider("Laplacian delta", 0, 255, 0)

    st.markdown("---")
    real_time = st.checkbox("Real-time update", value=True)
    apply_btn = st.button("Apply (if real-time off)")


# Helper functions
def ensure_odd(k):
    k = int(k)
    return k if k % 2 == 1 else (k - 1 if k > 1 else 1)


def to_gray_cv(img_pil):
    img = np.array(img_pil.convert("RGB"))
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def apply_gaussian_blur(gray, ksize, sigma):
    """Apply Gaussian blur to image"""
    k = ensure_odd(ksize)
    return cv2.GaussianBlur(gray, (k, k), sigmaX=sigma)


def apply_sobel(gray, ksize=3, direction="Both", scale=1, delta=0):
    k = ensure_odd(ksize)

    if direction == "Both":
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k, scale=scale, delta=delta)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k, scale=scale, delta=delta)
        sob = np.sqrt(grad_x ** 2 + grad_y ** 2)
    elif direction == "X":
        sob = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k, scale=scale, delta=delta)
    else:  # "Y"
        sob = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k, scale=scale, delta=delta)

    sob = np.absolute(sob)
    sob = np.uint8(255 * sob / np.max(sob)) if np.max(sob) != 0 else np.zeros_like(sob, dtype=np.uint8)
    return sob


def apply_sobel_xy(gray, ksize=3, scale=1, delta=0):
    """Apply Sobel in both directions separately"""
    k = ensure_odd(ksize)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k, scale=scale, delta=delta)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k, scale=scale, delta=delta)

    grad_x = np.absolute(grad_x)
    grad_y = np.absolute(grad_y)

    grad_x = np.uint8(255 * grad_x / np.max(grad_x)) if np.max(grad_x) != 0 else np.zeros_like(grad_x, dtype=np.uint8)
    grad_y = np.uint8(255 * grad_y / np.max(grad_y)) if np.max(grad_y) != 0 else np.zeros_like(grad_y, dtype=np.uint8)

    return grad_x, grad_y


def apply_laplacian(gray, ksize=3, scale=1, delta=0):
    k = ensure_odd(ksize)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=k, scale=scale, delta=delta)
    lap = np.absolute(lap)
    lap = np.uint8(255 * lap / np.max(lap)) if np.max(lap) != 0 else np.zeros_like(lap, dtype=np.uint8)
    return lap


def apply_canny(gray, low=50, high=150):
    edges = cv2.Canny(gray, low, high)
    return edges


def create_comparison_grid(original, sobel, sobel_x, sobel_y, laplacian, canny):
    """Create a grid comparing all edge detection results"""
    # Convert all to RGB for display
    original_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB) if len(original.shape) == 2 else original

    sobel_rgb = cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB)
    sobel_x_rgb = cv2.cvtColor(sobel_x, cv2.COLOR_GRAY2RGB)
    sobel_y_rgb = cv2.cvtColor(sobel_y, cv2.COLOR_GRAY2RGB)
    laplacian_rgb = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB)
    canny_rgb = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)

    # Add labels to images
    def add_label(image, text):
        img_with_text = image.copy()
        cv2.putText(img_with_text, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img_with_text

    # Create comparison grid
    row1 = np.hstack([add_label(original_rgb, "Original"), add_label(sobel_rgb, "Sobel (Both)")])
    row2 = np.hstack([add_label(sobel_x_rgb, "Sobel X"), add_label(sobel_y_rgb, "Sobel Y")])
    row3 = np.hstack([add_label(laplacian_rgb, "Laplacian"), add_label(canny_rgb, "Canny")])

    grid = np.vstack([row1, row2, row3])
    return grid


# Processing functions
def process_single_algorithm():
    if not uploaded_file:
        return None

    img_pil = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(img_pil)
    gray = to_gray_cv(img_pil)

    # Apply blur if enabled
    if use_blur:
        gray = apply_gaussian_blur(gray, blur_ksize, blur_sigma)

    if algo == "Sobel":
        out = apply_sobel(gray, sobel_ksize, "Both", sobel_scale, sobel_delta)
    elif algo == "Laplacian":
        out = apply_laplacian(gray, lap_ksize, lap_scale, lap_delta)
    else:  # Canny
        out = apply_canny(gray, canny_low, canny_high)

    return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)


def process_comparison():
    if not uploaded_file:
        return None

    img_pil = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(img_pil)
    gray = to_gray_cv(img_pil)

    # Apply blur if enabled
    if use_blur:
        gray = apply_gaussian_blur(gray, blur_ksize, blur_sigma)

    # Apply all algorithms
    sobel_both = apply_sobel(gray, sobel_ksize, "Both", sobel_scale, sobel_delta)
    sobel_x, sobel_y = apply_sobel_xy(gray, sobel_ksize, sobel_scale, sobel_delta)
    laplacian = apply_laplacian(gray, lap_ksize, lap_scale, lap_delta)
    canny = apply_canny(gray, canny_low, canny_high)

    # Create comparison grid
    comparison_grid = create_comparison_grid(
        gray, sobel_both, sobel_x, sobel_y, laplacian, canny
    )

    return comparison_grid


# Main layout
if compare_all:
    # Full width for comparison view
    st.subheader("Edge Detection Algorithm Comparison")
    placeholder = st.empty()

    if uploaded_file:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(uploaded_file, caption="Original Image", use_column_width=True)
        with col2:
            st.markdown("### Algorithm Comparison Grid")
            comparison_placeholder = st.empty()
    else:
        st.info("Please upload an image to see the comparison.")
else:
    # Two-column layout for single algorithm
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input (Original)")
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        else:
            st.info("Please upload an image to begin.")

    with col2:
        st.subheader("Output (Edges)")
        placeholder = st.empty()


# Processing & update logic
def process_and_show():
    if not uploaded_file:
        if compare_all:
            comparison_placeholder.info("No image uploaded.") if 'comparison_placeholder' in locals() else None
        else:
            placeholder.info("No image uploaded.")
        return

    if compare_all:
        result = process_comparison()
        if result is not None:
            comparison_placeholder.image(result, use_column_width=True, clamp=True)

            # Add algorithm descriptions
            with st.expander("Algorithm Descriptions"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("""
                    **Sobel Operator**
                    - First-order derivative
                    - Directional (X, Y, or Both)
                    - Good for horizontal/vertical edges
                    - Fast computation
                    """)

                with col2:
                    st.markdown("""
                    **Laplacian Operator**
                    - Second-order derivative
                    - Rotation-invariant
                    - Sensitive to noise
                    - Finds zero-crossings
                    """)

                with col3:
                    st.markdown("""
                    **Canny Edge Detector**
                    - Multi-stage algorithm
                    - Low error rate
                    - Good localization
                    - Uses hysteresis thresholding
                    """)
    else:
        result = process_single_algorithm()
        if result is not None:
            placeholder.image(result, use_column_width=True)


# Decide when to update
if real_time:
    process_and_show()
else:
    if apply_btn:
        process_and_show()

# Performance metrics
if uploaded_file and (real_time or apply_btn):
    with st.expander("Performance Information"):
        st.markdown("""
        **Typical Performance Characteristics:**
        - **Sobel**: Fastest, suitable for real-time applications
        - **Laplacian**: Moderate speed, sensitive to noise
        - **Canny**: Slower but produces cleaner edges with better accuracy

        **Use Cases:**
        - **Sobel**: When speed is critical and directional edges matter
        - **Laplacian**: When you need rotation-invariant edge detection
        - **Canny**: When edge quality and accuracy are most important
        """)

# Footer
st.markdown("---")
st.markdown("""
**Edge Detection Comparison Notes:**
- Adjust parameters to see how each algorithm responds differently
- Use Gaussian blur to reduce noise before edge detection
- Compare how each method handles different edge types and noise levels
- Observe the trade-offs between sensitivity and noise immunity
""")