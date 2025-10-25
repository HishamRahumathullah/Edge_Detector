# streamlit_app.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

st.set_page_config(layout="wide", page_title="Interactive Edge Detection UI")

st.title("Interactive Edge Detection UI")
st.markdown("Upload an image and experiment with Sobel, Laplacian and Canny edge detectors.")

# Sidebar: upload and algorithm selector
with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload an image (JPG, PNG, BMP)", type=["jpg","jpeg","png","bmp"])
    algo = st.radio("Select algorithm", ("Sobel", "Laplacian", "Canny"))

    st.markdown("### Algorithm parameters")
    # Canny params
    if algo == "Canny":
        canny_low = st.slider("Canny - Lower threshold", 0, 255, 50)
        canny_high = st.slider("Canny - Upper threshold", 0, 255, 150)
        gaussian_ksize = st.slider("Gaussian kernel size (odd)", 1, 31, 5)
        gaussian_sigma = st.slider("Gaussian sigma", 0.0, 10.0, 1.0, step=0.1)
        st.write(f"Current: low={canny_low}, high={canny_high}, ksize={gaussian_ksize}, sigma={gaussian_sigma}")
    # Sobel params
    elif algo == "Sobel":
        sobel_ksize = st.slider("Sobel kernel size (odd):", 1, 31, 3)
        sobel_dir = st.selectbox("Gradient direction", ("X", "Y", "Both"))
        st.write(f"Current: ksize={sobel_ksize}, dir={sobel_dir}")
    # Laplacian params
    else:
        lap_ksize = st.slider("Laplacian kernel size (odd)", 1, 31, 3)
        st.write(f"Current: ksize={lap_ksize}")

    st.markdown("---")
    real_time = st.checkbox("Real-time update", value=True)
    apply_btn = st.button("Apply (if real-time off)")

# Helper functions
def ensure_odd(k):
    k = int(k)
    return k if k % 2 == 1 else (k-1 if k>1 else 1)

def to_gray_cv(img_pil):
    img = np.array(img_pil.convert("RGB"))
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def apply_sobel(gray, ksize, direction):
    k = ensure_odd(ksize)
    dx = 1 if direction in ("X","Both") else 0
    dy = 1 if direction in ("Y","Both") else 0
    # OpenCV Sobel returns float, convert to abs then 8-bit
    sob = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=k)
    sob = np.absolute(sob)
    sob = np.uint8(255 * sob / np.max(sob)) if np.max(sob)!=0 else np.zeros_like(sob, dtype=np.uint8)
    return sob

def apply_laplacian(gray, ksize):
    k = ensure_odd(ksize)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=k)
    lap = np.absolute(lap)
    lap = np.uint8(255 * lap / np.max(lap)) if np.max(lap)!=0 else np.zeros_like(lap, dtype=np.uint8)
    return lap

def apply_canny(gray, low, high, gksize, sigma):
    k = ensure_odd(gksize)
    # apply gaussian blur if k>1
    if k > 1:
        gray_blur = cv2.GaussianBlur(gray, (k,k), sigmaX=sigma)
    else:
        gray_blur = gray
    edges = cv2.Canny(gray_blur, low, high)
    return edges

# Main layout: two columns for Input and Output
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
        placeholder.info("No image uploaded.")
        return

    img_pil = Image.open(uploaded_file).convert("RGB")
    gray = to_gray_cv(img_pil)

    if algo == "Sobel":
        out = apply_sobel(gray, sobel_ksize, sobel_dir)
    elif algo == "Laplacian":
        out = apply_laplacian(gray, lap_ksize)
    else:
        out = apply_canny(gray, canny_low, canny_high, gaussian_ksize, gaussian_sigma)

    # Convert single channel to RGB for display
    out_rgb = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    out_pil = Image.fromarray(out_rgb)
    placeholder.image(out_pil, use_column_width=True)

# Decide when to update
if real_time:
    process_and_show()
else:
    if apply_btn:
        process_and_show()

# Footer / help
st.markdown("---")
st.markdown("*Notes:* kernel sizes must be odd numbers. Adjust parameters to see differences between algorithms.")