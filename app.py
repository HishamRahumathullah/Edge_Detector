import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Configure the page
st.set_page_config(page_title="Edge Detection UI", layout="wide")

st.title("🖼️ Interactive Edge Detection UI")
st.markdown("Upload an image and experiment with different edge detection algorithms!")

# Initialize session state for image
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

## **Step 3: Image Upload Functionality**

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'bmp'])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.session_state.uploaded_image = image

    # Display original image
    st.subheader("Original Image")
    st.image(image, channels="BGR", use_column_width=True)

    # Convert to grayscale for processing
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ## **Step 4: Algorithm Selection**
    st.subheader("Edge Detection Settings")

    algorithm = st.selectbox(
        "Select Edge Detection Algorithm:",
        ["Sobel", "Laplacian", "Canny"]
    )

    ## **Step 5: Parameter Controls Based on Algorithm**

    col1, col2 = st.columns(2)

    with col1:
        if algorithm == "Sobel":
            st.markdown("**Sobel Parameters**")
            sobel_kernel = st.slider("Kernel Size", 1, 7, 3, step=2)
            sobel_direction = st.selectbox("Gradient Direction", ["X", "Y", "Both"])

        elif algorithm == "Laplacian":
            st.markdown("**Laplacian Parameters**")
            laplacian_kernel = st.slider("Kernel Size", 1, 7, 3, step=2)

        elif algorithm == "Canny":
            st.markdown("**Canny Parameters**")
            canny_low = st.slider("Low Threshold", 0, 255, 50)
            canny_high = st.slider("High Threshold", 0, 255, 150)
            canny_kernel = st.slider("Kernel Size", 3, 7, 3, step=2)
            canny_sigma = st.slider("Sigma (Gaussian Blur)", 0.1, 5.0, 1.0, step=0.1)

    with col2:
        ## **Step 6: Apply Selected Algorithm**
        st.markdown("**Edge Detection Result**")

        try:
            if algorithm == "Sobel":
                # Apply Sobel
                if sobel_direction == "X":
                    edges = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
                elif sobel_direction == "Y":
                    edges = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
                else:  # Both
                    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
                    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
                    edges = cv2.magnitude(sobel_x, sobel_y)

                # Convert to absolute value and scale to 0-255
                edges = np.absolute(edges)
                edges = np.uint8(255 * edges / np.max(edges)) if np.max(edges) > 0 else np.uint8(edges)

            elif algorithm == "Laplacian":
                # Apply Laplacian
                edges = cv2.Laplacian(gray_image, cv2.CV_64F, ksize=laplacian_kernel)
                edges = np.absolute(edges)
                edges = np.uint8(255 * edges / np.max(edges)) if np.max(edges) > 0 else np.uint8(edges)

            elif algorithm == "Canny":
                # Apply Gaussian blur first
                blurred = cv2.GaussianBlur(gray_image, (canny_kernel, canny_kernel), canny_sigma)
                # Apply Canny
                edges = cv2.Canny(blurred, canny_low, canny_high)

            # Display the result
            st.image(edges, use_column_width=True, caption=f"{algorithm} Edge Detection")

            # Show current parameters
            st.markdown("**Current Parameters:**")
            if algorithm == "Sobel":
                st.write(f"Kernel Size: {sobel_kernel}, Direction: {sobel_direction}")
            elif algorithm == "Laplacian":
                st.write(f"Kernel Size: {laplacian_kernel}")
            elif algorithm == "Canny":
                st.write(
                    f"Low Threshold: {canny_low}, High Threshold: {canny_high}, Kernel: {canny_kernel}, Sigma: {canny_sigma}")

        except Exception as e:
            st.error(f"Error applying {algorithm}: {str(e)}")

else:
    st.info("Please upload an image to get started!")