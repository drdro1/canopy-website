import concurrent.futures
import numpy as np
import streamlit as st
from PIL import Image
from pathlib import Path
import io
import os
import requests
import base64

PERCENT_MODEL_URL = "https://canopy-602911369526.europe-west1.run.app/predict-percent"
MASK_MODEL_URL = "https://canopy-602911369526.europe-west1.run.app/predict-mask"
# PERCENT_MODEL_URL = "http://localhost:8000/predict-percent"
# MASK_MODEL_URL = "http://localhost:8000/predict-mask"


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def save_variable_to_file(variable_name, variable_value):
    file_path = os.path.join(os.path.dirname(__file__), f"{variable_name}.txt")
    with open(file_path, "w") as file:
        file.write(str(variable_value))

def load_variable_from_file(variable_name):
    file_path = os.path.join(os.path.dirname(__file__), f"{variable_name}.txt")
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return file.read()
    else:
        return None

def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def get_image_files(country: str, type: str):
    dir_name = "pics/" + country + "_" + type
    image_files = sorted(Path(dir_name).glob("*.jpeg"))
    return image_files

def normalize_image(image):
    """Ensure image is converted to RGB and has consistent shape"""
    # Convert to RGB if it's grayscale
    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=-1)

    # If image has alpha channel, remove it
    if image.shape[2] == 4:
        image = image[:,:,:3]

    return image

def call_mask_model(image_path):
    try:
        response = requests.post(MASK_MODEL_URL, files={"file": open(image_path, "rb")})
        return response if response.status_code == 200 else None
    except requests.exceptions.RequestException:
        return None

def call_percent_model(image_path):
    try:
        response = requests.post(PERCENT_MODEL_URL, files={"file": open(image_path, "rb")})
        return response if response.status_code == 200 else None
    except requests.exceptions.RequestException:
        return None

def call_api_func(zone_name, image_files, button_col_value):
    cols = st.columns(len(image_files))
    startEnd = 'Start'
    for col_id, col in enumerate(cols):
        image_path = image_files[col_id]
        number = col_id + 1

        with col:
            st.image(str(image_path), use_container_width=True)
            button_cols = st.columns([1, button_col_value])
            with button_cols[1]:  # Use the second (right) column
                if st.button(f"{zone_name} {startEnd}", key=f"{zone_name}_{number}"):
                    st.session_state.selected_image_name = f'{zone_name} {number}'

                    with st.spinner("Predicting..."):
                        response = requests.post(MASK_MODEL_URL, files={"file": open(image_path, "rb")})

                    if response.status_code == 200:
                        data = response.json()

                        image_base64 = data["image"]
                        image_data = base64.b64decode(image_base64)

                        st.session_state.selected_image = Image.open(str(image_path))
                        st.session_state.selected_mask = Image.open(io.BytesIO(image_data))
                        st.session_state.percent_model_result = data["percent_cover"]
                        st.session_state.show_top = False
                        st.rerun()
                    else:
                        st.error(f"Error: {response.status_code}, {response.text}")
            startEnd = 'End' if startEnd == 'Start' else 'Start'

if "show_top" not in st.session_state:
    st.session_state.show_top = True
    st.session_state.project_type = load_variable_from_file('type')
    st.session_state.selected_image_name = None
    st.session_state.selected_image = None
    st.session_state.selected_mask = None
    st.session_state.percent_model_result = None

page_bg_color = '''
<style>
.stApp {
    background-color: black;
}
</style>
'''

col1, col2 = st.columns([1, 2])
with col1:
    st.image('pics/canopy_watch_logo.png', width=200, use_container_width=False)
with col2:
    st.title("Canopy Watch")

    if st.session_state.show_top:
        st.session_state.project_type = st.radio(
        "Select Project Type",
        ('Deforestation Project', 'Reforestation Project'),
        index=0 if st.session_state.project_type == 'Deforestation Project' else 1
        )

        st.markdown(
            """<style>
        div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
            font-size: 32px;
        }
            </style>
            """, unsafe_allow_html=True)


if st.session_state.show_top:
    if st.session_state.project_type == 'Deforestation Project':
        st.write(f"### Brazil")
        call_api_func('Brazil', get_image_files('brazil', 'defo'), 5)
        st.write(f"### Cameroon")
        call_api_func('Cameroon', get_image_files('cameroon', 'defo'), 10)
        st.write(f"### China")
        call_api_func('China', get_image_files('china', 'defo'), 5)
    else:
        st.write(f"### France")
        call_api_func('France', get_image_files('france', 'refo'), 5)
        st.write(f"### Mexico")
        call_api_func('Mexico', get_image_files('mexico', 'refo'), 5)
        st.write(f"### Peru")
        call_api_func('Peru', get_image_files('peru', 'refo'), 4)

if not st.session_state.show_top:
    title = f'{st.session_state.project_type} : {st.session_state.selected_image_name}'
    st.title(title)
    st.markdown(f"### Percentage Prediction: {st.session_state.percent_model_result}")
    # st.write("Prediction:", st.session_state.percent_model_result)

    image1 = st.session_state.selected_image
    image2 = st.session_state.selected_mask

    # Convert images to numpy arrays
    img1_array = normalize_image(np.array(image1))
    img2_array = normalize_image(np.array(image2))

     # Resize images to match
    height, width = min(img1_array.shape[0], img2_array.shape[0]), min(img1_array.shape[1], img2_array.shape[1])
    img1_array = img1_array[:height, :width]
    img2_array = img2_array[:height, :width]

    # Slider to control overlay
    overlay_percentage = st.slider(
        "Image Overlay",
        min_value=0,
        max_value=100,
        value=1,
        step=1
    )

    # Calculate blend
    blend_weight = overlay_percentage / 100
    blended_image = (
        img1_array * (1 - blend_weight) +
        img2_array * blend_weight
    ).astype(np.uint8)

    col1, col2, col3 = st.columns([3, 6, 2])
    with col2:
        st.image(blended_image, width=300, caption="Blended Image")
    with col3:
        if st.button(f"Go Back"):
            save_variable_to_file('type', st.session_state.project_type)
            st.session_state.show_top = True
            st.rerun()

    col1, col2, col3 = st.columns([3, 1, 2])
    with col1:
        st.image(st.session_state.selected_image, caption="Original Image", width=200)
    with col3:
        st.image(st.session_state.selected_mask, caption="Binary Mask", width=200)
