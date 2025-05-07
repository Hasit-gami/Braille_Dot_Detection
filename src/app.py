import streamlit as st
import PIL
from ultralytics import YOLO
from convert import convert_to_braille_unicode, parse_xywh_and_class, translate_braille_to_english
import numpy as np

st.title("Braille Detection and Recognition")

# Constants
MODEL_PATH = "./weights/yolov8_braille.pt"
DEFAULT_IMAGE_PATH = "./assets/braille dots.jpeg"
CONFIDENCE_THRESHOLD = 0.15

@st.cache_resource
def load_model():
    model = YOLO(MODEL_PATH)
    return model

def run_inference(model, image):
    res = model.predict(image, conf=CONFIDENCE_THRESHOLD)
    boxes = res[0].boxes
    list_boxes = parse_xywh_and_class(boxes)
    result = ""
    for box_line in list_boxes:
        str_left_to_right = ""
        box_classes = box_line[:, -1]
        for each_class in box_classes:
            str_left_to_right += convert_to_braille_unicode(model.names[int(each_class)])
        result += str_left_to_right + "\n"
    return result

model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file)
else:
    image = PIL.Image.open(DEFAULT_IMAGE_PATH)
    st.info("Using default image")

st.image(image, caption="Input Image", use_column_width=True)

if st.button("Run Inference"):
    with st.spinner("Running inference..."):
        braille_text = run_inference(model, image)
    st.success("Inference complete!")
    st.text_area("Detected Braille Unicode Text", braille_text, height=150)
    english_text = translate_braille_to_english(braille_text)
    st.text_area("Translated English Text", english_text, height=150)
