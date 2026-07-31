import streamlit as st
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

from convert import (
    convert_to_braille_unicode,
    parse_xywh_and_class,
    translate_braille_to_english,
)

MODEL_PATH = "./weights/yolov8_braille.pt"
CONFIDENCE_THRESHOLD = 0.15


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


def run_inference(model, image):
    results = model.predict(image, conf=CONFIDENCE_THRESHOLD)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return ""

    box_lines = parse_xywh_and_class(boxes)
    translated_lines = []

    for box_line in box_lines:
        line = ""

        for class_id in box_line[:, -1]:
            line += convert_to_braille_unicode(
                model.names[int(class_id)]
            )

        translated_lines.append(line)

    return "\n".join(translated_lines)


st.title("Braille Detection and Recognition")
st.write("Upload a clear JPG or PNG image containing Braille.")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.info("Upload an image to begin.")
    st.stop()

try:
    image = Image.open(uploaded_file).convert("RGB")
except (UnidentifiedImageError, OSError):
    st.error("The selected file is not a valid image.")
    st.stop()

st.image(
    image,
    caption="Input image",
    use_container_width=True,
)

if st.button("Run Inference", type="primary"):
    with st.spinner("Running Braille recognition..."):
        braille_text = run_inference(model, image)

    if not braille_text.strip():
        st.warning("No Braille characters were detected in this image.")
    else:
        english_text = translate_braille_to_english(braille_text)

        st.success("Inference complete!")
        st.text_area(
            "Detected Braille Unicode Text",
            braille_text,
            height=150,
        )
        st.text_area(
            "Translated English Text",
            english_text,
            height=150,
        )
