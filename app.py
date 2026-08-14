import io
import os

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


st.set_page_config(
    page_title="Bahraini Food Explorer",
    page_icon="🇧🇭",
    layout="wide"
)


FOOD_CLASSES = [
    "Sambosa",
    "Kubba",
    "Dates",
    "Balaleet",
    "Luqaimat",
    "Nekhy",
    "Khanfroosh",
    "Halwa",
    "Mattai",
    "Thareed"
]

FOOD_DESCRIPTIONS = {
    "Sambosa": "A crisp, triangular pastry filled with meat, cheese, or vegetables.",
    "Kubba": "A seasoned meat and bulgur shell, commonly fried and shaped like an oval.",
    "Dates": "Naturally sweet date-palm fruits served fresh or dried.",
    "Balaleet": "Sweet vermicelli flavored with saffron and cardamom, usually topped with egg.",
    "Luqaimat": "Small golden fried dough balls, often served with date syrup or honey.",
    "Nekhy": "Cooked chickpeas served warm with spices and lemon.",
    "Khanfroosh": "A soft, fragrant Bahraini fried cake flavored with saffron and cardamom.",
    "Halwa": "A glossy Bahraini sweet made with starch, sugar, saffron, nuts, and spices.",
    "Mattai": "A crunchy savory snack mixture commonly served with tea.",
    "Thareed": "Bread soaked in a rich meat-and-vegetable stew."
}

APP_FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_FOLDER, "best.pt")


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


def get_detections(result):
    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "Food": result.names[class_id],
            "Confidence": f"{confidence * 100:.1f}%",
            "Box": f"({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})"
        })

    return detections


def image_to_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


st.title("🇧🇭 Bahraini Food Explorer")
st.write(
    "Upload a meal photo or use your camera. The fine-tuned YOLO model will "
    "locate and identify the Bahraini foods it recognizes."
)

with st.sidebar:
    st.header("Detection Settings")
    confidence = st.slider(
        "Minimum confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.40,
        step=0.05
    )
    source = st.radio("Choose image source", ["Upload Image", "Use Camera"])

    st.header("Food Classes")
    for food in FOOD_CLASSES:
        st.write(f"• {food}")


if source == "Upload Image":
    image_file = st.file_uploader(
        "Upload an unseen food image",
        type=["jpg", "jpeg", "png"]
    )
else:
    image_file = st.camera_input("Take a photo of the food")


if image_file is None:
    st.info("Upload an image or take a camera photo to begin detection.")
elif not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Place your trained YOLO weights named "
        "best.pt in the same folder as app.py."
    )
else:
    image = Image.open(image_file).convert("RGB")

    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    try:
        with st.spinner("Detecting Bahraini food..."):
            model = load_model()
            result = model.predict(image, conf=confidence)[0]

        plotted_image = result.plot()[:, :, ::-1]
        annotated_image = Image.fromarray(plotted_image)
        detections = get_detections(result)

        with right_column:
            st.subheader("Detection Result")
            st.image(annotated_image, use_container_width=True)

        st.subheader("Detected Foods")

        if detections:
            detection_data = pd.DataFrame(detections)
            food_counts = detection_data["Food"].value_counts()

            count_columns = st.columns(min(len(food_counts), 4))
            for index, (food, count) in enumerate(food_counts.items()):
                count_columns[index % len(count_columns)].metric(food, int(count))

            st.dataframe(detection_data, use_container_width=True, hide_index=True)

            detected_names = list(dict.fromkeys(detection_data["Food"].tolist()))
            for food in detected_names:
                description = FOOD_DESCRIPTIONS.get(food.title())
                if description:
                    st.write(f"**{food}:** {description}")
        else:
            st.warning(
                "No selected food was detected. Try lowering the confidence "
                "or using a clearer, closer image."
            )

        st.download_button(
            "Download Annotated Image",
            data=image_to_bytes(annotated_image),
            file_name="bahraini_food_detection.jpg",
            mime="image/jpeg"
        )

    except Exception as error:
        st.error(f"The image could not be processed: {error}")


with st.expander("About this project"):
    st.write(
        "This prototype was created for a Bahraini food object-detection lab. "
        "It uses a fine-tuned YOLO model to detect multiple visible food items "
        "and place a bounding box around each detected object."
    )
