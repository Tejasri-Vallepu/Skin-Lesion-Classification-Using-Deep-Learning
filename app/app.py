
import streamlit as st
import pandas as pd

from pathlib import Path
from PIL import Image

from utils import (
    predict_image,
    generate_image_description,
    CLASS_NAMES
)


# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="SkinVision AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------
# LOAD CUSTOM CSS
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
CSS_PATH = Path(__file__).parent / "style.css"

if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as css_file:
        st.markdown(
            f"<style>{css_file.read()}</style>",
            unsafe_allow_html=True
        )


# --------------------------------
# SIDEBAR
# --------------------------------

with st.sidebar:

    st.markdown("## 🔬 SkinVision AI")

    st.caption("AI-powered image research dashboard")

    st.divider()

    st.markdown("### Navigation")
    st.markdown("🏠  Dashboard")
    st.markdown("🧠  CNN Classification")
    st.markdown("👁️  Vision Description")

    st.divider()

    st.markdown("### Model Information")

    st.info(
        """
        **CNN:** Fine-tuned MobileNetV2

        **VLM:** LLaVA 7B

        **Dataset:** HAM10000

        **Classes:** 7
        """
    )

    st.divider()

    st.caption("Developed for educational research.")


# --------------------------------
# HERO SECTION
# --------------------------------

st.markdown(
    """
    <div class="hero">
        <h2>SkinVision AI Dashboard</h2>
        <p>
        Explore deep learning image classification and
        vision-language descriptions using HAM10000.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------
# DISCLAIMER
# --------------------------------

st.warning(
    """
    **Research & Education Only**

    This application is not a medical diagnostic tool.
    Its model scores and generated descriptions are experimental
    outputs, not clinical findings or medical advice.
    """
)


# --------------------------------
# UPLOAD SECTION
# --------------------------------

st.markdown("## 📤 Upload Your Image")

st.write(
    "Upload a JPG, JPEG, or PNG image to explore both AI models."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.divider()

    # --------------------------------
    # IMAGE PREVIEW
    # --------------------------------

    st.markdown("## 🖼️ Image Preview")

    image_col, details_col = st.columns([1.2, 1])

    with image_col:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with details_col:

        st.markdown("### Image Details")

        st.markdown(
            f"""
            <div class="info-card">
                <p><b>File name</b></p>
                <p>{uploaded_file.name}</p>
                <hr>
                <p><b>Image dimensions</b></p>
                <p>{image.width} × {image.height}</p>
                <hr>
                <p><b>Image format</b></p>
                <p>{uploaded_file.type}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        analyze_button = st.button(
            "🚀 Analyze Image",
            type="primary",
            use_container_width=True
        )


    # --------------------------------
    # MODEL ANALYSIS
    # --------------------------------

    if analyze_button:

        st.divider()

        # CNN SECTION
        st.markdown("## 🧠 MobileNetV2 Classification")

        try:

            with st.spinner("Running MobileNetV2 model..."):

                predicted_class, predicted_score, scores = (
                    predict_image(image)
                )

            st.success("Classification completed!")

            metric1, metric2 = st.columns(2)

            with metric1:
                st.metric(
                    "Highest-scoring class",
                    predicted_class
                )

            with metric2:
                st.metric(
                    "Highest model score",
                    f"{predicted_score * 100:.2f}%"
                )

            score_df = pd.DataFrame({
                "Class": CLASS_NAMES,
                "Model Score": scores
            }).sort_values(
                "Model Score",
                ascending=False
            )

            chart_col, table_col = st.columns([1.2, 1])

            with chart_col:

                st.markdown("### 📊 Class Score Chart")

                st.bar_chart(
                    score_df.set_index("Class"),
                    y="Model Score"
                )

            with table_col:

                st.markdown("### 📋 Model Scores")

                st.dataframe(
                    score_df.style.format({
                        "Model Score": "{:.4f}"
                    }),
                    use_container_width=True,
                    hide_index=True
                )

            st.caption(
                "Scores are model outputs, not calibrated medical probabilities."
            )

        except Exception as error:

            st.error(f"Classification error: {error}")


        # VLM SECTION
        st.divider()

        st.markdown("## 👁️ LLaVA Vision Description")

        st.write(
            "Generate a general visual description of the uploaded image."
        )

        try:

            with st.spinner(
                "LLaVA 7B is analyzing the image..."
            ):

                description = generate_image_description(image)

            st.markdown(
                f"""
                <div class="info-card">
                    <h3>✨ Generated Image Description</h3>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.caption(
                "This is an AI-generated visual description, not a diagnosis."
            )

        except Exception as error:

            st.error(
                "LLaVA description could not be generated. "
                "Check that Ollama is running and llava:7b is installed."
            )

            st.caption(str(error))


else:

    st.info(
        "👆 Upload an image to begin your AI experiment."
    )


# --------------------------------
# MODEL OVERVIEW
# --------------------------------

st.divider()

st.markdown("## ⚙️ Project Architecture")

architecture_col1, architecture_col2 = st.columns(2)

with architecture_col1:

    st.markdown(
        """
        <div class="info-card">
            <h3>🧠 MobileNetV2</h3>
            <p><b>Task:</b> Seven-class image classification</p>
            <p><b>Framework:</b> TensorFlow / Keras</p>
            <p><b>Input:</b> 224 × 224 RGB image</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with architecture_col2:

    st.markdown(
        """
        <div class="info-card">
            <h3>👁️ LLaVA 7B</h3>
            <p><b>Task:</b> General visual description</p>
            <p><b>Runtime:</b> Ollama</p>
            <p><b>Input:</b> Uploaded image</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------
# FOOTER
# --------------------------------

st.divider()

st.markdown(
    """
    <p class="small-note" style="text-align:center;">
        SkinVision AI | HAM10000 Research Project |
        Built with Streamlit, TensorFlow and Ollama
    </p>
    """,
    unsafe_allow_html=True
)
