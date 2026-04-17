from pathlib import Path

import pandas as pd
import streamlit as st

from app_src.config import (
    CNN_TABLE,
    DATASET_STATS,
    DENSENET_TABLE,
    EFFICIENTNET_TABLE,
    METADATA_LOCALIZATION_OPTIONS,
    METADATA_SEX_OPTIONS,
    MOBILENET_TABLE,
    MODEL_RESULTS,
    PROJECT_INFO,
    RESNET_TABLE,
    VIT_TABLE,
)
from app_src.model_utils import (
    discover_sample_images,
    load_model,
    predict_image,
    read_uploaded_image,
)
from app_src.ui_components import (
    build_model_comparison_dataframe,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_training_curves,
    render_dataset_summary,
    render_header,
    render_metric_card,
    render_missing_artifact,
    render_prediction_result,
    render_sample_gallery,
)

MODEL_TAB_CONFIG = [
    {
        "label": "CNN",
        "table_data": CNN_TABLE,
    },
    {
        "label": "ResNet",
        "table_data": RESNET_TABLE,
    },
    {
        "label": "EfficientNet",
        "table_data": EFFICIENTNET_TABLE,
    },
    {
        "label": "MobileNet",
        "table_data": MOBILENET_TABLE,
    },
    {
        "label": "DenseNet",
        "table_data": DENSENET_TABLE,
    },
    {
        "label": "ViT",
        "table_data": VIT_TABLE,
    },
]


st.set_page_config(
    page_title="Skin Lesion Classification Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .main {
                background:
                    radial-gradient(circle at top right, rgba(57, 106, 177, 0.10), transparent 28%),
                    radial-gradient(circle at bottom left, rgba(218, 124, 48, 0.10), transparent 22%),
                    #f6f8fb;
            }
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            .metric-card {
                background: linear-gradient(145deg, #ffffff, #f8fbff);
                border: 1px solid #dde6f2;
                border-radius: 18px;
                padding: 1.05rem 1rem;
                box-shadow: 0 10px 25px rgba(18, 44, 77, 0.07);
                min-height: 118px;
            }
            .metric-label {
                color: #5d6b82;
                font-size: 0.9rem;
                margin-bottom: 0.35rem;
            }
            .metric-value {
                color: #10233f;
                font-size: 1.7rem;
                font-weight: 700;
                line-height: 1.2;
            }
            .metric-help {
                color: #73839a;
                font-size: 0.82rem;
                margin-top: 0.5rem;
            }
            .section-card {
                background: rgba(255, 255, 255, 0.92);
                border: 1px solid #e4ebf5;
                border-radius: 20px;
                padding: 1.2rem;
                box-shadow: 0 10px 24px rgba(18, 44, 77, 0.05);
            }
            .hero {
                padding: 1.35rem 1.5rem;
                background: linear-gradient(135deg, #10233f, #1f4e79);
                color: white;
                border-radius: 22px;
                margin-bottom: 1rem;
                box-shadow: 0 14px 30px rgba(16, 35, 63, 0.22);
            }
            .hero h1 {
                margin: 0;
                font-size: 2.05rem;
            }
            .hero p {
                margin: 0.55rem 0 0 0;
                color: #dce8f5;
                font-size: 1rem;
            }
            .badge {
                display: inline-block;
                padding: 0.25rem 0.6rem;
                background: rgba(255, 255, 255, 0.14);
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 999px;
                font-size: 0.82rem;
                margin-right: 0.45rem;
            }
            .best-model {
                border-left: 6px solid #2a9d8f;
                background: linear-gradient(145deg, rgba(42, 157, 143, 0.14), rgba(255, 255, 255, 0.96));
            }
            .interpret-box {
                border-radius: 16px;
                padding: 1rem 1.1rem;
                background: #fff8ec;
                border: 1px solid #f0d7ab;
                color: #5a4522;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_dataset_tab() -> None:
    render_dataset_summary(DATASET_STATS)

    additional_stats = pd.DataFrame(
        [
            {
                "Statistic": key.replace("_", " ").title(),
                "Value": str(value),
            }
            for key, value in DATASET_STATS["additional_statistics"].items()
        ]
    )
    st.markdown("#### Additional Statistics")
    st.dataframe(additional_stats, width="stretch", hide_index=True)

    st.markdown("### Sample Images")
    melanoma_images = discover_sample_images(Path("streamlit/assets/sample_images/melanoma"))
    nevus_images = discover_sample_images(Path("streamlit/assets/sample_images/nevus"))
    gallery_left, gallery_right = st.columns(2, gap="large")
    with gallery_left:
        render_sample_gallery("Melanoma Samples", melanoma_images, fallback_text="Add melanoma sample images to `assets/sample_images/melanoma/`.")
    with gallery_right:
        render_sample_gallery("Nevus Samples", nevus_images, fallback_text="Add nevus sample images to `assets/sample_images/nevus/`.")


def render_model_statistics_tab() -> None:
    comparison_df = build_model_comparison_dataframe(MODEL_RESULTS)
    best_model = next((model for model in MODEL_RESULTS if model.get("is_best")), MODEL_RESULTS[0])

    top_cols = st.columns(3, gap="large")
    with top_cols[0]:
        render_metric_card("Best Model", best_model["display_name"], "Based on your selected evaluation metrics.")
    with top_cols[1]:
        render_metric_card("Best ROC-AUC", f'{best_model["metrics"]["roc_auc"]:.3f}', "Higher values indicate better class separation.")
    with top_cols[2]:
        render_metric_card("Best F2-Score", f'{best_model["metrics"]["f2_score"]:.3f}', "F2-score emphasizes recall more strongly than precision.")

    st.markdown("### Model Comparison Table")
    styled_df = comparison_df.copy()
    styled_df["Best Model"] = styled_df["Best Model"].map(lambda value: "Yes" if value else "")
    st.dataframe(styled_df, width="stretch", hide_index=True)


def render_architecture_tab(table_data) -> None:
    columns = list(table_data[0].keys()) if table_data else []
    table_df = pd.DataFrame(table_data, columns=columns)
    st.dataframe(table_df, width="stretch", hide_index=True)


def render_prediction_tab() -> None:
    best_models = [model for model in MODEL_RESULTS if model.get("is_best")]
    model_options = {model["display_name"]: model for model in best_models}
    left_col, right_col = st.columns([0.95, 1.25], gap="large")

    with left_col:
        st.markdown("### Inference Controls")
        selected_display_name = st.selectbox("Select a model", options=list(model_options.keys()))
        selected_model = model_options[selected_display_name]
        uploaded_file = st.file_uploader(
            "Upload a skin lesion image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG",
        )
        st.markdown("#### Patient Metadata")
        age = st.number_input("Age", min_value=0, max_value=120, value=52, step=1)
        sex = st.selectbox("Sex", options=METADATA_SEX_OPTIONS, index=2)
        localization = st.selectbox("Lesion Location", options=METADATA_LOCALIZATION_OPTIONS, index=8)
        run_prediction = st.button("Run Prediction", type="primary", width="stretch")

    with right_col:
        st.markdown("### Prediction Output")
        if uploaded_file is None:
            render_missing_artifact("Upload an image to preview the lesion and run inference.")
            return

        image = read_uploaded_image(uploaded_file)
        if image is None:
            st.error("The uploaded file could not be read as an image. Please try another file.")
            return

        st.image(image, caption="Uploaded Image", width=320)

        if not run_prediction:
            return

        with st.spinner("Loading model and running inference..."):
            model, device, load_message = load_model(selected_model)
            if model is None:
                st.error(load_message)
                return

            metadata_input = {
                "age": age,
                "sex": sex,
                "localization": localization,
            }
            prediction = predict_image(model, image, device, selected_model, metadata_input=metadata_input)
            if prediction["status"] != "success":
                st.error(prediction["message"])
                return

        result_cols = st.columns(2, gap="large")
        with result_cols[0]:
            render_prediction_result(prediction)


def main() -> None:
    inject_styles()
    render_header(PROJECT_INFO)

    tab_labels = ["Dataset Overview", "Model Statistics"]
    tab_labels.extend(config["label"] for config in MODEL_TAB_CONFIG)
    tab_labels.append("Prediction")
    tabs = st.tabs(tab_labels)

    overview_tab = tabs[0]
    models_tab = tabs[1]
    architecture_tabs = tabs[2:-1]
    prediction_tab = tabs[-1]

    with overview_tab:
        render_dataset_tab()
    with models_tab:
        render_model_statistics_tab()
    for tab, tab_config in zip(architecture_tabs, MODEL_TAB_CONFIG):
        with tab:
            render_architecture_tab(tab_config["table_data"])
    with prediction_tab:
        render_prediction_tab()


if __name__ == "__main__":
    main()
