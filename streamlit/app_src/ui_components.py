import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_header(project_info: dict) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <span class="badge">50.039 Deep Learning</span>
            <h1>{project_info["title"]}</h1>
            <p>{project_info["subtitle"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dataset_summary(dataset_stats: dict) -> None:
    metric_cols = st.columns(4, gap="large")
    with metric_cols[0]:
        render_metric_card("Total Images", f'{dataset_stats["total_images"]:,}')
    with metric_cols[1]:
        render_metric_card("Melanoma Images", f'{dataset_stats["melanoma_count"]:,}', "Positive class label = 1")
    with metric_cols[2]:
        render_metric_card("Nevus Images", f'{dataset_stats["nevus_count"]:,}', "Negative class label = 0")
    with metric_cols[3]:
        render_metric_card(
            "Distribution",
            f'{dataset_stats["melanoma_pct"]:.1f}% / {dataset_stats["nevus_pct"]:.1f}%',
            "Melanoma vs Nevus",
        )

    st.markdown("### Model Notes")
    note_left, note_right = st.columns(2, gap="large")
    with note_left:
        st.info(dataset_stats["imbalance_note"])
    with note_right:
        st.success(dataset_stats["weighted_note"])


def plot_class_distribution(dataset_stats: dict) -> go.Figure:
    distribution_df = pd.DataFrame(
        {
            "Class": ["Melanoma", "Nevus"],
            "Count": [dataset_stats["melanoma_count"], dataset_stats["nevus_count"]],
            "Percentage": [dataset_stats["melanoma_pct"], dataset_stats["nevus_pct"]],
        }
    )
    fig = px.bar(
        distribution_df,
        x="Class",
        y="Count",
        color="Class",
        text="Percentage",
        color_discrete_sequence=["#d1495b", "#2d6a8e"],
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Number of Images",
    )
    return fig


def render_sample_gallery(title: str, image_paths: list, fallback_text: str) -> None:
    st.markdown(f"#### {title}")
    if not image_paths:
        render_missing_artifact(fallback_text)
        return

    gallery_cols = st.columns(3)
    for idx, image_path in enumerate(image_paths):
        with gallery_cols[idx % 3]:
            st.image(str(image_path), caption=image_path.name, width="stretch")


def build_model_comparison_dataframe(model_results: list[dict]) -> pd.DataFrame:
    rows = []
    for model in model_results:
        rows.append(
            {
                "Model": model["display_name"],
                "Val F2": model["metrics"]["val_f2"],
                "Test F2": model["metrics"]["f2_score"],
                "Recall": model["metrics"]["recall"],
                "Precision": model["metrics"]["precision"],
                "AUC": model["metrics"]["roc_auc"],
                "Best Model": model.get("is_best", False),
            }
        )
    return pd.DataFrame(rows).sort_values(by=["Best Model", "AUC"], ascending=[False, False])


def plot_confusion_matrix(model_result: dict) -> go.Figure:
    matrix = model_result["confusion_matrix"]
    fig = px.imshow(
        matrix,
        x=["Predicted Nevus", "Predicted Melanoma"],
        y=["Actual Nevus", "Actual Melanoma"],
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
    )
    fig.update_layout(
        title="Confusion Matrix",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        coloraxis_showscale=False,
    )
    return fig


def plot_training_curves(model_name: str, training_history: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=training_history["epoch"],
            y=training_history["train_loss"],
            mode="lines+markers",
            name="Train Loss",
            line=dict(color="#d1495b", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=training_history["epoch"],
            y=training_history["val_loss"],
            mode="lines+markers",
            name="Val Loss",
            line=dict(color="#2d6a8e", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=training_history["epoch"],
            y=training_history["train_accuracy"],
            mode="lines+markers",
            name="Train Accuracy",
            line=dict(color="#2a9d8f", width=3, dash="dot"),
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=training_history["epoch"],
            y=training_history["val_accuracy"],
            mode="lines+markers",
            name="Val Accuracy",
            line=dict(color="#f4a261", width=3, dash="dot"),
            yaxis="y2",
        )
    )
    fig.update_layout(
        title=model_name,
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Epoch",
        yaxis=dict(title="Loss"),
        yaxis2=dict(title="Accuracy", overlaying="y", side="right", range=[0, 1]),
    )
    return fig


def render_missing_artifact(message: str) -> None:
    st.markdown(
        f"""
        <div class="section-card">
            <strong>Placeholder</strong><br />
            <span>{message}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_result(prediction: dict) -> None:
    render_metric_card(
        "Prediction",
        prediction["predicted_label"],
        f'Confidence: {prediction["confidence"] * 100:.2f}%',
    )
