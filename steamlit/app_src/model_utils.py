import sys
from pathlib import Path
from typing import Any

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app_src.config import (
    ALLOWED_IMAGE_SUFFIXES,
    CLASS_LABELS,
    INPUT_SIZE,
    METADATA_LOCALIZATION_OPTIONS,
)
from src.models.efficientnet import EfficientNetB0WithMetadata
from src.models.mobilenet import MobileNetV3LargeWithMetadata


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_tta_transforms() -> list[transforms.Compose]:
    def _base(extra: list | None = None) -> transforms.Compose:
        ops = [transforms.Resize(INPUT_SIZE)]
        if extra:
            ops.extend(extra)
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return transforms.Compose(ops)

    return [
        _base(),
        _base([transforms.RandomHorizontalFlip(p=1.0)]),
        _base([transforms.RandomVerticalFlip(p=1.0)]),
        _base([transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p=1.0)]),
        _base([transforms.RandomRotation(degrees=(90, 90))]),
        _base([transforms.RandomRotation(degrees=(180, 180))]),
        _base([transforms.RandomRotation(degrees=(270, 270))]),
        _base([transforms.RandomRotation(degrees=(45, 45))]),
    ]


def discover_sample_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return [path for path in sorted(folder.iterdir()) if path.suffix.lower() in ALLOWED_IMAGE_SUFFIXES][:6]


def read_uploaded_image(uploaded_file: Any) -> Image.Image | None:
    try:
        return Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None


def build_metadata_tensor(metadata_input: dict, device: torch.device) -> torch.Tensor:
    age = float(metadata_input.get("age", 52)) / 100.0
    sex = metadata_input.get("sex", "unknown")
    localization = metadata_input.get("localization", "unknown")

    sex_map = {"male": 0.0, "female": 1.0, "unknown": 0.5}
    metadata_features = [age, sex_map.get(sex, 0.5)]
    metadata_features.extend(float(localization == category) for category in METADATA_LOCALIZATION_OPTIONS)

    return torch.tensor(metadata_features, dtype=torch.float32, device=device).unsqueeze(0)


def create_model_architecture(model_config: dict) -> torch.nn.Module:
    architecture = model_config.get("architecture")

    if architecture == "efficientnet_b0_with_metadata":
        return EfficientNetB0WithMetadata(
            metadata_dim=model_config.get("metadata_dim", 17),
            num_classes=model_config.get("num_classes", 1),
            freeze_backbone=model_config.get("freeze_backbone", True),
            dropout=model_config.get("dropout", 0.5),
        )

    if architecture == "mobilenet_v3_large_with_metadata":
        return MobileNetV3LargeWithMetadata(
            metadata_dim=model_config.get("metadata_dim", 17),
            num_classes=model_config.get("num_classes", 1),
            freeze_backbone=model_config.get("freeze_backbone", False),
            dropout=model_config.get("dropout", 0.5),
        )

    raise NotImplementedError(
        f"Unsupported architecture `{architecture}`. Add it to `create_model_architecture()` "
        "in `app_src/model_utils.py`."
    )


@st.cache_resource(show_spinner=False)
def load_model(model_config: dict) -> tuple[torch.nn.Module | dict | None, torch.device, str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        checkpoint_type = model_config.get("checkpoint_type", "full_model")
        if checkpoint_type == "ensemble_state_dict":
            component_models = []
            component_paths = []

            for component_config in model_config.get("component_configs", []):
                model_path = Path(component_config["file_path"])
                if not model_path.exists():
                    return None, device, f"Model file not found at `{model_path}`. Add your trained model to this path first."

                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                component_model = create_model_architecture(component_config)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    component_model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    component_model.load_state_dict(checkpoint)

                component_model.eval()
                component_model.to(device)
                component_models.append(component_model)
                component_paths.append(str(model_path))

            return {
                "type": "ensemble",
                "models": component_models,
                "use_tta": model_config.get("use_tta", False),
                "decision_threshold": float(model_config.get("decision_threshold", 0.5)),
            }, device, f"Loaded ensemble from {', '.join(component_paths)}."

        model_path = Path(model_config["file_path"])
        if not model_path.exists():
            return None, device, f"Model file not found at `{model_path}`. Add your trained model to this path first."

        if checkpoint_type == "full_model":
            model = torch.load(model_path, map_location=device, weights_only=False)
        else:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model = create_model_architecture(model_config)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

        model.eval()
        model.to(device)
        return model, device, f"Loaded model from `{model_path}`."
    except NotImplementedError as exc:
        return None, device, str(exc)
    except Exception as exc:
        return None, device, f"Failed to load model from `{model_path}`. Details: {exc}"


def predict_image(
    model: torch.nn.Module | dict,
    image: Image.Image,
    device: torch.device,
    model_config: dict,
    metadata_input: dict | None = None,
) -> dict:
    transform = build_transform()

    try:
        metadata_tensor = build_metadata_tensor(metadata_input, device) if metadata_input is not None else None

        if isinstance(model, dict) and model.get("type") == "ensemble":
            threshold = float(model.get("decision_threshold", 0.5))
            inference_transforms = build_tta_transforms() if model.get("use_tta") else [transform]
            component_probabilities = []

            with torch.inference_mode():
                for component_model in model["models"]:
                    tta_probabilities = []
                    for inference_transform in inference_transforms:
                        image_tensor = inference_transform(image).unsqueeze(0).to(device)
                        logits = component_model(image_tensor, metadata_tensor)
                        probability = torch.sigmoid(logits).reshape(-1)[0]
                        tta_probabilities.append(probability)

                    component_probabilities.append(torch.stack(tta_probabilities).mean())

                melanoma_probability_tensor = torch.stack(component_probabilities).mean().unsqueeze(0)
                nevus_probability_tensor = 1.0 - melanoma_probability_tensor
                predicted_index = (melanoma_probability_tensor >= threshold).long()
                confidence = torch.where(
                    predicted_index == 1,
                    melanoma_probability_tensor,
                    nevus_probability_tensor,
                )
        else:
            image_tensor = transform(image).unsqueeze(0).to(device)
            with torch.inference_mode():
                if metadata_tensor is not None:
                    logits = model(image_tensor, metadata_tensor)
                else:
                    logits = model(image_tensor)

                if isinstance(logits, (tuple, list)):
                    logits = logits[0]

                if logits.ndim == 2 and logits.shape[1] == 1:
                    threshold = float(model_config.get("decision_threshold", 0.5))
                    melanoma_probability_tensor = torch.sigmoid(logits).squeeze(1)
                    nevus_probability_tensor = 1.0 - melanoma_probability_tensor
                    predicted_index = (melanoma_probability_tensor >= threshold).long()
                    confidence = torch.where(
                        predicted_index == 1,
                        melanoma_probability_tensor,
                        nevus_probability_tensor,
                    )
                else:
                    probabilities = torch.softmax(logits, dim=1)
                    confidence, predicted_index = torch.max(probabilities, dim=1)
                    melanoma_probability_tensor = probabilities[:, 1]
                    nevus_probability_tensor = probabilities[:, 0]

        predicted_label = CLASS_LABELS[int(predicted_index.item())]
        melanoma_probability = float(melanoma_probability_tensor.item())
        nevus_probability = float(nevus_probability_tensor.item())

        return {
            "status": "success",
            "model_name": model_config["display_name"],
            "predicted_index": int(predicted_index.item()),
            "predicted_label": predicted_label,
            "confidence": float(confidence.item()),
            "melanoma_probability": melanoma_probability,
            "nevus_probability": nevus_probability,
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Inference failed. Please verify your model output shape and preprocessing logic. Details: {exc}",
        }
