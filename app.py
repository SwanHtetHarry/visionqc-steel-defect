from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="VisionQC Steel Defect Inspector",
    page_icon="🔍",
    layout="centered"
)


# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

MODEL_PATH = Path(
    "models/final_resnet18_finetuned_best.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DESCRIPTIONS = {
    "crazing":
        "Fine crack-like patterns that appear across the steel surface.",

    "inclusion":
        "Foreign material or impurities embedded in the steel surface.",

    "patches":
        "Irregular surface regions with noticeable texture or appearance differences.",

    "pitted_surface":
        "Small pits or depressions distributed across the steel surface.",

    "rolled-in_scale":
        "Oxide scale that becomes pressed into the steel during rolling.",

    "scratches":
        "Linear surface damage caused by mechanical contact or abrasion."
}


# --------------------------------------------------
# IMAGE TRANSFORM
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = models.resnet18(
        weights=None
    )

    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        len(CLASS_NAMES)
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)
    model.eval()

    return model


model = load_model()


# --------------------------------------------------
# GRAD-CAM
# --------------------------------------------------

def generate_gradcam(
    model,
    image,
    target_layer
):

    activations_store = {}
    gradients_store = {}

    def forward_hook(module, input, output):
        activations_store["value"] = output

    def backward_hook(
        module,
        grad_input,
        grad_output
    ):
        gradients_store["value"] = grad_output[0]

    forward_handle = (
        target_layer.register_forward_hook(
            forward_hook
        )
    )

    backward_handle = (
        target_layer.register_full_backward_hook(
            backward_hook
        )
    )

    input_tensor = (
        transform(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    model.zero_grad()

    outputs = model(input_tensor)

    predicted_index = (
        outputs.argmax(dim=1).item()
    )

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    confidence = probabilities[
        0,
        predicted_index
    ].item()

    score = outputs[
        0,
        predicted_index
    ]

    score.backward()

    activations = (
        activations_store["value"]
    )

    gradients = (
        gradients_store["value"]
    )

    weights = gradients.mean(
        dim=(2, 3),
        keepdim=True
    )

    cam = (
        weights * activations
    ).sum(
        dim=1,
        keepdim=True
    )

    cam = F.relu(cam)

    cam = F.interpolate(
        cam,
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )

    cam = cam.squeeze()

    cam = cam - cam.min()

    cam = cam / (
        cam.max() + 1e-8
    )

    gradcam = (
        cam.detach()
        .cpu()
        .numpy()
    )

    forward_handle.remove()
    backward_handle.remove()

    return (
        predicted_index,
        confidence,
        gradcam
    )


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("VisionQC Steel Defect Inspector")

st.write(
    "Upload a steel surface image and the AI model will "
    "predict the defect type and show which area most "
    "influenced the prediction."
)

st.divider()


# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload steel surface image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp"
    ]
)


# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.markdown("### Selected Image")

    st.image(
        image,
        width=350
    )

    analyze_button = st.button(
        "Analyze Image",
        type="primary"
    )

    if analyze_button:

        with st.spinner(
            "Analyzing steel surface..."
        ):

            target_layer = (
                model.layer4[-1].conv2
            )

            (
                predicted_index,
                confidence,
                gradcam
            ) = generate_gradcam(
                model,
                image,
                target_layer
            )

        predicted_class = (
            CLASS_NAMES[
                predicted_index
            ]
        )

        display_name = (
            predicted_class
            .replace("_", " ")
            .title()
        )


        # ------------------------------------------
        # RESULT
        # ------------------------------------------

        st.divider()

        st.subheader(
            "AI Inspection Result"
        )

        result_col1, result_col2 = (
            st.columns(2)
        )

        with result_col1:

            st.metric(
                "Detected Defect",
                display_name
            )

        with result_col2:

            st.metric(
                "Model Confidence",
                f"{confidence * 100:.2f}%"
            )

        st.info(
            DESCRIPTIONS[
                predicted_class
            ]
        )


        # ------------------------------------------
        # VISUAL EXPLANATION
        # ------------------------------------------

        st.subheader(
            "Visual Explanation"
        )

        st.write(
            "Grad-CAM highlights the regions that had "
            "the strongest influence on the model's prediction."
        )

        resized_image = image.resize(
            (224, 224)
        )

        image_array = np.array(
            resized_image
        )

        fig, ax = plt.subplots(
            figsize=(5, 5)
        )

        ax.imshow(
            image_array
        )

        ax.imshow(
            gradcam,
            cmap="jet",
            alpha=0.45
        )

        ax.axis("off")

        ax.set_title(
            f"{display_name} — "
            f"{confidence * 100:.2f}%"
        )

        visual_col1, visual_col2 = (
            st.columns(2)
        )

        with visual_col1:

            st.markdown(
                "**Original Image**"
            )

            st.image(
                image,
                use_container_width=True
            )

        with visual_col2:

            st.markdown(
                "**Grad-CAM Explanation**"
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

        plt.close(fig)

        st.caption(
            "Warm-colored regions indicate areas that "
            "contributed more strongly to the model's prediction."
        )


        # ------------------------------------------
        # DISCLAIMER
        # ------------------------------------------

        st.warning(
            "This application is intended for educational "
            "and research purposes. AI predictions should "
            "not replace professional industrial inspection."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "VisionQC uses a fine-tuned ResNet18 deep-learning model "
    "with Grad-CAM explainability for steel surface defect classification."
)