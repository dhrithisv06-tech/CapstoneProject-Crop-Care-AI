from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ===== YOUR CNN MODEL =====

num_classes = 15

class PlantCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            3, 32,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            32, 64,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(
            64 * 32 * 32,
            128
        )

        self.fc2 = nn.Linear(
            128,
            num_classes
        )

    def forward(self, x):

        x = self.pool(
            torch.relu(
                self.conv1(x)
            )
        )

        x = self.pool(
            torch.relu(
                self.conv2(x)
            )
        )

        x = x.view(
            x.size(0), -1
        )

        x = torch.relu(
            self.fc1(x)
        )

        x = self.fc2(x)

        return x


# ===== LOAD MODEL =====

model = PlantCNN()

model.load_state_dict(
    torch.load(
        "plant_model.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()


# ===== CLASS LABELS =====

class_names = [
    "Pepper Bell Bacterial Spot",
    "Pepper Bell Healthy",
    "Potato Early Blight",
    "Potato Healthy",
    "Potato Late Blight",
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Healthy",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Mosaic Virus",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites",
    "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus"
]


@app.route("/")
def home():
    return render_template(
        "index.html"
    )


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    if "image" not in request.files:
        return jsonify({
            "error":
            "No image uploaded"
        })

    file = request.files["image"]

    filepath = os.path.join(
        app.config[
            "UPLOAD_FOLDER"
        ],
        file.filename
    )

    file.save(filepath)

    image = Image.open(
        filepath
    ).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(
            (128, 128)
        ),
        transforms.ToTensor()
    ])

    image = transform(
        image
    ).unsqueeze(0)

    with torch.no_grad():

        output = model(image)

        probabilities = (
            torch.softmax(
                output,
                dim=1
            )
        )

        confidence = (
            torch.max(
                probabilities
            ).item() * 100
        )

        predicted_class = (
            torch.argmax(
                output,
                dim=1
            ).item()
        )

    disease = class_names[
        predicted_class
    ]

    return jsonify({
        "disease":
        disease,
        "confidence":
        f"{confidence:.2f}%"
    })


if __name__ == "__main__":
    app.run(debug=True)