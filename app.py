from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import timm
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import io

from fastapi.middleware.cors import CORSMiddleware
# CORS Middleware configuration
origins = [
    "http://localhost:5173",  # Allow the frontend running on localhost:5173
    "http://127.0.0.1:5173",  # Allow localhost as an additional origin
]
# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize the model
class DeepfakeModel(torch.nn.Module):
    def __init__(self):
        super(DeepfakeModel, self).__init__()
        self.model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=2)
    
    def forward(self, x):
        return self.model(x)

model = DeepfakeModel().to(device)
model.load_state_dict(torch.load("efficientnet_b3_deepfake.pth", map_location=device))
model.eval()

# Image Transformation Pipeline
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# FastAPI App Initialization
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of origins to allow
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)
# Prediction Function
def predict(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = np.array(image)
    image = transform(image=image)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, prediction = torch.max(output, 1)

    return "Fake" if prediction.item() == 1 else "Real"

# API Endpoint for Predictions
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        prediction = predict(image_bytes)
        return JSONResponse(content={"prediction": prediction})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
