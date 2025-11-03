# main.py
import os
import torch
import json
from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder

app = FastAPI(title="Context Classifier API")

# === CONFIGURATION ===
MODEL_DIR = os.getenv("MODEL_DIR", "/opt/dlami/nvme/Folder_Final_results/model")
MAX_LEN = int(os.getenv("MAX_LEN", 256))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === LOAD MODEL & TOKENIZER ===
class_file = os.path.join(MODEL_DIR, "class_names.txt")
if not os.path.exists(class_file):
    raise FileNotFoundError(f"class_names.txt not found at: {class_file}")

with open(class_file, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]

le = LabelEncoder()
le.fit(class_names)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

@app.get("/")
def root():
    return {"message": "Context Classifier API is running!"}


class TextInput(BaseModel):
    text: str


@app.post("/predict")
def predict_text(payload: TextInput):
    text = payload.text.strip()
    if not text:
        return {"error": "Empty text provided"}

    inputs = tokenizer(
        [text],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).cpu().numpy()[0]
        predicted_label = le.inverse_transform([pred])[0]

    return {"prediction": predicted_label}
