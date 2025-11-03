import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# === CONFIGURATION (Read from Docker ENV) ===
DATA_DIR = os.getenv("DATA_DIR","infeer_Data")  # default fallback
MODEL_DIR = os.getenv("OUTPUT_DIR","/opt/dlami/nvme/Folder_Final_results/model")
MAX_LEN = int(os.getenv("MAX_LEN", 256))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Configuration:")
print(f"   DATA_DIR  = {DATA_DIR}")
print(f"   MODEL_DIR = {MODEL_DIR}")
print(f"   MAX_LEN   = {MAX_LEN}")
print(f"   BATCH_SIZE= {BATCH_SIZE}")
print(f"   DEVICE    = {DEVICE}")

# === LOAD CLASS NAMES ===
class_file = os.path.join(MODEL_DIR, "class_names.txt")
if not os.path.exists(class_file):
    raise FileNotFoundError(f"class_names.txt not found at: {class_file}")

with open(class_file, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]

le = LabelEncoder()
le.fit(class_names)

# === LOAD MODEL & TOKENIZER ===
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

# === HELPER: READ ALL FOLDERS ===
def read_all_folders(root_dir):
    folder_texts = []
    folder_names = []

    if not os.path.exists(root_dir):
        print(f" DATA_DIR path does not exist: {root_dir}")
        return pd.DataFrame()

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
        if not txt_files:
            print(f"No .txt file found in {folder_name}, skipping...")
            continue

        txt_path = os.path.join(folder_path, txt_files[0])
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    folder_texts.append(text)
                    folder_names.append(folder_name)
        except Exception as e:
            print(f"Could not read {txt_path}: {e}")

    return pd.DataFrame({"folder_name": folder_names, "text": folder_texts})

# === PREDICT FUNCTION ===
def predict_batchwise(df: pd.DataFrame):
    inputs = df["text"].tolist()
    encodings = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    input_ids = encodings["input_ids"].to(DEVICE)
    attention_mask = encodings["attention_mask"].to(DEVICE)
    all_preds = []

    for i in range(0, len(input_ids), BATCH_SIZE):
        batch_ids = input_ids[i:i + BATCH_SIZE]
        batch_mask = attention_mask[i:i + BATCH_SIZE]
        with torch.no_grad():
            outputs = model(input_ids=batch_ids, attention_mask=batch_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)

    df["predicted_class"] = le.inverse_transform(all_preds)
    return df

# === MAIN EXECUTION ===
print(f"\n🔹 Scanning folders in {DATA_DIR}...")
df = read_all_folders(DATA_DIR)

if df.empty:
    print("No valid folders found. Exiting...")
else:
    print(f"Found {len(df)} folders. Running batch prediction (batch size={BATCH_SIZE})...")
    df_pred = predict_batchwise(df)

    # Save results
    os.makedirs(MODEL_DIR, exist_ok=True)
    out_json = os.path.join(MODEL_DIR, "folder_predictions.json")

    result_dict = dict(zip(df_pred["folder_name"], df_pred["predicted_class"]))
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)

    print(f" Saved predictions to {out_json}")
