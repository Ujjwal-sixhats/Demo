# car_scene_classifier.py
import argparse, json
import torch, open_clip
from PIL import Image
from pathlib import Path
import json, subprocess, sys

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

# Load model + preprocess
model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
model = model.to(DEVICE).eval()
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

CLASS_NAMES = ["interior", "exterior", "not_car"]
CLASS_PROMPTS = [
    "a photo of a car interior, dashboard, steering wheel, seats",
    "a photo of a car exterior, outside view of a car body",
    "not a car photo",
]

# Precompute normalized text features once
with torch.no_grad():
    text_tokens = tokenizer(CLASS_PROMPTS).to(DEVICE)
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    LOGIT_SCALE = model.logit_scale.exp()

def classify(img_path: str, tie_margin: float = 0.01, abstain_prob: float = 0.20):
    """
    tie_margin: interior vs exterior too-close -> 'unsure'
    abstain_prob: if top class prob below this -> 'unsure'
    """
    img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        image_features = model.encode_image(img)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        probs = (LOGIT_SCALE * image_features @ text_features.T).softmax(dim=-1).squeeze(0)

    p_int, p_ext, p_not = map(float, probs.tolist())
    scores = {"interior": p_int, "exterior": p_ext, "not_car": p_not}

    top_idx = int(torch.argmax(probs).item())
    top_label = CLASS_NAMES[top_idx]
    top_prob = float(probs[top_idx])

    if top_prob < abstain_prob:
        return "unsure", scores

    if top_label in ("interior", "exterior") and abs(p_int - p_ext) < tie_margin:
        return "unsure", scores

    return top_label, scores

if __name__ == "__main__":

    # Hard-coded config
    IMG_PATH = Path("Data\car200.jpg")   # use r"" or forward slashes on Windows
    TIE_MARGIN = 0.01
    ABSTAIN_PROB = 0.20

    # What to pass to test9.py when label == "exterior"
    ARG_FOR_TEST9 = str(IMG_PATH)       # change to whatever argument you need

    if not IMG_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMG_PATH.resolve()}")

    label, scores = classify(
        str(IMG_PATH),
        tie_margin=TIE_MARGIN,
        abstain_prob=ABSTAIN_PROB,
    )
    print(label)
    print(json.dumps(scores, indent=2))

    if label == "exterior" or label=="not_car": 
        # Runs: python test9.py <ARG_FOR_TEST9> using the same Python interpreter
        subprocess.run([sys.executable, "test9.py", ARG_FOR_TEST9], check=True)

    else :
        subprocess.run([sys.executable, "test11.py", ARG_FOR_TEST9], check=True)
    #elif label == "interior":
        #subprocess.run([sys.executable, "test11.py", ARG_FOR_TEST9], check=True)

    #else :
        #print("Enter a car image please")

    
