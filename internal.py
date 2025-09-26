import os
import io
from pathlib import Path
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image
from google import genai
from google.genai import types
from io import BytesIO
import base64
import sys
from google import genai
from dotenv import load_dotenv
# =========================
# ImageNet normalization
# =========================

load_dotenv()


def save_binary_file(file_name, data):
    f = open(file_name, "wb")
    f.write(data)
    f.close()


def generate(image_path:str):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )


    prompt = (
        "Select only window glass: windshield, front-left/right windows, rear-left/right windows, rear glass, quarter windows, and sunroof (if present).",
        "Make the windows of the car to be opaque light grey But keep the rest internal details of the image same.",
        "Hide exterior content completely. If any outside details remain visible, increase tint density until they are not recognizable.",
        "Preserve natural glass reflections and specular highlights; do not introduce banding or fogging.",
        "Do not modify: seats, console, dashboard, steering wheel, pillars (A/B/C), headliner, mirrors, screens, badges, or trim. Rubber window seals must remain unchanged.",
        "Keep overall interior exposure and color unchanged.",
    )

    #prompt = (
        #"Blend these two images",
    #)

    #image1 = Image.open("Data/background1.png")

    image2 = Image.open(image_path)

    #image2 = Image.open("ComfyUI_temp_zyfck_00018_.png")

    response = client.models.generate_content(
        model="gemini-2.5-flash-image-preview",
        contents=[prompt, image2]
    )

    for part in response.candidates[0].content.parts:
        if part.text is not None:
            print(part.text)
        elif part.inline_data is not None:
            image = Image.open(BytesIO(part.inline_data.data))
            image.save("tainted_image.png")


if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1]:
        IMG_PATH = Path(sys.argv[1]).expanduser()

    if not IMG_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMG_PATH.resolve()}")
    

    generate(str(IMG_PATH))


    img = cv2.imread("tainted_image.png") 
    img_rgb_tuned = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    # Show
    plt.imshow(img_rgb_tuned)
    plt.axis("off")
    plt.show()