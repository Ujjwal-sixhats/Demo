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
# =========================
# ImageNet normalization
# =========================

from dotenv import load_dotenv
# =========================
# ImageNet normalization
# =========================

load_dotenv() 

# Hello How are yOu. I am fine. 
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =========================
# BiRefNet / RMBG-2.0 helpers
# =========================
def _preprocess_for_birefnet(pil_img: Image.Image, side: int = 1024) -> np.ndarray:
    """
    Return (1,3,H,W) float32 normalized tensor for ONNX.
    """
    img = pil_img.convert("RGB").resize((side, side))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return arr[None, ...]  # NCHW


def _postprocess_alpha(alpha_01: np.ndarray, out_size) -> Image.Image:
    """
    alpha_01: (1,1,H,W) or (H,W); returns PIL 'L' alpha (0..255) resized to out_size.
    """
    if alpha_01.ndim == 4:
        alpha_01 = alpha_01.squeeze(0).squeeze(0)
    alpha = (np.clip(alpha_01, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(alpha).resize(out_size, Image.BILINEAR)


class BiRefNetONNX:
    """
    Lightweight ONNX wrapper. Adjust input/output names if your model differs.
    """
    def __init__(self, model_path: str, providers=None):
        self.session = ort.InferenceSession(
            model_path,
            providers=providers or ["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def infer_alpha(self, pil_img: Image.Image, side: int = 1024) -> Image.Image:
        x = _preprocess_for_birefnet(pil_img, side)  # (1,3,side,side)
        y = self.session.run([self.output_name], {self.input_name: x})[0]  # (1,1,H,W)
        return _postprocess_alpha(y, pil_img.size)


# =========================
# Cutout + mask via BiRefNet
# =========================
def get_car_and_mask_birefnet(image_path: str, call: bool,
                              model_path: str = "models/model.onnx",
                              side: int = 1024):
    """
    Returns:
        car_rgba: np.ndarray (H, W, 4) RGBA with alpha applied
        mask:     np.ndarray (H, W) uint8 {0,255}
    Also writes files to saved2/... depending on `call`.
    """
    with open(image_path, "rb") as f:
        raw = f.read()
    pil_in = Image.open(io.BytesIO(raw)).convert("RGB")

    engine = BiRefNetONNX(model_path)
    alpha_L = engine.infer_alpha(pil_in, side=side)   # PIL 'L' (0..255)

    rgba = pil_in.convert("RGBA")
    rgba.putalpha(alpha_L)
    car_rgba = np.array(rgba)  # (H,W,4) RGBA uint8

    mask = np.array(alpha_L, dtype=np.uint8)
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    # Save as BGRA for OpenCV-consistency in your existing code
    car_bgra = cv2.cvtColor(car_rgba, cv2.COLOR_RGBA2BGRA)
    Path("saved2").mkdir(parents=True, exist_ok=True)
    if call:
        cv2.imwrite("saved2/segmented_car.png", car_bgra)
        cv2.imwrite("saved2/segmented_mask.png", mask)
    else:
        cv2.imwrite("saved2/segmented_scaled_car.png", car_bgra)
        cv2.imwrite("saved2/segmented_scaled_mask.png", mask)

    return car_rgba, mask


# =========================
# Utility: ensure RGBA (NumPy)
# =========================
def ensure_rgba(img_bgr_or_bgra: np.ndarray,
                fallback_alpha: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Ensure an image is RGBA (H,W,4) uint8 from BGR/BGRA input. Uses fallback_alpha if needed.
    """
    if img_bgr_or_bgra is None:
        raise ValueError("Input image is None (failed to read).")
    if img_bgr_or_bgra.ndim != 3:
        raise ValueError("Input image must have 3 channels (BGR/BGRA).")

    h, w, c = img_bgr_or_bgra.shape
    if c == 4:
        bgra = img_bgr_or_bgra
    elif c == 3:
        if fallback_alpha is None:
            alpha = np.full((h, w), 255, dtype=np.uint8)
        else:
            if fallback_alpha.shape[:2] != (h, w):
                raise ValueError("fallback_alpha shape does not match image.")
            alpha = fallback_alpha.astype(np.uint8)
        bgra = np.dstack([img_bgr_or_bgra, alpha])
    else:
        raise ValueError(f"Unsupported channel count: {c}")

    rgba = bgra[:, :, [2, 1, 0, 3]]  # BGRA -> RGBA
    return rgba


# =========================
# Compute scale factor from mask geometry
# =========================
def calculate_scale_factor(segmentation_mask: np.ndarray,
                           bounding_box_xywh: Tuple[int, int, int, int]) -> float:
    """
    Fit the mask's tight extents inside (bw, bh). Returns a scalar scale factor.
    """
    _, _, bw, bh = bounding_box_xywh

    coords = np.column_stack(np.where(segmentation_mask > 0))
    if coords.shape[0] == 0:
        print("No foreground in mask; using scale=1.0")
        return 1.0

    shape_center = coords.mean(axis=0)  # [cy, cx]
    max_dist_y = np.max(np.abs(coords[:, 0] - shape_center[0]))
    max_dist_x = np.max(np.abs(coords[:, 1] - shape_center[1]))

    if max_dist_y < 1e-6 or max_dist_x < 1e-6:
        return 1.0

    scale_x = (bw / 2.0) / max_dist_x
    scale_y = (bh / 2.0) / max_dist_y
    return float(min(scale_x, scale_y))


# =========================
# Scale + save car (RGBA path)
# =========================
def image_scale_and_save_rgba(layer_rgba: np.ndarray,
                              scale_factor: float,
                              out_path: Path = Path("saved2/scaled_image.png")) -> Path:
    """
    Multiply dimensions by `scale_factor` and save PNG with alpha.
    """
    if scale_factor <= 0:
        raise ValueError("scale_factor must be > 0")

    if layer_rgba.dtype != np.uint8:
        layer_rgba = layer_rgba.astype(np.uint8)

    layer_pil = Image.fromarray(layer_rgba, mode="RGBA")

    # ✅ Multiply (not add) for scaling
    new_w = int(round(layer_pil.width  * scale_factor))
    new_h = int(round(layer_pil.height * scale_factor))
    layer_pil = layer_pil.resize((new_w, new_h), resample=Image.LANCZOS)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    layer_pil.save(out_path)
    return out_path


def save_scaled_car(car_img_path: str,
                    car_mask_path: str,
                    bounding_box_xywh: Tuple[int, int, int, int],
                    out_path: str = "saved2/scaled_image.png") -> str:
    """
    Load segmented car + mask, compute scale to fit (bw, bh), scale, and save PNG with alpha.
    """
    car_bgr_or_bgra = cv2.imread(car_img_path, cv2.IMREAD_UNCHANGED)
    if car_bgr_or_bgra is None:
        raise FileNotFoundError(f"Could not read car image at {car_img_path}")

    car_mask = cv2.imread(car_mask_path, cv2.IMREAD_GRAYSCALE)
    if car_mask is None:
        raise FileNotFoundError(f"Could not read mask at {car_mask_path}")

    car_rgba = ensure_rgba(car_bgr_or_bgra, fallback_alpha=car_mask)

    _, _, bw, bh = bounding_box_xywh
    scale_factor = calculate_scale_factor(car_mask, (0, 0, bw, bh))

    out_file = image_scale_and_save_rgba(
        layer_rgba=car_rgba,
        scale_factor=scale_factor,
        out_path=Path(out_path),
    )
    return str(out_file)


# =========================
# PIL utilities for compositing
# =========================
def get_cut_out(image_path: str) -> Image.Image:
    """
    Runs BiRefNet cutout → returns a PIL RGBA image (no BytesIO misuse).
    """
    car_rgba_np, _ = get_car_and_mask_birefnet(image_path, call=False)
    if car_rgba_np.dtype != np.uint8:
        car_rgba_np = car_rgba_np.astype(np.uint8)
    return Image.fromarray(car_rgba_np, mode="RGBA")


def tight_crop_rgba_pil(img_rgba: Image.Image) -> Image.Image:
    """
    Crop an RGBA image to the tight bounding box of alpha>0 pixels.
    """
    alpha = np.array(img_rgba.split()[-1])
    ys, xs = np.where(alpha > 0)
    if ys.size == 0:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    y0, y1 = int(ys.min()), int(ys.max() + 1)
    x0, x1 = int(xs.min()), int(xs.max() + 1)
    return img_rgba.crop((x0, y0, x1, y1))


def center_paste_rgba(canvas: Image.Image,
                      patch: Image.Image,
                      position: Optional[Tuple[int, int]] = None) -> None:
    """
    Paste `patch` (RGBA) onto `canvas` (RGBA). If position None, center it.
    Uses the patch's own alpha as mask.
    """
    cw, ch = canvas.size
    pw, ph = patch.size
    if position is None:
        x = (cw - pw) // 2
        y = (ch - ph) // 2
    else:
        x, y = position
    canvas.paste(patch, (x, 3200), patch)


def load_background_pil(width=10000,
                        height=7000,
                        image_path: Optional[str] = 'Data/background1.png') -> Image.Image:
    """
    Returns a PIL RGBA canvas of the requested size.
    If image_path exists, it is loaded and resized; otherwise returns a transparent canvas.
    """
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).convert("RGBA").resize((width, height), Image.LANCZOS)
    else:
        # Transparent canvas; change to a solid color if you prefer, e.g., (255,255,255,255)
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    return img


def place_car_on_large_bg(src_path: str,
                          out_path: str,
                          auto_fit: bool = False) -> str:
    """
    1) Extract car with BiRefNet (RGBA)
    2) Tight-crop by alpha
    3) (optional) Fit within canvas (not used here)
    4) Center on a 12000x8000 RGBA background
    5) Save PNG
    """
    # 1) cutout as PIL
    cutout_rgba = get_cut_out(src_path)

    # 2) tight crop
    cutout_rgba = tight_crop_rgba_pil(cutout_rgba)

    # 3) optional fit (add logic if you want margins / max size)
    # if auto_fit:
    #     ...

    # 4) compose on 12000x8000
    canvas = load_background_pil(10000, 7000, image_path='Data/background1.png')
    center_paste_rgba(canvas, cutout_rgba, position=None)

    # 5) save
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_p)
    return str(out_p)

def save_binary_file(file_name, data):
    f = open(file_name, "wb")
    f.write(data)
    f.close()


def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )


    prompt = (
        "Add shadows of the car on the floor of the background. While creating shadows of the car on the floor of the background assume that the light is coming from the top",
        "Keep all the details of the car intact and blend it properly with the background and does not look pasted. Also Make the outer edges of the car smooth and mixed with background",
        "Give me a high qualty image of same dimension as input image",
    )

    #prompt = (
        #"Blend these two images",
    #)

    #image1 = Image.open("Data/background1.png")

    image2 = Image.open("saved2/resized_for_gemini.jpg")

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
            image.save("generated_image2.png")

    '''
    files = [
        # Make the file available in local system working directory
        client.files.upload(file="saved/result.png"),
    ]
    model = "gemini-2.0-flash-exp-image-generation"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=files[0].uri,
                    mime_type=files[0].mime_type,
                ),
                types.Part.from_text(text="""keep everything in this image same, only add shadows and reflection of the car on the floor of the background and do not change the lightening of the floor. Also keep the color combination of the background and it's floor same"""),
            ],
        )
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_modalities=[
            "image",
            "text",
        ],
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_CIVIC_INTEGRITY",
                threshold="OFF",  # Off
            ),
        ],
        response_mime_type="text/plain",
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
            continue
        if chunk.candidates[0].content.parts[0].inline_data:
            file_name = "output.jpg"
            save_binary_file(
                file_name, chunk.candidates[0].content.parts[0].inline_data.data
            )
            print(
                "File of mime type"
                f" {chunk.candidates[0].content.parts[0].inline_data.mime_type} saved"
                f"to: {file_name}"
            )
        else:
            print(chunk.text)

    '''
def flatten_to_rgb(img: Image.Image, bg=(255, 255, 255)) -> Image.Image:
    """If image has alpha, composite over a solid background to get RGB."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        # Ensure RGBA for compositing
        base = Image.new("RGBA", img.size, (*bg, 255))
        base.paste(img.convert("RGBA"), mask=img.getchannel("A"))
        return base.convert("RGB")
    return img.convert("RGB") if img.mode != "RGB" else img
# =========================
# Main
# =========================
if __name__ == "__main__":
    # Bounding box into which the car should fit (for computing scale)
    
    bounding_box = (150, 3200, 6600, 3000)  # (x, y, bw, bh) -- only bw,bh matter for scaling

    # 1) Segment the car from an input photo and save car + mask (BGRA + PNG mask)
    #    Set `call=True` to write to saved2/segmented_car.png + saved2/segmented_mask.png


    if len(sys.argv) > 1 and sys.argv[1]:
        IMG_PATH = Path(sys.argv[1]).expanduser()

    if not IMG_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMG_PATH.resolve()}")
    
    car_img_rgba_np, car_mask_np = get_car_and_mask_birefnet(str(IMG_PATH), call=True)
    car_img_path = "saved2/segmented_car.png"
    car_mask_path = "saved2/segmented_mask.png"

    # 2) Scale the car to fit inside (bw, bh) and save
    scaled_out_path = "saved2/scaled_image.png"
    out_scaled = save_scaled_car(
        car_img_path=car_img_path,
        car_mask_path=car_mask_path,
        bounding_box_xywh=bounding_box,
        out_path=scaled_out_path,
    )

    # 3) Place BOTH versions on a 12000x8000 background (saved separately)
    #    a) Original segmented car
    '''
    out1 = place_car_on_large_bg(
        src_path=car_img_path,  # original segmented
        out_path="saved2/placed_car_before_12000x8000.png",
        auto_fit=False,
    )
    '''
    #    b) Scaled version
    out2 = place_car_on_large_bg(
        src_path=out_scaled,    # scaled PNG
        out_path="saved2/placed_car_after_12000x8000.png",
        auto_fit=False,
    )

    img = Image.open("saved2/placed_car_after_12000x8000.png")

    max_width = 4096
    w_percent = max_width / float(img.size[0])
    h_size = int((float(img.size[1]) * float(w_percent)))

    img_resized = img.resize((max_width, h_size), Image.LANCZOS)
    img_resized_final = flatten_to_rgb(img_resized)

    # Save as JPEG or WebP under 7MB
    img_resized_final.save("saved2/resized_for_gemini.jpg", "JPEG", quality=85, optimize=True)

    
    generate() 

    img = cv2.imread("generated_image2.png") 
    img_rgb_tuned = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    # Show
    plt.imshow(img_rgb_tuned)
    plt.axis("off")
    plt.show()
    

    #print("Saved (original on bg):", out1)
    print("Saved (scaled  on bg):", out2)
    print("Scaled car saved at   :", out_scaled)
