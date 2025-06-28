import base64
import io
from typing import List, Dict, Any

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

from modules.crop import detect_receipt_edges, crop_with_points
from modules.detect import detect_objects, ocr_on_objects

app = FastAPI(
    title="API Project DeepLearning",
    description="detect receipt edges, crop, and perform OCR",
    version="6.3.5",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Points(BaseModel):
    """Represents the four corner points of the receipt."""
    points: List[List[int]] = Field(
        ...,
        example=[[100, 50], [800, 55], [810, 1200], [90, 1210]],
        description="A list of four [x, y] coordinates for the corners: TL, TR, BR, BL."
    )


class ProcessRequest(BaseModel):
    """The request model for the main processing endpoint."""
    image_b64: str = Field(
        ...,
        description="Base64 encoded string of the original image.",
        example="data:image/jpeg;base64,/9j/4AAQSkZJRg..."
    )
    points: List[List[float]] = Field(
        ...,
        example=[[100.0, 50.0], [800.0, 55.0], [810.0, 1200.0], [90.0, 1210.0]],
        description="The final four corner points for cropping."
    )


def read_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Decodes image bytes into a NumPy array."""
    try:
        image_stream = io.BytesIO(image_bytes)
        image = Image.open(image_stream)
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")


@app.get("/", summary="Root", include_in_schema=False)
def read_root():
    return {"message": "Welcome to our API. See /docs for usage."}


@app.post(
    "/detect-edges",
    summary="Detect Receipt Edges",
    response_model=Points,
    tags=["1. Edge Detection"]
)
async def api_detect_edges(file: UploadFile = File(..., description="Image file of the receipt.")):
    """
    Accepts an image file and attempts to automatically detect the four corners
    of the receipt.

    Returns the corner points which can be used in a UI for adjustment.
    """
    image_bytes = await file.read()
    image_bgr = read_image_from_bytes(image_bytes)

    pts, _ = detect_receipt_edges(image_bgr)

    if pts is None:
        raise HTTPException(
            status_code=404,
            detail="Could not detect receipt edges. The image may be unclear or the receipt not prominent."
        )

    return {"points": pts.tolist()}


@app.post(
    "/process-receipt",
    summary="Crop, Detect Objects, and OCR",
    response_model=Dict[str, List[Dict[str, Any]]],
    tags=["2. OCR Processing"]
)
async def api_process_receipt(request: ProcessRequest):
    """
    Receives the original image (as base64) and the final corner points.
    It then performs the following steps:
    1. Crops the receipt based on the provided points.
    2. Runs object detection on the cropped receipt.
    3. Performs OCR on each detected object.
    4. Returns the structured OCR data.
    """
    try:
        header, b64_data = request.image_b64.split(",", 1)
        image_bytes = base64.b64decode(b64_data)
        image_bgr = read_image_from_bytes(image_bytes)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid base64 image string. Ensure it includes the data."
        )

    pts_np = np.array(request.points, dtype="float32")
    if pts_np.shape != (4, 2):
        raise HTTPException(
            status_code=400,
            detail="Points must be an array of 4 [x, y] coordinates."
        )

    try:
        cropped_image = crop_with_points(image_bgr, pts_np)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    if cropped_image.size == 0:
        raise HTTPException(status_code=500, detail="Cropping resulted in an empty image.")

    detections = detect_objects(cropped_image)
    if not detections:
        raise HTTPException(
            status_code=404,
            detail="No objects were detected on the cropped receipt."
        )

    ocr_results = ocr_on_objects(cropped_image, detections)

    return ocr_results

@app.post(
    "/receipt",
    summary="Scan receipt (File Upload + Points)",
    response_model=Dict[str, List[Dict[str, Any]]],
    tags=["3. Scan Receipt"]
)
async def scan_receipt_with_file(
    file: UploadFile = File(..., description="Image file of the receipt"),
    points: List[float] = Form(
        ..., description="Flat list of 8 floats representing 4 corner points: [x1,y1, x2,y2, x3,y3, x4,y4]"
    )
):
    try:
        contents = await file.read()
        image_np = np.frombuffer(contents, np.uint8)
        image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Failed to decode image.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file. {str(e)}")

    try:
        pts_np = np.array(points, dtype="float32").reshape((4, 2))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Points must be a list of 8 float values (4 x [x, y])."
        )

    try:
        cropped_image = crop_with_points(image_bgr, pts_np)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if cropped_image.size == 0:
        raise HTTPException(status_code=500, detail="Cropping resulted in an empty image.")

    detections = detect_objects(cropped_image)
    if not detections:
        raise HTTPException(
            status_code=404,
            detail="No objects were detected on the cropped receipt."
        )

    ocr_results = ocr_on_objects(cropped_image, detections)

    formatted: Dict[str, List[Dict[str, Any]]] = {}
    for cls, items in ocr_results.items():
        formatted[cls] = []
        for entry in items:
            text = entry['text']
            data: Dict[str, Any] = {'bbox': entry['bbox']}
            if cls == 'product_item':
                m = re.match(r"^(.+?)\s+(\d+)\s+(\d+)\s+([\d,]+)$", text)
                if m:
                    name, qty, price, total = m.groups()
                    data.update({
                        'product_name': name.strip(),
                        'quantity': int(qty),
                        'price_per_item': int(price),
                        'total_price': int(total.replace(',', ''))
                    })
                else:
                    data['raw_text'] = text

            elif 'voucher' in cls:
                nums = re.findall(r"\(([\d,]+)\)", text)
                if nums:
                    amount = nums[-1]
                    last_pat = re.escape(f"({amount})")
                    m2 = re.search(last_pat + r"\s*$", text)
                    if m2:
                        name_part = text[:m2.start()].rstrip(" :")
                    else:
                        name_part = text
                    name_upper = name_part.strip().upper()
                    if name_upper in ['TUNAI', 'KEMBALI', 'TOTAL']:
                        data['raw_text'] = text
                    else:
                        data.update({
                            'voucher_name': name_part.strip(),
                            'voucher_price': int(amount.replace(',', ''))
                        })
                else:
                    data['raw_text'] = text

            elif 'discount' in cls or 'disc' in cls.lower():
                nums = re.findall(r"\((-?[\d,]+)\)", text)
                if not nums:
                    nums = re.findall(r"-?[\d,]+", text)
                if nums:
                    val = nums[-1]
                    data['discount'] = abs(int(val.replace(',', '')))
                else:
                    data['raw_text'] = text

            else:
                data['text'] = text

            formatted[cls].append(data)

    return formatted
