# API Project DeepLearning

Kelompok 1 of Pengantar DeepLearning's Final Project API. A FastAPI-based service for detecting receipt edges, cropping the receipt, performing object detection, and extracting text via OCR.

## Table of Contents

* [Features](#features)
* [Requirements](#requirements)
* [Installation](#installation)
* [Configuration](#configuration)
* [Running the Server](#running-the-server)
* [API Endpoints](#api-endpoints)

  * [1. Detect Receipt Edges](#1-detect-receipt-edges)
  * [2. Crop, Detect Objects, and OCR](#2-crop-detect-objects-and-ocr)
* [Examples](#examples)
* [Project Structure](#project-structure)

## Features

* Automatically detect the four corners of a receipt in an uploaded image.
* Crop the receipt based on user-adjusted corner points.
* Perform object detection on cropped receipts (e.g., detect items or regions).
* Run OCR on each detected object and return structured text results.

## Requirements

### Without Docker

* Python 3.8 or higher
* pip

### With Docker

* Docker Engine (version 20.10+)
* (Optional) Docker Compose for multi-container setups

## Installation

### Without Docker

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/deeplearning-receipt-ocr.git
   cd deeplearning-receipt-ocr
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

### With Docker

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/deeplearning-receipt-ocr.git
   cd deeplearning-receipt-ocr
   ```

2. **Build the Docker image**

   ```bash
   docker build -t receipt-ocr-api .
   ```

## Configuration

### Without Docker

* Ensure you have the required Python packages installed:

  * fastapi
  * uvicorn
  * numpy
  * opencv-python
  * pillow
  * pydantic
  * (any additional modules for your detection and OCR models)
* Place your trained model files under `modules/detect/` or update paths in the code.

### With Docker

* The `Dockerfile` copies application code and model assets into the image.
* To keep models external, mount a host directory at runtime:

  ```bash
  docker run -d -v /path/to/models:/app/modules/detect/models --name receipt-ocr -p 80:80 receipt-ocr-api
  ```
* Pass environment variables (e.g., model path or config flags):

  ```bash
  docker run -d -e MODEL_PATH=/app/modules/detect/models --name receipt-ocr -p 80:80 receipt-ocr-api
  ```

## Running the Server

### Without Docker

```bash
uvicorn main:app --host 0.0.0.0 --port 80
```

Visit interactive docs at `http://localhost:80/docs`.

### With Docker

```bash
docker run -d --name receipt-ocr -p 80:80 receipt-ocr-api
```

Your API will be available at `http://localhost/docs` for the OpenAPI documentation.

## API Endpoints

### 1. Detect Receipt Edges

* **URL:** `/detect-edges`
* **Method:** `POST`
* **Content-Type:** `multipart/form-data`
* **Form Data:**

  * `file`: Image file (jpeg, png) of the receipt.

**Response Model:**

```plaintext
{ "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] }
```

**Description:**
Detects the four corner points of the receipt and returns them for UI adjustments.

### 2. Crop, Detect Objects, and OCR

* **URL:** `/process-receipt`
* **Method:** `POST`
* **Content-Type:** `application/json`
* **Request Body:**

  ```json
  {
    "image_b64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "points": [[100.0,50.0],[800.0,55.0],[810.0,1200.0],[90.0,1210.0]]
  }
  ```

**Response Model:**
A JSON object mapping detected object labels to an array of OCR result entries:

```json
{
  "item_label": [
    { "text": "Example Text", "confidence": 0.98, ... },
    ...
  ],
  ...
}
```

**Description:**

1. Decodes the base64 image.
2. Crops the image using provided points.
3. Runs object detection on the cropped image.
4. Performs OCR on each detected region.
5. Returns structured OCR data.

## Examples

### Detect Edges

```bash
curl -X POST "http://localhost:80/detect-edges" \
  -F "file=@path/to/receipt.jpg"
```

### Process Receipt

```bash
curl -X POST "http://localhost:80/process-receipt" \
  -H "Content-Type: application/json" \
  -d '{
      "image_b64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
      "points": [[100,50],[800,55],[810,1200],[90,1210]]
    }'
```

## Project Structure

```
├── main.py              # FastAPI application entry point
├── modules/
│   ├── crop.py          # Receipt edge detection and cropping utilities
│   └── detect.py        # Object detection and OCR routines
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```
