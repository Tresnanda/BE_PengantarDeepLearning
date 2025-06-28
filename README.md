# API Project DeepLearning

Kelompok 1 of Pengantar DeepLearning's Final Project API. A FastAPI-based service for detecting receipt edges, cropping the receipt, performing object detection, and extracting text via OCR with advanced post-processing.

## Table of Contents

* [Features](#features)
* [Requirements](#requirements)
* [Installation](#installation)
* [Configuration](#configuration)
* [Running the Server](#running-the-server)
* [API Endpoints](#api-endpoints)

  * [1. Detect Receipt Edges](#1-detect-receipt-edges)
  * [2. Process Receipt (Crop, Detect, OCR, Post-Processing)](#2-process-receipt-crop-detect-ocr-post-processing)
* [Examples](#examples)
* [Project Structure](#project-structure)

## Features

* **Edge Detection**: Automatically detect the four corners of a receipt in an uploaded image.
* **Cropping**: Crop the receipt image based on provided corner points.
* **Object Detection**: Detect regions/items on the receipt using a trained YOLO model.
* **OCR**: Extract text from each detected region with Tesseract.
* **Post-OCR Processing**:

  * **Product Items**: Parse lines like `JAVANA TEH MLATI 350 3 3000 15,000` into structured fields: `product_name`, `quantity`, `price_per_item`, `total_price`.
  * **Vouchers**: Extract the rightmost parenthesized amount and treat the preceding text as `voucher_name`, ignoring entries labeled `TUNAI`, `KEMBALI`, or `TOTAL`.
  * **Discounts**: Capture numbers either inside parentheses or standalone, take the rightmost match, convert to an absolute `discount` value.

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
   docker build -t receipt-processor-fastapi .
   ```

## Configuration

### Without Docker

* Ensure required Python packages are installed:

  * fastapi, uvicorn, numpy, opencv-python, pillow, pydantic, torch, ultralytics, pytesseract
* Place model weights under `modules/detect/best.pt` or update the path in code.

### With Docker

* Models and code are copied into the image by the Dockerfile.
* To override or mount external model directory:

  ```bash
  docker run -d -v /path/to/models:/app/modules/detect --name receipt-ocr -p 80:80 receipt-processor-fastapi
  ```

## Running the Server

### Without Docker

```bash
uvicorn main:app --host 0.0.0.0 --port 80
```

Visit interactive docs at `http://localhost:80/docs`.

### With Docker

```bash
docker run -d --name receipt-ocr -p 80:80 receipt-processor-fastapi
```

OpenAPI docs available at `http://localhost/docs`.

## API Endpoints

### 1. Detect Receipt Edges

* **URL:** `/detect-edges`
* **Method:** `POST`
* **Content-Type:** `multipart/form-data`
* **Form Data**:

  * `file`: Image file (jpeg, png) of the receipt.

**Response:**

```json
{ "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] }
```

Detects four corner points for UI adjustment.

### 2. Process Receipt (Crop, Detect, OCR, Post-Processing)

* **URL:** `/process-receipt`
* **Method:** `POST`
* **Content-Type:** `application/json`
* **Request Body**:

```json
{
  "image_b64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "points": [[100.0,50.0],[800.0,55.0],[810.0,1200.0],[90.0,1210.0]]
}
```

**Response:**
A JSON mapping each detected class to an array of structured entries. For example:

```json
{
  "product_item": [
    {
      "bbox": [x1,y1,x2,y2],
      "product_name": "JAVANA TEH MLATI 350",
      "quantity": 3,
      "price_per_item": 3000,
      "total_price": 15000
    }
  ],
  "product_voucher": [
    {
      "bbox": [x1,y1,x2,y2],
      "voucher_name": "VC THE UNIVENUS",
      "voucher_price": 7500
    }
  ],
  "product_discount": [
    {
      "bbox": [x1,y1,x2,y2],
      "discount": 4600
    }
  ]
}
```

Fields are parsed according to class-specific regex rules:

* **product\_item**: name, quantity, price\_per\_item, total\_price
* **voucher**: voucher\_name (left of last `(...)`), voucher\_price
* **discount**: absolute numeric value from last match

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
│   └── detect.py        # Object detection, OCR, and post-processing logic
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```
