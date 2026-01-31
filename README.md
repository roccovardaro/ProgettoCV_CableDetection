# 🔌 Cable Detection and Segmentation Project

**Computer Vision Course - Academic Year 2025/2026**  
*University of Calabria - DIMES Department*

This repository contains the implementation of a **Cable Detection and Segmentation** system. The goal is to identify the pixels corresponding to electrical cables in aerial images and estimate the line equation for each detected object.

## 📂 Repository Structure

- `CableDetection.py`: Custom PyTorch dataset for semantic segmentation based on COCO annotations, with automatic image and mask loading and preprocessing.
- `Fine_Tuning_UNET++.ipynb`: Notebook for training and fine-tuning the model.
- `Create_Json_Predictions.ipynb`: Generates model predictions on the test dataset and saves them in JSON format for evaluation.
- `TestCableDetection.ipynb`: Computes evaluation metrics by comparing predictions with ground truth annotations.
- `testModel.ipynb`: Loads a trained model to perform inference on a single image, visualizing the predicted mask.
- `training_UNET++.ipynb`: Manages the full initial training of the UNET++ segmentation model using the original dataset.

## 🧠 Architecture and Design Choices

To recognize thin structures like cables, a **U-Net++** architecture was used.

- **Model**: U-Net++ introduces dense and nested connections between encoder and decoder, reducing the semantic gap of feature maps and improving edge precision.
- **Backbone**: `timm-resnest50d` (Pretrained). Uses ResNeSt50 for robust feature extraction.
- **Training Hardware**: NVIDIA A100 (80GB VRAM) on Google Colab Pro.

### Training Strategy

- **Loss Function**: Combination of **Focal Loss** (to handle background/foreground imbalance) and **Dice Loss** (to optimize F1-Score and geometric overlap).
- **Optimizer**: AdamW.
- **Data Augmentation**:
  - *Geometric*: Horizontal/vertical flips, discrete rotations (90°, 180°).
  - *Deformable*: **Elastic Transform** to simulate curvature and prevent overfitting on straight lines.

## 📊 Dataset and Performance

The project uses a subset of the **TTPLA** dataset:

- **Train Set**: 842 images (700x700).
- **Test Set**: 400 images (700x700).

### Evaluation Metrics

The evaluation is based on the **Line Detection Score (LDS)**:

$$ LDS = mAP + mAR + 2 \cdot \exp(-0.12 \cdot \Delta\theta) $$

where $\Delta\theta$ is the minimum angular error compared to the ground truth.

## 📄 Matricola.json File Specifications

The file must contain the model predictions on the test dataset, strictly following the COCO format and including the geometric specifications for cable detection.

### General Information

- **File Name:** `{your_student_id}.json` (e.g., `123456.json`)  
- **Data Type:** `List[Dict]` (A list of JSON objects)

### 🔑 Detailed Field Description

| Field | Type | Description |
|-------|------|-------------|
| `image_id` | `int` | Unique ID of the test image. |
| `category_id` | `int` | ID of the predicted class (e.g., `0` or `1` for cable). |
| `id` | `int` | Unique ID of the single instance/prediction. |
| `score` | `float` | Model confidence (0.0–1.0). |
| `area` | `float` | Total pixel area of the mask. |

#### Detection & Segmentation

- **`bbox`**: `[x_min, y_min, width, height]`  
  *Standard COCO format: top-left coordinates followed by width and height.*

- **`segmentation`**: Binary mask encoded in **RLE (Run-Length Encoding)**.  
  - `size`: `[height, width]` of the image  
  - `counts`: Alphanumeric string representing the compressed mask

#### Cable Geometry (`lines`)

The `lines` field contains the line parameters in polar coordinates `[ρ, θ]`:

- **ρ (Rho)**: Minimum perpendicular distance from the origin (0,0) to the line  
- **θ (Theta)**: Angle in radians of the perpendicular with the X-axis (`0 ≤ θ ≤ 2π`)

### 📄 Example Structure

```json
[
  {
    "image_id": 0,
    "category_id": 0,
    "bbox": [539.80, -3.11, 160.13, 566.24],
    "segmentation": {
      "size": [1080, 1920],
      "counts": "TVc;4he04L4L5K7I35J1..."
    },
    "score": 0.96875,
    "lines": [328, 2.2968],
    "area": 394153.45,
    "id": 0
  }
]
```
## 🚀 Usage Instructions

1. **Installation**:
    ```bash
    pip install -r requirements.txt
    ```
2. **Training**:  
Run the notebook `Fine_Tuning_UNET++.ipynb` to train the model.

3. **Generate Output**:  
Use `Create_Json_Predictions.ipynb` to produce the prediction file `matricola.json` in COCO-compliant format.
