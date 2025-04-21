# YOLO Result Format

When using YOLO models (e.g., with the Ultralytics `YOLO` class), the output of a prediction is a `Results` object. Here’s a breakdown of its structure and how to work with the results:

---

## 🎯 Model Prediction Result

### `result.boxes`
- Contains all bounding boxes detected in the image.

#### `xyxy`
- Bounding box coordinates in `[x1, y1, x2, y2]` format.
  - `x1`, `y1`: Top-left corner.
  - `x2`, `y2`: Bottom-right corner.

#### `conf`
- Confidence score for each predicted bounding box (range: 0.0–1.0).

#### `cls`
- Class index predicted for the bounding box.

---

## 🏷️ `result.names`
- A dictionary mapping class indices to human-readable class names.
- Example: `{0: 'person', 1: 'bicycle', 2: 'car', ...}`

---

## 🖼️ `result.orig_img`
- The original image that was passed into the model, stored as a NumPy array.

---

## 🖍️ `result.plot()`
- Visualizes the prediction results by drawing bounding boxes and labels on the image.

---

## 💾 `result.to(...)`
- Export results in various formats:
  - Options: `json`, `pandas`, `txt`, `dict`, etc.

---

## 🧾 Example Usage
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model("image.jpg")

print(results[0].boxes.xyxy)  # Coordinates
print(results[0].boxes.conf)  # Confidence scores
print(results[0].boxes.cls)   # Class indices
results[0].plot()             # Visualize
```
