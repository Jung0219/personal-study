# COCO Annotation Format

The COCO (Common Objects in Context) dataset format is a widely used structure for object detection, segmentation, and captioning tasks. It organizes annotation data in a single JSON file with the following main sections:

---

## 📁 `info`
- Contains high-level metadata about the dataset.
- Typical fields: `description`, `url`, `version`, `year`, `contributor`, `date_created`.

---

## 🪪 `licenses`
- Information about the dataset's usage license.
- Each entry may include: `id`, `name`, `url`.

---

## 🖼️ `images`
- A list of all images in the dataset.
- Each entry corresponds to one image and typically includes:
  - `id`: Unique identifier for the image.
  - `file_name`: Image file name.
  - `width`, `height`: Dimensions of the image.
  - `license`: Optional link to license entry.
  - `date_captured`: (Optional) Timestamp of capture.

---

## 🔖 `annotations`
- Each entry describes an object instance within an image.
- Fields:
  - `id`: Unique annotation ID.
  - `image_id`: References the image that this annotation belongs to.
  - `category_id`: References the category of the object.
  - `bbox`: Bounding box in `[x, y, width, height]` format (top-left origin).
  - `area`: Area of the bounding box (`width * height`), or segmentation area if provided.
  - `iscrowd`: Indicates group annotations; `0` = single object, `1` = crowd/group (e.g., a group of people).
  - `segmentation`: (Optional) Polygon(s) that outline the object.
  - `keypoints`, `num_keypoints`: (Optional) Used in pose estimation datasets.

---

## 🏷️ `categories`
- Defines all the object classes in the dataset.
- Each entry includes:
  - `id`: Unique identifier for the category.
  - `name`: Human-readable class label.
  - `supercategory`: (Optional) Higher-level grouping, e.g., 'animal' for 'dog', 'cat', etc.

---

## 🧾 Example Structure
```json
{
  "info": [...],
  "licenses": [...],
  "images": [...],
  "annotations": [...],
  "categories": [...]
}
```
