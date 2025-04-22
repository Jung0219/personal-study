from pathlib import Path

# Define the markdown content again due to code state reset
exdark_readme_content = """# 🌙 Understanding the ExDark Dataset

**Exclusively Dark (ExDark)** is a dataset specifically curated for object detection and classification in low-light environments. It provides both image-level and object-level annotations, helping researchers address challenges posed by illumination issues in computer vision.

---

## 🔗 Resources
- **GitHub**: [ExDark GitHub Repo](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)
- **Paper**: [ExDark Arxiv PDF](http://cs-chan.com/doc/cviu.pdf)

---

## 📊 Dataset Overview
- **Total Images**: 7,363 low-light images
- **Sources**: Internet image search, and sub-sampling from ImageNet, COCO, and PASCAL VOC
- **Annotations**:
  - Image-level class labels
  - Object-level bounding boxes (majority labeled as "people")
- **Annotation Tool**: Piotr’s Computer Vision Matlab Toolbox

---

## 💡 Lighting Conditions (10 Types)
Each image is labeled with one of the following lighting types:

| Type     | Description |
|----------|-------------|
| **Low**      | Extremely dark, minimal illumination, barely visible details. |
| **Ambient**  | Weak illumination, no visible light source. |
| **Object**   | Bright object in a dark scene, no visible light source. |
| **Single**   | Single visible light source (e.g., streetlight). |
| **Weak**     | Multiple but faint light sources. |
| **Strong**   | Multiple bright and visible light sources. |
| **Screen**   | Bright screen (e.g., TV or monitor) in indoor scenes. |
| **Window**   | Bright daylight from windows in indoor scenes. |
| **Shadow**   | Daylight scenes where subjects are in shadow. |
| **Twilight** | Taken during dawn or dusk with soft natural light. |

---

## 🧾 Image Annotation Format

Each image has a corresponding `.txt` file with annotation entries. Format:
% bbGt version=3 <Object Class> <left> <top> <width> <height> <...unused fields...>

- First line is a version header.
- First 16 characters: Annotation tool data (unused).
- Object class and bounding box: left (x), top (y), width, height.
- Occlusion/orientation info (columns 6–12) are not used.

🧪 **Example**:
Bicycle 515 349 268 179 0 0 0 0 0 0 0


---

## 📋 imageclasslist.txt Format

Each line includes:
1. **Image Path** (e.g., `Bicycle/2015_00004.jpg`)
2. **Object Class**:  
   `Bicycle(1), Boat(2), Bottle(3), Bus(4), Car(5), Cat(6), Chair(7), Cup(8), Dog(9), Motorbike(10), People(11), Table(12)`
3. **Lighting Type**:  
   `Low(1), Ambient(2), Object(3), Single(4), Weak(5), Strong(6), Screen(7), Window(8), Shadow(9), Twilight(10)`
4. **Scene Type**:  
   `Indoor(1), Outdoor(2)`
5. **Experiment Set**:  
   `Training(1), Validation(2), Testing(3)`

---

## ⚠️ Notes
- The dataset is too small (~7k images) to train a full deep model from scratch.
- Recommended approach: **fine-tune using transfer learning** with pre-trained models.

---
"""

exdark_readme_path = Path("/mnt/data/ExDark_README.md")
exdark_readme_path.write_text(exdark_readme_content)

exdark_readme_path.name


