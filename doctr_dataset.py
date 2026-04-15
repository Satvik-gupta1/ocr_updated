import json
import os
import re
import cv2
import numpy as np
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

# ---------- CONFIG ----------
DATASET_DIR = "Dataset"
OUTPUT_DIR = "Doctr_json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- INIT MODEL ----------
model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

# ---------- PARSE FILENAME ----------
def parse_filename(filename):
    """Extract image_id and column info from filename like img_1_c1of2.jpg"""
    match = re.match(r'img_(\d+)_c(\d+)of(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None, None, None

# ---------- PROCESS EACH IMAGE ----------
image_files = sorted([f for f in os.listdir(DATASET_DIR) if f.lower().endswith('.jpg')])

for img_file in image_files:
    img_path = os.path.join(DATASET_DIR, img_file)
    stem = os.path.splitext(img_file)[0]

    image_id, column, total_columns = parse_filename(stem)
    if image_id is None:
        print(f"Skipping {img_file} — couldn't parse filename")
        continue

    print(f"Processing {img_file} ...")

    # Load image
    image = cv2.imread(img_path)
    h, w = image.shape[:2]

    # Run DocTR
    doc = DocumentFile.from_images(img_path)
    result = model(doc)
    json_output = result.export()

    # Collect all lines from DocTR output
    lines_out = []
    line_id = 1

    for page in json_output['pages']:
        for block in page['blocks']:
            for line in block['lines']:
                (nx1, ny1), (nx2, ny2) = line['geometry']

                # Convert normalized → pixel (global = same as local since no crop info)
                x = int(nx1 * w)
                y = int(ny1 * h)
                lw = int((nx2 - nx1) * w)
                lh = int((ny2 - ny1) * h)

                lines_out.append({
                    "id": line_id,
                    "bbox_global": {"x": x, "y": y, "w": lw, "h": lh},
                    "bbox_local": {"x": x, "y": y, "w": lw, "h": lh}
                })
                line_id += 1

    # Build output JSON matching Dataset_json format
    output = {
        "image_id": image_id,
        "source": img_path.replace("\\", "/"),
        "column": column,
        "total_columns": total_columns,
        "crop": {"x": 0, "y": 0, "w": w, "h": h},
        "num_lines": len(lines_out),
        "lines": lines_out
    }

    out_path = os.path.join(OUTPUT_DIR, stem + ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"  -> {len(lines_out)} lines saved to {out_path}")

print("\nDone. All JSONs saved in Doctr_json/")
