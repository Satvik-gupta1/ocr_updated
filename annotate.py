import json
import os
import cv2

# ---------- CONFIG ----------
DATASET_DIR  = "Dataset"
DOCTR_DIR    = "Doctr_json"
OUTPUT_DIR   = "Annotated_Data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BOX_COLOR  = (0, 255, 0)   # green for line boxes
TEXT_COLOR = (0, 0, 255)   # red for line id label
THICKNESS  = 2
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5

# ---------- PROCESS ----------
json_files = sorted([f for f in os.listdir(DOCTR_DIR) if f.endswith('.json')])

for jf in json_files:
    stem     = os.path.splitext(jf)[0]
    img_path = os.path.join(DATASET_DIR, stem + ".jpg")
    json_path = os.path.join(DOCTR_DIR, jf)

    if not os.path.exists(img_path):
        print(f"[MISSING IMAGE] {img_path} — skipping")
        continue

    image = cv2.imread(img_path)

    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    for line in data['lines']:
        b  = line['bbox_local']
        x1 = b['x']
        y1 = b['y']
        x2 = x1 + b['w']
        y2 = y1 + b['h']

        cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, THICKNESS)
        cv2.putText(image, str(line['id']), (x1, max(y1 - 4, 10)),
                    FONT, FONT_SCALE, TEXT_COLOR, 1, cv2.LINE_AA)

    out_path = os.path.join(OUTPUT_DIR, stem + ".jpg")
    cv2.imwrite(out_path, image)
    print(f"Saved → {out_path}  ({len(data['lines'])} lines)")

print(f"\nDone. Annotated images saved in {OUTPUT_DIR}/")
