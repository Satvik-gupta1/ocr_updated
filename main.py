from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import cv2
import matplotlib.pyplot as plt

# 1. Load model
model = ocr_predictor(det_arch='db_resnet50',
                      reco_arch='crnn_vgg16_bn',
                      pretrained=True)

# 2. Load image
doc = DocumentFile.from_images("image.png")

# 3. Run OCR
result = model(doc)

# 4. Export JSON
json_output = result.export()
print(json_output)   # optional

# 5. Load image using OpenCV
img = cv2.imread("image.png")
h, w, _ = img.shape

# 6. Draw bounding boxes
for page in json_output['pages']:
    for block in page['blocks']:
        for line in block['lines']:
            
            # LINE LEVEL BOX (BLUE)
            (x1, y1), (x2, y2) = line['geometry']
            pt1 = (int(x1 * w), int(y1 * h))
            pt2 = (int(x2 * w), int(y2 * h))
            cv2.rectangle(img, pt1, pt2, (255, 0, 0), 2)

            for word in line['words']:
                
                # WORD LEVEL BOX (GREEN)
                (x1, y1), (x2, y2) = word['geometry']
                pt1 = (int(x1 * w), int(y1 * h))
                pt2 = (int(x2 * w), int(y2 * h))
                cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)

# 7. Show output
plt.figure(figsize=(10, 14))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# 8. Save output (optional)
cv2.imwrite("output.png", img)

# ADD THIS AT THE END OF main.py

import json

with open("doctr_output.json", "w", encoding="utf-8") as f:
    json.dump(json_output, f, ensure_ascii=False, indent=2)

print("✅ DocTR JSON saved as doctr_output.json")