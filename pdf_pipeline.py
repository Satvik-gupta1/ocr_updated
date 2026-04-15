import json
import cv2
import numpy as np
from pdf2image import convert_from_path
import easyocr

# ---------- CONFIG ----------
PDF_PATH = "input.pdf"
OUTPUT_JSON = "pdf_output_clean.json"

# ---------- INIT OCR ----------
reader = easyocr.Reader(['hi'])

# ---------- PDF → IMAGES ----------
pages = convert_from_path(PDF_PATH)

final_output = []

# ---------- PROCESS EACH PAGE ----------
for page_index, pil_image in enumerate(pages):

    print(f"Processing Page {page_index+1}")

    # convert PIL → OpenCV
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape

    # ---------- OCR ----------
    ocr_results = reader.readtext(image)

    page_data = {
        "page": f"page {page_index+1}",
        "annotations": []
    }

    for box, text, conf in ocr_results:

        # ---------- LINE BBOX ----------
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]

        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))

        line_entry = {
            "label": "TextLine",
            "text": text,
            "polygon": [
                {"x": x1, "y": y1},
                {"x": x2, "y": y1},
                {"x": x2, "y": y2},
                {"x": x1, "y": y2}
            ],
            "words": []
        }

        # ---------- WORD SPLIT ----------
        words = text.split()

        word_width = (x2 - x1) / len(words) if words else 1

        for i, word_text in enumerate(words):

            wx = int(x1 + i * word_width)
            wy = y1
            ww = int(word_width)
            wh = y2 - y1

            word_entry = {
                "label": "Word",
                "text": word_text,
                "bbox": [wx, wy, ww, wh],
                "characters": []
            }

            # ---------- CHAR SPLIT ----------
            chars = list(word_text)

            if chars:
                char_width = ww / len(chars)

                for j, ch in enumerate(chars):
                    cx = int(wx + j * char_width)

                    word_entry["characters"].append({
                        "char": ch,
                        "bbox": [
                            cx,
                            wy,
                            int(char_width),
                            wh
                        ]
                    })

            line_entry["words"].append(word_entry)

        page_data["annotations"].append(line_entry)

    final_output.append(page_data)

# ---------- SAVE ----------
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(final_output, f, ensure_ascii=False, indent=2)

print("🔥 CLEAN PDF OCR DONE →", OUTPUT_JSON)