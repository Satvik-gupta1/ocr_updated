import json
import cv2
import easyocr

# ---------- CONFIG ----------
IMAGE_PATH = "image.png"
DOCTR_JSON_PATH = "doctr_output.json"
OUTPUT_PATH = "final_output.json"

# ---------- LOAD ----------
image = cv2.imread(IMAGE_PATH)
h, w, _ = image.shape

with open(DOCTR_JSON_PATH, "r", encoding="utf-8") as f:
    doctr_data = json.load(f)

reader = easyocr.Reader(['hi'])

# ---------- GLOBAL OCR ----------
ocr_results = reader.readtext(image)

# ---------- HELPERS ----------
def get_centroid(box):
    cx = sum(p[0] for p in box) / 4
    cy = sum(p[1] for p in box) / 4
    return cx, cy

def is_inside(cx, cy, x1, y1, x2, y2):
    return (cx >= x1 and cx <= x2 and
            cy >= y1 and cy <= y2)

# ---------- MAIN ----------
final_output = {"annotations": []}

for page in doctr_data["pages"]:
    for block in page["blocks"]:
        for line in block["lines"]:

            # ---- Convert normalized → pixel ----
            (nx1, ny1), (nx2, ny2) = line["geometry"]

            x1 = int(nx1 * w)
            y1 = int(ny1 * h)
            x2 = int(nx2 * w)
            y2 = int(ny2 * h)

            collected = []

            # ---- Centroid filtering ----
            for (box, text, conf) in ocr_results:
                cx, cy = get_centroid(box)

                if is_inside(cx, cy, x1, y1, x2, y2):
                    collected.append((box, text))

            # ---- Sort left → right ----
            collected.sort(key=lambda item: sum(p[0] for p in item[0]) / 4)

            # ---- Line text ----
            line_text = " ".join([t for (_, t) in collected])

            line_entry = {
                "label": "TextLine",
                "text": line_text,
                "polygon": [
                    {"x": x1, "y": y1},
                    {"x": x2, "y": y1},
                    {"x": x2, "y": y2},
                    {"x": x1, "y": y2}
                ],
                "words": []
            }

            # ==========================================================
            # 🔥 UPDATED WORD LOGIC (THIS IS THE FIX)
            # ==========================================================
            for (box, text) in collected:

                xs = [p[0] for p in box]
                ys = [p[1] for p in box]

                wx = int(min(xs))
                wy = int(min(ys))
                ww = int(max(xs) - wx)
                wh = int(max(ys) - wy)

                # 🔥 Split text into words
                split_words = text.split()

                if len(split_words) == 0:
                    continue

                word_width = ww / len(split_words)

                for i, word_text in enumerate(split_words):

                    sub_wx = int(wx + i * word_width)
                    sub_ww = int(word_width)

                    word_entry = {
                        "label": "Word",
                        "text": word_text,
                        "bbox": [sub_wx, wy, sub_ww, wh],
                        "characters": []
                    }

                    # ---- Character split ----
                    chars = list(word_text)

                    if len(chars) > 0:
                        char_width = sub_ww / len(chars)

                        for j, ch in enumerate(chars):
                            cx_char = int(sub_wx + j * char_width)

                            word_entry["characters"].append({
                                "char": ch,
                                "bbox": [
                                    cx_char,
                                    wy,
                                    int(char_width),
                                    wh
                                ]
                            })

                    line_entry["words"].append(word_entry)

            final_output["annotations"].append(line_entry)

# ---------- SAVE ----------
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_output, f, ensure_ascii=False, indent=2)

print("🔥 FINAL CLEAN JSON GENERATED → final_output.json")