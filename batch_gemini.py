import json
import os
import cv2
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------- LOAD ENV ----------
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ---------- CONFIG ----------
IMAGE_PATH = "image.png"
INPUT_JSON = "final_output.json"
OUTPUT_JSON = "final_output_gemini_single.json"

# ---------- LOAD ----------
image = cv2.imread(IMAGE_PATH)

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

lines = data["annotations"]

# ---------- PREPARE INPUT ----------
indexed_input = ""
for i, ann in enumerate(lines):
    indexed_input += f"[{i}] {ann['text']}\n"

# encode image
_, buffer = cv2.imencode(".png", image)
img_bytes = buffer.tobytes()

# ---------- PROMPT ----------
prompt = f"""
You are given:
1. An image of Hindi text
2. OCR-extracted lines (may contain errors)

Your task:
Correct ONLY the text of each line using BOTH the image and OCR text.

STRICT RULES:
1. Keep EXACT same number of lines.
2. DO NOT merge or split lines.
3. DO NOT remove or add lines.
4. DO NOT add extra sentences.
5. Only correct OCR mistakes.
6. Use the IMAGE as primary reference.
7. Prefer correct Hindi words and phrases.


INPUT LINES:
{indexed_input}

OUTPUT FORMAT (STRICT):
[0] corrected line
[1] corrected line
...

Return ONLY corrected lines.
"""

# ---------- GEMINI CALL ----------
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        types.Part.from_text(text=prompt),
        types.Part.from_bytes(data=img_bytes, mime_type="image/png")
    ]
)

output_text = response.text.strip()

# ---------- PARSE ----------
def parse_output(text, expected_len):
    lines = text.split("\n")
    corrected = [""] * expected_len

    for line in lines:
        if line.strip().startswith("["):
            try:
                idx = int(line.split("]")[0][1:])
                content = line.split("]", 1)[1].strip()
                if 0 <= idx < expected_len:
                    corrected[idx] = content
            except:
                continue

    return corrected

corrected_lines = parse_output(output_text, len(lines))

# ---------- APPLY ----------
for i, ann in enumerate(lines):
    if corrected_lines[i]:
        ann["text"] = corrected_lines[i]

# ---------- REBUILD WORDS + CHARS ----------
for ann in lines:

    words = ann["text"].split()
    ann["words"] = []

    x1 = ann["polygon"][0]["x"]
    y1 = ann["polygon"][0]["y"]
    x2 = ann["polygon"][2]["x"]
    y2 = ann["polygon"][2]["y"]

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

        chars = list(word_text)
        if len(chars) > 0:
            char_width = ww / len(chars)

            for j, ch in enumerate(chars):
                cx_char = int(wx + j * char_width)

                word_entry["characters"].append({
                    "char": ch,
                    "bbox": [
                        cx_char,
                        wy,
                        int(char_width),
                        wh
                    ]
                })

        ann["words"].append(word_entry)

# ---------- SAVE ----------
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump({"annotations": lines}, f, ensure_ascii=False, indent=2)

print("🔥 SINGLE CALL GEMINI DONE →", OUTPUT_JSON)