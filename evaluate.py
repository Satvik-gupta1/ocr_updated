import json
import os
import numpy as np

# ---------- CONFIG ----------
REF_DIR = "Dataset_json"
PRED_DIR = "Doctr_json"
IOU_THRESHOLD = 0.5

# ---------- HELPERS ----------

def xywh_to_xyxy(bbox):
    """Convert {x, y, w, h} dict to [x1, y1, x2, y2]"""
    x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
    return [x, y, x + w, y + h]

def compute_iou(a, b):
    """Compute IoU between two [x1,y1,x2,y2] boxes"""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter = inter_w * inter_h

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0

def evaluate_image(ref_data, pred_data):
    """
    Compare predicted lines vs reference lines for one image.
    Reference uses bbox_global (full image coords).
    Predicted uses bbox_local (cropped image coords) → offset by crop origin.
    Returns TP, FP, FN, list of IoU scores for matched pairs.
    """
    crop = ref_data.get("crop", {"x": 0, "y": 0})
    crop_x, crop_y = crop['x'], crop['y']

    # Reference boxes — already global
    ref_boxes = [xywh_to_xyxy(line['bbox_global']) for line in ref_data['lines']]

    # Predicted boxes — shift by crop offset to align with global coords
    pred_boxes = []
    for line in pred_data['lines']:
        b = line['bbox_local']
        shifted = {
            'x': b['x'] + crop_x,
            'y': b['y'] + crop_y,
            'w': b['w'],
            'h': b['h']
        }
        pred_boxes.append(xywh_to_xyxy(shifted))

    matched_ref  = set()
    matched_pred = set()
    iou_scores   = []

    # Greedy matching: for each ref box find best pred box
    for i, ref_box in enumerate(ref_boxes):
        best_iou  = 0.0
        best_j    = -1

        for j, pred_box in enumerate(pred_boxes):
            if j in matched_pred:
                continue
            iou = compute_iou(ref_box, pred_box)
            if iou > best_iou:
                best_iou = iou
                best_j   = j

        if best_iou >= IOU_THRESHOLD and best_j != -1:
            matched_ref.add(i)
            matched_pred.add(best_j)
            iou_scores.append(best_iou)

    TP = len(matched_ref)
    FN = len(ref_boxes)  - TP
    FP = len(pred_boxes) - TP

    return TP, FP, FN, iou_scores

# ---------- MAIN ----------

total_TP, total_FP, total_FN = 0, 0, 0
all_iou_scores = []
per_image_results = []

ref_files = sorted([f for f in os.listdir(REF_DIR) if f.endswith('.json')])

for ref_file in ref_files:
    pred_file = ref_file  # same filename expected in both folders
    pred_path = os.path.join(PRED_DIR, pred_file)

    if not os.path.exists(pred_path):
        print(f"  [MISSING PRED] {pred_file} — skipping")
        continue

    with open(os.path.join(REF_DIR, ref_file), encoding='utf-8') as f:
        ref_data = json.load(f)
    with open(pred_path, encoding='utf-8') as f:
        pred_data = json.load(f)

    TP, FP, FN, iou_scores = evaluate_image(ref_data, pred_data)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    mean_iou  = float(np.mean(iou_scores)) if iou_scores else 0.0

    per_image_results.append({
        "file":      ref_file,
        "ref_lines": len(ref_data['lines']),
        "pred_lines": len(pred_data['lines']),
        "TP": TP, "FP": FP, "FN": FN,
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "mean_iou":  round(mean_iou,  4)
    })

    total_TP += TP
    total_FP += FP
    total_FN += FN
    all_iou_scores.extend(iou_scores)

# ---------- AGGREGATE ----------
precision_overall = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
recall_overall    = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
f1_overall        = (2 * precision_overall * recall_overall /
                     (precision_overall + recall_overall)
                     if (precision_overall + recall_overall) > 0 else 0.0)
miou_overall      = float(np.mean(all_iou_scores)) if all_iou_scores else 0.0

# ---------- PRINT ----------
print(f"\n{'='*60}")
print(f"  DocTR Line Detection Evaluation  (IoU threshold = {IOU_THRESHOLD})")
print(f"{'='*60}")
print(f"{'File':<30} {'Ref':>4} {'Pred':>4} {'TP':>4} {'FP':>4} {'FN':>4}  {'P':>6}  {'R':>6}  {'F1':>6}  {'mIoU':>6}")
print(f"{'-'*90}")

for r in per_image_results:
    print(f"{r['file']:<30} {r['ref_lines']:>4} {r['pred_lines']:>4} "
          f"{r['TP']:>4} {r['FP']:>4} {r['FN']:>4}  "
          f"{r['precision']:>6.3f}  {r['recall']:>6.3f}  "
          f"{r['f1']:>6.3f}  {r['mean_iou']:>6.3f}")

print(f"{'='*90}")
print(f"{'OVERALL':<30} {'':>4} {'':>4} "
      f"{total_TP:>4} {total_FP:>4} {total_FN:>4}  "
      f"{precision_overall:>6.3f}  {recall_overall:>6.3f}  "
      f"{f1_overall:>6.3f}  {miou_overall:>6.3f}")
print(f"{'='*90}\n")

# ---------- SAVE ----------
summary = {
    "iou_threshold": IOU_THRESHOLD,
    "overall": {
        "total_TP": total_TP,
        "total_FP": total_FP,
        "total_FN": total_FN,
        "precision": round(precision_overall, 4),
        "recall":    round(recall_overall,    4),
        "f1_score":  round(f1_overall,        4),
        "mean_iou":  round(miou_overall,      4)
    },
    "per_image": per_image_results
}

with open("evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("Results saved → evaluation_results.json")
