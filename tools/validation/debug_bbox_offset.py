"""诊断检测框偏移问题：对比 GT 与预测框"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from target_module.image_detect_module.config import Config
from target_module.image_detect_module.detectors import OnnxDetector


def load_yolo_labels(label_path: str, img_w: int, img_h: int) -> list[dict]:
    """加载 YOLO 格式标注 → xyxy 像素坐标"""
    labels = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)
            labels.append({"cls_id": cls_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return labels


def center_of(box: dict) -> tuple[float, float]:
    return (box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2


def main():
    img_dir = r"D:\Desktop\fuxian_lake_datasets\new_datasets\UAV\visible"
    label_dir = os.path.join(img_dir, "labels")

    detector = OnnxDetector(
        onnx_path=Config.ONNX_VISIBLE_MODEL_PATH,
        conf_thres=0.3,
        iou_thres=Config.IOU_THRESHOLD,
        classes=Config.CLASSES,
    )
    print(f"Model input shape: {detector.input_shape}")
    print(f"Model img_size: {detector.img_size}")

    images = sorted(Path(img_dir).glob("*.jpg"))[:20]

    offsets_x, offsets_y = [], []
    out_dir = "results/bbox_debug"
    os.makedirs(out_dir, exist_ok=True)

    for img_path in images:
        label_path = os.path.join(label_dir, img_path.stem + ".txt")
        if not os.path.isfile(label_path):
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_boxes = load_yolo_labels(label_path, w, h)
        detections = detector.detect(img)

        vis = img.copy()

        # 画 GT（绿色）
        for gt in gt_boxes:
            cv2.rectangle(vis, (gt["x1"], gt["y1"]), (gt["x2"], gt["y2"]), (0, 255, 0), 2)
            cv2.putText(vis, f"GT cls{gt['cls_id']}", (gt["x1"], gt["y1"] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 画预测（红色）
        for det in detections:
            cv2.rectangle(vis, (det["x1"], det["y1"]), (det["x2"], det["y2"]), (0, 0, 255), 2)
            cv2.putText(vis, f"Pred {det['class']} {det['confidence']:.2f}",
                        (det["x1"], det["y2"] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 匹配 GT ↔ Pred 并计算偏移
        for gt in gt_boxes:
            gt_cx, gt_cy = center_of(gt)
            best_dist = float("inf")
            best_det = None
            for det in detections:
                det_cx, det_cy = center_of(det)
                dist = ((gt_cx - det_cx)**2 + (gt_cy - det_cy)**2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_det = det

            if best_det and best_dist < max(w, h) * 0.3:
                det_cx, det_cy = center_of(best_det)
                dx = det_cx - gt_cx
                dy = det_cy - gt_cy
                offsets_x.append(dx)
                offsets_y.append(dy)
                # 画连线
                cv2.arrowedLine(vis,
                                (int(gt_cx), int(gt_cy)),
                                (int(det_cx), int(det_cy)),
                                (255, 0, 255), 2)
                cv2.putText(vis, f"dx={dx:.0f} dy={dy:.0f}",
                            (int(gt_cx), int(gt_cy) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        out_path = os.path.join(out_dir, img_path.name)
        cv2.imencode('.jpg', vis)[1].tofile(out_path)
        print(f"  {img_path.name}: {len(gt_boxes)} GT, {len(detections)} pred")

    if offsets_x:
        print(f"\n=== 偏移统计 ({len(offsets_x)} 个匹配) ===")
        print(f"  dx: mean={np.mean(offsets_x):.1f}, std={np.std(offsets_x):.1f}")
        print(f"  dy: mean={np.mean(offsets_y):.1f}, std={np.std(offsets_y):.1f}")
        print(f"  总偏移: {np.sqrt(np.mean(offsets_x)**2 + np.mean(offsets_y)**2):.1f}px")
    else:
        print("\n无匹配的 GT/预测对")

    # 额外：打印一个具体的 letterbox 参数来验证
    print("\n=== Letterbox 调试 ===")
    test_img = cv2.imread(str(images[0]))
    if test_img is not None:
        h0, w0 = test_img.shape[:2]
        print(f"  原图: {w0}x{h0}")
        print(f"  模型: {detector.img_size}")
        r = min(detector.img_size[0] / h0, detector.img_size[1] / w0)
        new_unpad_w = int(round(w0 * r))
        new_unpad_h = int(round(h0 * r))
        dw = (detector.img_size[1] - new_unpad_w) / 2
        dh = (detector.img_size[0] - new_unpad_h) / 2
        print(f"  缩放比: {r:.4f}")
        print(f"  缩放后: {new_unpad_w}x{new_unpad_h}")
        print(f"  padding: dw={dw:.1f}, dh={dh:.1f}")

        # 手动测试 letterbox
        lb_img, ratio, (dw2, dh2) = detector.letterbox(test_img, new_shape=detector.img_size, auto=False)
        print(f"  letterbox 实际: shape={lb_img.shape}, dw={dw2:.1f}, dh={dh2:.1f}")


if __name__ == "__main__":
    main()
