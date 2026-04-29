import os
import time

import numpy as np

from cv_utils import imread_unicode

from .config import Config
from .detectors import OnnxDetector
from .utils.file_utils import get_file_type


class ImageProcessor:
    def __init__(self):
        print("Initializing Visible Light ONNX Detector...")
        try:
            self.visible_detector = OnnxDetector(
                onnx_path=Config.ONNX_VISIBLE_MODEL_PATH,
                conf_thres=Config.VISIBLE_CONF_THRESH,
            )
        except Exception as e:
            print(f"Failed to load visible detector: {e}")
            self.visible_detector = None

        print("Initializing Infrared ONNX Detector...")
        try:
            self.infrared_detector = OnnxDetector(
                onnx_path=Config.ONNX_INFRARED_MODEL_PATH,
                conf_thres=Config.INFRARED_CONF_THRESH,
            )
        except Exception as e:
            print(f"Failed to load infrared detector: {e}")
            self.infrared_detector = None

        self.processing_times = []
        self._all_detections = []

    def get_average_processing_time(self):
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)

    def reset_processing_times(self):
        self.processing_times = []

    def process(self, image_path, file_type=None, conf_override=None):
        img = imread_unicode(image_path)
        if img is None:
            print(f"无法读取图像: {image_path}")
            return None

        resolved_type = self._resolve_file_type(image_path, file_type)
        stats = self._process_image_array(img, resolved_type, conf_override=conf_override)
        if stats is None:
            return None

        print(f"[INFO] 图像处理完成: {os.path.basename(image_path)}")
        print(f"   - 类型: {resolved_type}")
        print("   - 模式: ONNX模型检测")
        print(f"   - 处理时间: {stats['processing_time']:.3f}s")
        print(f"   - 检测目标: {stats['count']} 个")
        return stats

    def process_frame(self, frame, file_type, conf_override=None):
        """Run detection directly on an in-memory BGR frame."""
        if frame is None:
            print("Input frame is None.")
            return None
        if not isinstance(frame, np.ndarray):
            print(f"Unsupported frame type: {type(frame)}")
            return None
        if frame.size == 0:
            print("Input frame is empty.")
            return None

        resolved_type = self._resolve_file_type("frame.jpg", file_type)
        return self._process_image_array(frame, resolved_type, conf_override=conf_override)

    def _resolve_file_type(self, source_name, file_type):
        if file_type is None or file_type == "unknown":
            file_type = get_file_type(os.path.basename(source_name))
            if file_type == "unknown":
                file_type = "visible"
        return file_type

    def _select_detector(self, file_type):
        if file_type == "visible":
            return self.visible_detector
        return self.infrared_detector

    def _process_image_array(self, img, file_type, conf_override=None):
        detector = self._select_detector(file_type)
        if detector is None:
            print(f"Detector for type '{file_type}' is not available.")
            return None

        start_time = time.time()
        results = detector.detect(img, conf_override=conf_override)
        stats = self._format_detection_results(results, file_type, img)
        stats["processing_time"] = time.time() - start_time
        self.processing_times.append(stats["processing_time"])
        return stats

    def _format_detection_results(self, results, file_type, img):
        self._all_detections = []
        boxes_formatted = []
        for i, result in enumerate(results):
            w = int(result["x2"] - result["x1"])
            h = int(result["y2"] - result["y1"])
            size_str = f"{w}x{h}"

            self._all_detections.append((
                result["x1"],
                result["y1"],
                result["x2"],
                result["y2"],
                result["confidence"],
                size_str,
                result["class"],
                result["class_confidence"],
            ))

            boxes_formatted.append({
                "id": i + 1,
                "confidence": result["confidence"],
                "h": h,
                "w": w,
                "x": int(result["x1"]),
                "y": int(result["y1"]),
                "class": result["class"],
            })

        return {
            "boxes": boxes_formatted,
            "count": len(results),
            "total": len(results),
            "type": file_type,
            "processing_time": 0.0,
            "_frame": img,
        }
