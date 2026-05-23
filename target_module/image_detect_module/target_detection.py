#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
目标识别函数封装模块
提供简化的目标识别接口，输入图片，输出检测结果
"""

import sys

# Windows 控制台默认 GBK 编码，无法输出 UTF-8 中文，强制切换为 UTF-8
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

import cv2
from cv_utils import imread_unicode, imwrite_unicode
import os
import time
import threading
import uuid
import numpy as np
from typing import Dict, List, Optional, Union
from visualization import visualize_detections

# 导入必要的检测组件
# 检查是否作为脚本直接运行
if __name__ == "__main__":
    # 直接运行时使用绝对导入
    import sys
    import os
    # 将当前文件的父目录添加到 Python 路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    
    # 延迟导入ImageProcessor以避免循环导入
    ImageProcessor = None
    from image_detect_module.config import Config
    from image_detect_module.utils.file_utils import ensure_dir, get_file_type
else:
    # 作为模块导入时延迟导入ImageProcessor
    from .image_processor import ImageProcessor
    from .config import Config
    from .utils.file_utils import ensure_dir, get_file_type

class TargetDetector:
    """目标检测器主类

    支持两种模式：
    1. 纯检测模式（默认）：每次调用返回独立检测结果
    2. 追踪模式（enable_tracking=True）：跨帧维护 track_id，
       适合离散图片输入场景（每 1-3s 输入一张图片）
    """

    def __init__(self):
        """初始化目标检测器"""
        self.processor = None
        self._tracker = None
        self._tracking_lock = threading.Lock()
        self._initialize_components()

    def _initialize_components(self):
        """初始化检测组件"""
        try:
            # 动态导入ImageProcessor以避免循环导入
            try:
                self.processor = ImageProcessor()
                print("[OK] 图片处理器初始化成功")
            except ImportError as e:
                print(f"[WARN] ImageProcessor导入失败: {e}")
                self.processor = None

        except Exception as e:
            print(f"[ERROR] 组件初始化失败: {e}")
            self.processor = None

    def _get_tracker(self):
        """惰性初始化追踪器（首次调用时创建）"""
        if self._tracker is None:
            if __name__ == "__main__":
                from image_detect_module.utils.tracker import MultiObjectTracker
            else:
                from .utils.tracker import MultiObjectTracker
            self._tracker = MultiObjectTracker(
                frame_rate=Config.IMAGE_TRACKER_FRAME_RATE,
                tracker_type=Config.TRACKER_TYPE,
                min_hits=Config.IMAGE_TRACKER_MIN_HITS,
            )
        return self._tracker

    def reset_tracker(self):
        """重置追踪状态（清除所有轨迹和 ID 计数器）"""
        if self._tracker is not None:
            self._tracker.reset()

    def detect_from_image_file(self, image_path: str, output_dir: Optional[str] = None,
                               file_type: Optional[str] = None,
                               enable_tracking: bool = False) -> Dict:
        """
        从图片文件路径进行目标检测

        参数:
            image_path (str): 图片文件路径
            output_dir (str): 输出目录，如果为None则使用默认目录
            file_type (str): 图像类型 ('infrared'/'visible')，为None时自动从文件名判断

        返回:
            dict: 检测结果
        """
        start_time = time.time()
        
        try:
            # 1. 验证文件存在
            if not os.path.exists(image_path):
                return self._create_error_response(f"图片文件不存在: {image_path}")
            
            # 2. 验证图片是否可读
            if not self._validate_image(image_path):
                return self._create_error_response("无效的图片文件")
            
            # 3. 执行图片检测
            detection_result = self._perform_detection(image_path, output_dir,
                                                       file_type=file_type,
                                                       enable_tracking=enable_tracking)
            
            # 4. 记录处理时间
            processing_time = time.time() - start_time
            print(f"[INFO] 图片处理完成，耗时: {processing_time:.3f}秒")
            
            return detection_result
            
        except Exception as e:
            print(f"[ERROR] 图片处理过程中发生错误: {e}")
            return self._create_error_response(f"处理失败: {str(e)}")
    
    def _validate_image(self, image_path: str) -> bool:
        """验证图片文件是否有效"""
        try:
            # img = cv2.imread(image_path)
            img = imread_unicode(image_path)
            if img is None:
                return False
            
            # 检查图片尺寸
            height, width = img.shape[:2]
            if height < 10 or width < 10:
                return False
            
            return True
            
        except Exception as e:
            print(f"[ERROR] 图片验证失败: {e}")
            return False
    
    def _perform_detection(self, image_path: str, output_dir: Optional[str] = None,
                           file_type: Optional[str] = None,
                           enable_tracking: bool = False) -> Dict:
        """执行图片检测"""
        try:
            if self.processor is None:
                print("[WARN] 图片处理器不可用")

            # 设置输出目录
            if output_dir:
                ensure_dir(output_dir)
                final_output_dir = output_dir
            else:
                final_output_dir = getattr(Config, 'OUTPUT_DIR', './results')
                ensure_dir(final_output_dir)

            # 执行检测
            detection_stats = self.processor.process(image_path, file_type=file_type)
            
            if detection_stats is None:
                print("error")
                return self._create_error_response("检测过程失败")
            
            # 获取检测框信息
            # 优先使用 detection_stats 中的 boxes (新格式)
            if "boxes" in detection_stats:
                boxes = detection_stats["boxes"]
            else:
                boxes = self._extract_detection_boxes()

            # 追踪：为检测框分配跨帧持久 track_id
            if enable_tracking and boxes:
                frame = detection_stats.pop("_frame", None)
                frame_shape = frame.shape if frame is not None else (1080, 1920, 3)
                with self._tracking_lock:
                    tracked = self._get_tracker().update(boxes, frame_shape, frame=frame)
                for tb in tracked:
                    tb["id"] = tb["track_id"]
                boxes = tracked
            else:
                detection_stats.pop("_frame", None)

            # 确定图片类型
            image_type = detection_stats.get("type", "unknown")

            # 构建成功响应
            response = {
                "data": {
                    "boxes": boxes,
                    "count": len(boxes),
                    "result_image_path": ""  # 不保存结果图片
                },
                "message": "Success",
                "success": True,
                "timestamp": int(time.time() * 1000),
                "type": image_type
            }

            print(f"[INFO] 检测完成: 发现 {len(boxes)} 个目标")
            return response
            
        except Exception as e:
            print(f"[ERROR] 检测执行失败: {e}")
            return self._create_error_response(f"检测执行失败: {str(e)}")
    
    def _extract_detection_boxes(self) -> List[Dict]:
        """提取检测框信息"""
        boxes = []
        all_detections = getattr(self.processor, '_all_detections', [])
        
        # 为每个检测目标分配唯一ID
        detection_id = 1
        for detection in all_detections:
            if len(detection) >= 5:
                x1, y1, x2, y2, conf = detection[:5]
                
                box_info = {
                    "id": detection_id,
                    "confidence": float(conf),
                    "h": int(y2 - y1),
                    "w": int(x2 - x1),
                    "x": int(x1),
                    "y": int(y1)
                }
                
                # 添加分类信息（如果可用）
                if len(detection) >= 7:
                    class_info = detection[6] if detection[6] != "Unknown" else None
                    if class_info:
                        box_info["class"] = str(class_info)
                
                # 添加分类置信度（如果可用）- 使用实际的分类置信度
                if len(detection) >= 5 and len(detection) >= 7:
                    # 如果有分类信息，使用第4个元素作为分类置信度
                    class_conf = detection[4] if detection[4] > 0 else None
                    if class_conf:
                        box_info["class_confidence"] = float(class_conf)
                
                boxes.append(box_info)
                detection_id += 1
        
        return boxes
    
    def _create_error_response(self, error_message: str) -> Dict:
        """创建错误响应"""
        return {
            "data": {
                "boxes": [],
                "count": 0,
                "result_image_path": ""
            },
            "message": error_message,
            "success": False,
            "timestamp": int(time.time() * 1000),
            "type": "unknown"
        }


# 全局检测器实例（单例模式）
_detector_instance = None

def get_detector() -> TargetDetector:
    """获取检测器实例（单例模式）"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = TargetDetector()
    return _detector_instance


def detect_targets(image_input: Union[str, np.ndarray],
                  output_dir: Optional[str] = None,
                  enable_tracking: bool = False) -> Dict:
    """
    目标识别主函数 - 统一接口
    
    参数:
        image_input (str): 图片文件路径
        output_dir (str): 输出目录，如果为None则使用默认目录
        
    返回:
        dict: 检测结果，格式如下:
        {
            "data": {
                "boxes": [
                    {
                        "id": 1,                 # 检测目标唯一ID
                        "confidence": 0.95,      # 置信度
                        "h": 19,                 # 标记矩形高度
                        "w": 43,                 # 标记矩形宽度
                        "x": 676,                # 标记矩形左上角像素坐标x
                        "y": 497,                # 标记矩形左上角像素坐标y
                        "class": "USV",          # 可选：目标分类
                        "class_confidence": 0.8  # 可选：分类置信度
                    }
                ],
                "count": 1,                      # 识别到数量
                "result_image_path": "./results/result_uuid.jpg"  # 保存路径
            },
            "message": "Success",
            "success": true,
            "timestamp": 1753774440304,
            "type": "infrared"                   # infrared：红外, light：可见光
        }
        
    示例使用:
        # 从文件路径检测
        result = detect_targets("path/to/image.jpg")
    """
    try:
        detector = get_detector()
        
        if isinstance(image_input, str):
            # 文件路径输入
            return detector.detect_from_image_file(
                image_path=image_input,
                output_dir=output_dir,
                enable_tracking=enable_tracking,
            )
        else:
            return {
                "data": {"boxes": [], "count": 0, "result_image_path": ""},
                "message": "输入类型错误，只支持文件路径(str)",
                "success": False,
                "timestamp": int(time.time() * 1000),
                "type": "unknown"
            }
    except Exception as e:
        print(f"[ERROR] 目标检测失败: {e}")
        return {
            "data": {"boxes": [], "count": 0, "result_image_path": ""},
            "message": f"检测失败: {str(e)}",
            "success": False,
            "timestamp": int(time.time() * 1000),
            "type": "unknown"
        }
