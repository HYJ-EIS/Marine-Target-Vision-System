"""
图片检测模块

提供目标检测的核心算法，包括：
- 统一的目标检测接口
- ONNX推理检测器
- 配置管理
"""

from .config import Config

__all__ = [
    'Config'
]
