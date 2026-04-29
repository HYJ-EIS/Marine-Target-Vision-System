import cv2
import numpy as np

# 视频模式下按 track_id 分配的颜色列表（BGR 格式）
_TRACK_COLORS = [
    (0, 0, 255),    # 红
    (0, 255, 0),    # 绿
    (255, 0, 0),    # 蓝
    (0, 255, 255),  # 黄
    (255, 0, 255),  # 品红
    (255, 128, 0),  # 橙
    (0, 128, 255),  # 浅橙
    (128, 255, 0),  # 黄绿
    (255, 0, 128),  # 玫瑰
    (128, 0, 255),  # 紫
]


def visualize_detections(image, detection_type, detection_data, frame_num=None,
                         track_history: dict | None = None):
    """
    检测结果可视化函数
    
    参数:
        image (np.array): 输入图像 (RGB格式)
        detection_type (str): 检测类别，'image' 或 'video'
        detection_data (dict): 检测结果数据，格式为:
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
                                              "class": "USV",          # 目标分类
                                              "class_confidence": 0.8  # 分类置信度
                                          }
                                      ],
                                      "count": 1,
                                      "result_image_path": "./results/result_uuid.jpg"
                                  },
                                  "message": "Success",
                                  "success": true,
                                  "timestamp": 1753774440304,
                                  "type": "infrared"  # infrared：红外, light：可见光
                              }
        frame_num (int): 当前帧数 (仅video模式使用，可选)
    返回:
        np.array: 可视化后的图像
    """
    # 创建图像的副本
    vis_image = image.copy()
    
    # 从检测数据中提取boxes信息
    if not detection_data or not detection_data.get('data') or not detection_data['data'].get('boxes'):
        return vis_image
    
    boxes = detection_data['data']['boxes']

    # 绘制检测框
    for box in boxes:
        # 提取基本信息
        h = box.get('h', 0)
        w = box.get('w', 0)
        x = box.get('x', 0)
        y = box.get('y', 0)
        confidence = box.get('confidence', 0.0)
        target_class = box.get('class', 'unknown')
        # track_id 优先（来自追踪器），否则回退到 id（来自检测器）
        track_id = box.get('track_id', box.get('id', 0))
        class_confidence = box.get('class_confidence', confidence)

        # 转换坐标格式：从(x,y,w,h)到(x1,y1,x2,y2)
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        # 视频模式：按 track_id 分配颜色，更容易区分不同目标
        if detection_type == 'video' and track_id:
            color = _TRACK_COLORS[(int(track_id) - 1) % len(_TRACK_COLORS)]
        elif target_class == 'USV':
            color = (0, 0, 255)
        elif target_class in ('fish_ship', 'fishship'):
            color = (255, 0, 255)
        elif target_class == 'UAV':
            color = (128, 0, 128)
        elif target_class == 'car':
            color = (0, 255, 0)
        else:
            color = (255, 255, 0)

        # 绘制轨迹历史（视频模式，可选）
        if detection_type == 'video' and track_history is not None and track_id in track_history:
            pts = track_history[track_id]
            for i in range(1, len(pts)):
                cv2.line(vis_image, pts[i - 1], pts[i], color, 1, cv2.LINE_AA)

        # 绘制检测框
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

        # 构建标签文本
        if detection_type == 'video':
            label = f"{target_class}:{class_confidence:.2f} ID:{track_id}"
        else:
            label = f"{target_class}:{class_confidence:.2f}"
        
        # 绘制标签背景和文本
        text_y = y1 - 5 if y1 - 5 > 10 else y1 + 20
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis_image, (x1, text_y - text_size[1] - 2), 
                     (x1 + text_size[0], text_y + 2), color, -1)
        cv2.putText(vis_image, label, (x1, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return vis_image
