import cv2
from cv_utils import imread_unicode, imwrite_unicode
import numpy as np
import onnxruntime as ort
import time
import json  # [新增] 导入json库

class YOLOv5ONNX:
    def __init__(self, onnx_path, classes=['USV', 'fishship', 'UAV'], conf_thres=0.5, iou_thres=0.5):
        """
        classes: 类别名称列表，例如 ['car', 'person']
        """
        self.onnx_path = onnx_path
        self.classes = classes  # [新增] 保存类别名称
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Initialize ONNX Runtime session
        # Use CPUExecutionProvider to avoid CUDA errors if CUDA/cuDNN environment is not perfectly set up
        # If you need GPU, ensure CUDA 12.x and cuDNN 9.x are in PATH
        providers = ['CPUExecutionProvider']
        # providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] 
        
        self.session = ort.InferenceSession(self.onnx_path, providers=providers)
        
        self.inputs = self.session.get_inputs()
        self.input_name = self.inputs[0].name
        self.input_shape = self.inputs[0].shape
        self.input_type = self.inputs[0].type
        self.input_dtype = self._resolve_input_dtype(self.input_type)
        self.img_size = (self.input_shape[2], self.input_shape[3])

        self.outputs = self.session.get_outputs()
        self.output_name = self.outputs[0].name

    @staticmethod
    def _resolve_input_dtype(onnx_type):
        if not isinstance(onnx_type, str):
            return np.float32
        type_key = onnx_type.lower()
        type_map = {
            "tensor(float16)": np.float16,
            "tensor(float32)": np.float32,
            "tensor(float)": np.float32,
            "tensor(float64)": np.float64
        }
        return type_map.get(type_key, np.float32)
        
    def preprocess(self, img_path):
        img0 = imread_unicode(img_path)
        if img0 is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Padded resize (Letterbox)
        img = self.letterbox(img0, new_shape=self.img_size, auto=False)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        # Normalize
        img = img.astype(self.input_dtype)
        img /= np.array(255.0, dtype=self.input_dtype)
        
        if len(img.shape) == 3:
            img = img[None]
            
        return img0, img

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)

        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        elif scaleFill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, ratio, (dw, dh)

    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        if ratio_pad is None:
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]
        coords[:, [1, 3]] -= pad[1]
        coords[:, :4] /= gain
        self.clip_coords(coords, img0_shape)
        return coords

    def clip_coords(self, boxes, shape):
        boxes[:, 0] = boxes[:, 0].clip(0, shape[1])
        boxes[:, 1] = boxes[:, 1].clip(0, shape[0])
        boxes[:, 2] = boxes[:, 2].clip(0, shape[1])
        boxes[:, 3] = boxes[:, 3].clip(0, shape[0])

    def detect(self, img_path, result_save_path="result.jpg"):
        """
        执行检测，返回处理后的图片和 JSON 数据字典
        """
        # 1. Preprocess
        img0, img = self.preprocess(img_path)
        
        # 2. Inference
        pred = self.session.run([self.output_name], {self.input_name: img})[0]
        pred = np.squeeze(pred)

        # 3. Postprocess
        scores = pred[:, 4] * pred[:, 5]
        mask = scores > self.conf_thres
        
        pred = pred[mask]
        scores = scores[mask]
        
        # 初始化返回的数据结构
        json_data = {
            "data": {
                "boxes": [],
                "count": 0,
                "result_image_path": result_save_path
            }
        }

        if len(pred) == 0:
            print("No detections found.")
            return img0, json_data  # 返回空结果

        boxes = self.xywh2xyxy(pred[:, :4])
        
        # NMS
        boxes_nms = np.array(pred[:, :4])
        boxes_nms[:, 0] = boxes_nms[:, 0] - boxes_nms[:, 2] / 2
        boxes_nms[:, 1] = boxes_nms[:, 1] - boxes_nms[:, 3] / 2
        
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_nms.tolist(), 
            scores=scores.tolist(), 
            score_threshold=self.conf_thres, 
            nms_threshold=self.iou_thres
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            det_boxes = boxes[indices]
            det_scores = scores[indices]
            # 获取每个框对应的类别索引 (假设第5列之后是类别概率)
            det_class_indices = np.argmax(pred[indices, 5:], axis=1)
            
            # Rescale boxes
            det_boxes = self.scale_coords(self.img_size, det_boxes, img0.shape).round()
            
            # [修改] 遍历检测结果，同时画图和填充JSON
            for i, box in enumerate(det_boxes):
                x1, y1, x2, y2 = map(int, box)
                score = float(det_scores[i]) # 必须转为 Python float
                class_idx = int(det_class_indices[i])
                
                # 获取类名 (防止索引越界)
                class_name = self.classes[class_idx] if class_idx < len(self.classes) else "unknown"

                # 1. 计算 JSON 需要的格式 (x, y, w, h)
                w = x2 - x1
                h = y2 - y1
                
                # 2. 构建单个目标的字典
                # confidence: 总体置信度 (obj_conf * class_conf)
                # class_confidence: 类别条件概率 (class_conf), 即 pred[5:] 里的原始值
                # 在 YOLOv5 postprocess 中，score = obj_conf * class_conf
                # 所以要还原原始 class_conf, 可以用 score / obj_conf (前提是有obj_conf)
                # 但这里的 pred 已经是经过 NMS 筛选的，我们可能拿不到原始未乘积的 obj_conf
                # 或者，我们可以简单地把 class_conf 也设为 score，因为在单类别检测中它们通常很接近

                # 更准确的做法（如果我们有保留中间变量）：
                # raw_obj_conf = pred[indices[i], 4] 
                # raw_class_conf = pred[indices[i], 5:]
                
                # 在当前简化逻辑下，我们直接使用 score (即最终置信度)
                box_info = {
                    "id": i + 1,
                    "confidence": round(score, 4), # Final Score
                    "h": h,
                    "w": w,
                    "x": x1,
                    "y": y1,
                    "class": class_name,
                    # "class_confidence": round(score, 4) # 重复了，如果你想要区分，可以尝试修改逻辑获取原始class_conf
                }
                json_data["data"]["boxes"].append(box_info)

                # 3. 画图 (Draw)
                label = f"{class_name} {score:.2f}"
                cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img0, (x1, y1), c2, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.putText(img0, label, (x1, y1 - 2), 0, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            
            # 更新总数量
            json_data["data"]["count"] = len(det_boxes)
            print(f"Detected {len(det_boxes)} objects.")
            
        return img0, json_data


if __name__ == "__main__":
    # 配置路径
    onnx_file = r"D:\Desktop\FFCA-YOLO-main-copy\runs\train\car_ir_finetune_hard_stage2\weights\best.onnx"
    img_file = r"D:\Desktop\car_datasets\ir_hard\images\1_000055_000.jpg"
    output_img_path = "result_onnx.jpg"
    
    # 如果你的模型训练时有多个类别（如 car, ship），请在这里按顺序修改列表
    my_classes = ["car"] 
    
    model = YOLOv5ONNX(onnx_file, classes=my_classes)
    
    # [修改] 接收两个返回值：图片 和 JSON数据
    result_img, result_json = model.detect(img_file, result_save_path=output_img_path)
    
    # 保存图片
    imwrite_unicode(output_img_path, result_img)
    print(f"Image saved to {output_img_path}")
    
    # [新增] 打印 JSON
    print("-" * 30)
    print("生成的 JSON 数据:")
    print(json.dumps(result_json, indent=4, ensure_ascii=False))
    
    # [可选] 将 JSON 保存到文件
    with open("result.json", "w", encoding='utf-8') as f:
        json.dump(result_json, f, indent=4, ensure_ascii=False)