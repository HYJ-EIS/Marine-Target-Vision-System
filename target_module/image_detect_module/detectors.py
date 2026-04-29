import cv2
from cv_utils import imwrite_unicode
import numpy as np
import os
import sys

# 将 PyTorch 自带的 cuDNN DLL 目录加入 PATH，使 ONNX Runtime 能找到 cudnn64_9.dll
try:
    _torch_module = __import__("torch")
except Exception:
    _torch_module = None

if _torch_module is not None:
    _torch_lib = os.path.join(os.path.dirname(os.path.abspath(_torch_module.__file__)), "lib")
    if os.path.isdir(_torch_lib) and _torch_lib not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")

import onnxruntime as ort
class OnnxDetector:
    def __init__(self, onnx_path, classes=None, conf_thres=None, iou_thres=None):
        from .config import Config
        if classes is None:
            classes = Config.CLASSES
        if conf_thres is None:
            conf_thres = Config.VISIBLE_CONF_THRESH
        if iou_thres is None:
            iou_thres = Config.IOU_THRESHOLD
        self.onnx_path = onnx_path
        self.classes = classes
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Initialize ONNX Runtime session
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
             providers.insert(0, 'CUDAExecutionProvider')
             
        # print(f"Loading ONNX model from {self.onnx_path} using providers: {providers}")
        self.session = ort.InferenceSession(self.onnx_path, providers=providers)
        
        self.inputs = self.session.get_inputs()
        self.input_name = self.inputs[0].name
        self.input_shape = self.inputs[0].shape
        self.input_type = self.inputs[0].type
        self.input_dtype = self._resolve_input_dtype(self.input_type)
        
        # print(f"🔍 DEBUG: Model {onnx_path}")
        # print(f"   Input Name: {self.input_name}")
        # print(f"   Input Shape: {self.input_shape}")
        
        # Determine image size from input shape (assuming NCHW or similar)
        # Handle dynamic axes (strings) or fixed sizes
        if len(self.input_shape) == 4:
            h = self.input_shape[2]
            w = self.input_shape[3]
            # Default to 640 if dynamic
            if isinstance(h, str): h = 640
            if isinstance(w, str): w = 640
            self.img_size = (h, w)
        else:
            self.img_size = (640, 640)

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

    def preprocess(self, img0):
        img = self.letterbox(img0, new_shape=self.img_size, auto=False)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = img.astype(self.input_dtype)
        img /= np.array(255.0, dtype=self.input_dtype)
        if len(img.shape) == 3:
            img = img[None]
        return img

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
        
    def detect(self, img, conf_override=None):
        img_prep = self.preprocess(img)
        conf_thres = self.conf_thres if conf_override is None else float(conf_override)
        
        # Handle fixed batch size mismatch
        # Some models are exported with fixed batch size (e.g. 4)
        input_batch_size = self.input_shape[0] if isinstance(self.input_shape[0], int) else 1
        
        if input_batch_size > 1 and img_prep.shape[0] < input_batch_size:
            pad_size = input_batch_size - img_prep.shape[0]
            pad = np.zeros((pad_size, *img_prep.shape[1:]), dtype=img_prep.dtype)
            img_input = np.concatenate([img_prep, pad], axis=0)
        else:
            img_input = img_prep

        # print(f"DEBUG: Detect info:")
        # print(f"  Input Name: {self.input_name}")
        # print(f"  Input Shape: {img_input.shape}")
        
        pred = self.session.run([self.output_name], {self.input_name: img_input})[0]
        
        # Handle batch dimension in output
        # If output is (Batch, Anchors, Vals), we want (Anchors, Vals) for the first image
        if pred.ndim == 3:
            pred = pred[0]
        
        pred = np.squeeze(pred)
        
        detections = []
        if pred.ndim == 1:
            pred = pred[None] 
            
        if pred.size == 0 or len(pred.shape) < 2:
             return detections

        # In YOLOv5/v8-like outputs
        # Column 4 is objectness confidence, 5+ are class probabilities
        if pred.shape[1] > 5:
            scores = pred[:, 4] * np.max(pred[:, 5:], axis=1) # Objectness * Max Class Prob
        else:
            scores = pred[:, 4] # Just objectness if no class probs? Or single class?
            # Assuming single class scenario where col 4 is confidence if shape is (N, 5) x,y,w,h,conf
            if pred.shape[1] == 5:
                 scores = pred[:, 4]

        mask = scores > conf_thres
        
        if not np.any(mask):
             return detections

        pred = pred[mask]
        scores = scores[mask]
        
        if len(pred) > 0:
            boxes = self.xywh2xyxy(pred[:, :4])
            
            boxes_nms = np.array(pred[:, :4])
            boxes_nms[:, 0] = boxes_nms[:, 0] - boxes_nms[:, 2] / 2
            boxes_nms[:, 1] = boxes_nms[:, 1] - boxes_nms[:, 3] / 2
            
            indices = cv2.dnn.NMSBoxes(
                bboxes=boxes_nms.tolist(), 
                scores=scores.tolist(), 
                score_threshold=conf_thres, 
                nms_threshold=self.iou_thres
            )
            
            if len(indices) > 0:
                indices = indices.flatten()
                det_boxes = boxes[indices]
                det_scores = scores[indices]
                
                det_boxes = self.scale_coords(self.img_size, det_boxes, img.shape).round()
                
                for i, idx in enumerate(indices):
                    x1, y1, x2, y2 = map(int, det_boxes[i])
                    score = float(det_scores[i])
                    
                    if pred.shape[1] > 5:
                        class_probs = pred[idx, 5:]
                        class_idx = np.argmax(class_probs) if len(class_probs) > 0 else 0
                    else:
                        class_idx = 0
                    
                    class_name = self.classes[class_idx] if class_idx < len(self.classes) else "unknown"
                    
                    detections.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "confidence": score,
                        "class": class_name,
                        "class_confidence": score
                    })

        return detections
