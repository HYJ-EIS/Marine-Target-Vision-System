"""统一多目标跟踪接口。

运行时支持 supervision ByteTrack，以及仓库内 vendored 官方源码的
OC-SORT 和 BoT-SORT。MS-DC-ELT 使用独立的生命周期入口，不经过本模块。

输入格式：
    [{"x": int, "y": int, "w": int, "h": int,
      "confidence": float, "class": str}, ...]

输出格式：
    [{"track_id": int, "x": int, "y": int, "w": int, "h": int,
      "confidence": float, "class": str}, ...]
"""

from __future__ import annotations

import numpy as np

from target_module.image_detect_module.config import Config
from target_module.image_detect_module.utils.official_adapter import OfficialTrackerAdapter
from target_module.image_detect_module.utils.msdc_detection import _get_default_conf_thresh


class MultiObjectTracker:
    """将项目检测框格式适配到保留的 tracker 实现。"""

    _CLASS_TO_ID: dict[str, int] = {c: i for i, c in enumerate(Config.CLASSES)}
    _ID_TO_CLASS: dict[int, str] = {i: c for i, c in enumerate(Config.CLASSES)}

    def __init__(
        self,
        frame_rate: float = 25.0,
        tracker_type: str | None = None,
        min_hits: int | None = None,
    ):
        self.frame_rate = max(1.0, float(frame_rate))
        self.tracker_type = tracker_type or Config.TRACKER_TYPE
        default_min_hits = (
            Config.BYTETRACK_MIN_HITS if self.tracker_type == "bytetrack"
            else Config.TRACKER_MIN_HITS
        )
        self._min_hits = min_hits if min_hits is not None else default_min_hits
        self._file_type = "visible"

        if self.tracker_type == "bytetrack":
            import supervision as sv

            self._sv = sv
            self._bt = self._new_bytetrack()
        elif self.tracker_type in ("official_ocsort", "official_botsort"):
            self._official = OfficialTrackerAdapter(
                tracker_type=self.tracker_type,
                frame_rate=self.frame_rate,
                min_hits=self._min_hits,
            )
        else:
            raise ValueError(f"不支持的追踪器类型: {self.tracker_type}")

        print(f"[Tracker] 已初始化 {self.tracker_type} (fps={self.frame_rate:.1f})")

    def _new_bytetrack(self):
        threshold = _get_default_conf_thresh(self._file_type)
        tracker = self._sv.ByteTrack(
            track_activation_threshold=threshold,
            lost_track_buffer=Config.BYTETRACK_LOST_BUFFER,
            minimum_matching_threshold=Config.BYTETRACK_MATCH_COST_THRESH,
            frame_rate=self.frame_rate,
            # supervision 从创建后的第一次 update 开始计数。
            minimum_consecutive_frames=max(1, self._min_hits - 1),
        )
        # 原库默认新轨迹阈值为 high + 0.1，并按 FPS / 30 缩放 buffer。
        tracker.det_thresh = threshold
        tracker.max_time_lost = int(Config.BYTETRACK_LOST_BUFFER)
        return tracker

    def update(
        self,
        detections: list[dict],
        frame_shape: tuple,
        frame: np.ndarray | None = None,
        file_type: str | None = None,
    ) -> list[dict]:
        """对单帧执行跟踪更新。"""
        if self.tracker_type == "bytetrack":
            if file_type is not None:
                self._file_type = file_type
            threshold = _get_default_conf_thresh(self._file_type)
            self._bt.track_activation_threshold = threshold
            self._bt.det_thresh = threshold
            return self._update_bytetrack(detections, frame_shape)
        return self._official.update(detections, frame_shape, frame=frame)

    def reset(self) -> None:
        """重置跟踪状态。"""
        if self.tracker_type == "bytetrack":
            self._bt = self._new_bytetrack()
        else:
            self._official.reset()

    def get_recent_tracks(self, max_time_since_update: int = 1) -> list[dict]:
        """返回适配器提供的短间隔预测轨迹；不支持时返回空列表。"""
        if max_time_since_update <= 0 or self.tracker_type == "bytetrack":
            return []
        return self._official.get_recent_tracks(max_time_since_update=max_time_since_update)

    def _update_bytetrack(
        self,
        detections: list[dict],
        frame_shape: tuple,
    ) -> list[dict]:
        sv = self._sv
        if not detections:
            self._bt.update_with_detections(sv.Detections.empty())
            return []

        height, width = frame_shape[:2]
        xyxy_list: list[list[float]] = []
        conf_list: list[float] = []
        cls_list: list[int] = []
        for detection in detections:
            x1 = float(detection["x"])
            y1 = float(detection["y"])
            x2 = x1 + float(detection["w"])
            y2 = y1 + float(detection["h"])
            x1, x2 = max(0.0, x1), min(float(width), x2)
            y1, y2 = max(0.0, y1), min(float(height), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            xyxy_list.append([x1, y1, x2, y2])
            conf_list.append(float(detection.get("confidence", 0.0)))
            cls_list.append(
                int(self._CLASS_TO_ID.get(detection.get("class", Config.CLASSES[0]), 0))
            )

        if not xyxy_list:
            return []

        tracked = self._bt.update_with_detections(
            sv.Detections(
                xyxy=np.array(xyxy_list, dtype=np.float32),
                confidence=np.array(conf_list, dtype=np.float32),
                class_id=np.array(cls_list, dtype=int),
            )
        )
        if tracked.tracker_id is None or len(tracked) == 0:
            return []

        results = []
        for index in range(len(tracked)):
            x1, y1, x2, y2 = tracked.xyxy[index]
            confidence = float(tracked.confidence[index])
            class_id = int(tracked.class_id[index])
            results.append({
                "track_id": int(tracked.tracker_id[index]),
                "x": int(x1),
                "y": int(y1),
                "w": int(x2 - x1),
                "h": int(y2 - y1),
                "confidence": round(confidence, 4),
                "class": self._ID_TO_CLASS.get(class_id, Config.CLASSES[0]),
                "class_confidence": round(confidence, 4),
            })
        return results
