import os


class Config:
    # 项目根目录，默认取当前工作目录
    BASE_DIR = os.path.abspath(".")
    # 图片接口上传文件保存目录
    INPUT_FOLDER = os.path.join(BASE_DIR, "uploads")
    # 检测结果、输出视频和派生文件默认输出目录
    OUTPUT_DIR = os.path.join(BASE_DIR, "results")

    # 检测类别列表，索引顺序需与模型输出类别顺序一致
    CLASSES = ["USV", "fishship", "UAV"]

    # NMS 使用的 IoU 阈值
    IOU_THRESHOLD = 0.5
    # 可见光模型的默认置信度阈值
    VISIBLE_CONF_THRESH = 0.5
    # 红外模型的默认置信度阈值
    INFRARED_CONF_THRESH = 0.65

    # 可见光 ONNX 模型文件路径
    ONNX_VISIBLE_MODEL_PATH = os.path.join(BASE_DIR, "target_module", "models", "A_S_F_rgb_FFCA.onnx")
    # 红外 ONNX 模型文件路径
    ONNX_INFRARED_MODEL_PATH = os.path.join(BASE_DIR, "target_module", "models", "A_S_F_ir_FFCA.onnx")

    # 视频跟踪默认算法，可选 bytetrack / ocsort / botsort / dist_tracker / official_ocsort / official_botsort
    TRACKER_TYPE = "botsort"
    # 新轨迹激活所需的最低检测置信度
    TRACKER_ACTIVATION_THRESH = 0.3
    # 轨迹丢失后允许保留的最大帧数
    TRACKER_LOST_BUFFER = 80
    # 轨迹与检测框关联时的匹配阈值
    TRACKER_MATCH_THRESH = 0.2
    # 轨迹从候选状态转为确认状态所需的最少命中次数
    TRACKER_MIN_HITS = 3
    # 基于中心点距离做关联时的距离阈值，单位像素
    TRACKER_DIST_THRESH = 150.0

    # Dist-Tracker FLIT 代价中 L2 距离权重，IoU distance 权重为 1 - gamma
    DIST_TRACKER_GAMMA = 0.25
    # Dist-Tracker FLIT 匹配代价阈值，越小越严格
    DIST_TRACKER_MATCH_THRESH = 0.8
    # Dist-Tracker FLIT 是否融合检测置信度
    DIST_TRACKER_FUSE_SCORE = True

    # 全局运动补偿算法类型，主要供 BoT-SORT 使用
    GMC_METHOD = "sparse_flow"
    # 全局运动补偿前的下采样倍率
    GMC_DOWNSCALE = 2

    # 图片离散跟踪场景下首次输出轨迹所需的最少命中次数
    IMAGE_TRACKER_MIN_HITS = 1
    # 图片离散跟踪场景假定的输入帧率，单位 Hz
    IMAGE_TRACKER_FRAME_RATE = 1.0

    # RTSP 输入流地址
    VIDEO_RTSP_INPUT = "rtsp://localhost:8554/video"
    # RTSP 输出流地址
    VIDEO_RTSP_OUTPUT = "rtsp://localhost:8554/output"
    # 视频处理默认本地输出文件路径
    VIDEO_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "output.mp4")

    # 批量抽帧任务的输入视频根目录
    EXTERNAL_FRAMES_INPUT_ROOT = r"D:\Desktop\烟台项目数据\原始数据集\视频"
    # 批量抽帧任务的输出根目录
    EXTERNAL_FRAMES_OUTPUT_ROOT = r"D:\Desktop\烟台项目数据\原始数据集\external_frames"
    # 批量抽帧任务默认使用的跟踪算法
    EXTERNAL_FRAMES_TRACKER = "botsort"
    # 批量抽帧任务中轨迹确认所需的最少命中次数
    EXTERNAL_FRAMES_TRACKER_MIN_HITS = 1
    # 批量抽帧任务中轨迹最多允许连续丢失的帧数
    EXTERNAL_FRAMES_TRACKER_MAX_MISSED = 2
    # 批量抽帧任务中可见光检测置信度阈值
    EXTERNAL_FRAMES_VISIBLE_CONF_THRESH = 0.45
    # 批量抽帧任务中红外检测置信度阈值
    EXTERNAL_FRAMES_INFRARED_CONF_THRESH = 0.45
    # 判定目标发生位移变化的最小运动阈值
    EXTERNAL_FRAMES_MOTION_THRESHOLD = 0.15
    # 判定目标姿态变化时使用的角度阈值，单位度
    EXTERNAL_FRAMES_POSE_ANGLE_THRESHOLD_DEG = 12.0
    # 判定目标面积变化时使用的面积变化阈值
    EXTERNAL_FRAMES_AREA_THRESHOLD = 0.12
    # RGB / IR 成对样本时间对齐允许的最大时间差，单位秒
    EXTERNAL_FRAMES_PAIR_TOLERANCE_SEC = 0.25
    # 感知哈希相似度阈值，值越小要求越相似
    EXTERNAL_FRAMES_PHASH_HAMMING_THRESHOLD = 4
    # 活跃场景最小时间间隔，低于该值视为同一连续片段，单位秒
    EXTERNAL_FRAMES_ACTIVE_SCENE_MIN_GAP_SEC = 0.4
    # 活跃场景最大时间间隔，超过该值会切分片段，单位秒
    EXTERNAL_FRAMES_ACTIVE_SCENE_MAX_GAP_SEC = 1.2
    # 全图级代表帧聚类使用的哈希距离阈值
    EXTERNAL_FRAMES_ACTIVE_SCENE_GLOBAL_HASH_THRESHOLD = 10
    # 目标区域级代表帧聚类使用的哈希距离阈值
    EXTERNAL_FRAMES_ACTIVE_SCENE_TARGET_HASH_THRESHOLD = 8
    # 用于判定检测结果具有新颖性的运动变化阈值
    EXTERNAL_FRAMES_DETECTION_NOVELTY_MOTION_THRESHOLD = 0.30
    # 用于判定检测结果具有新颖性的面积变化阈值
    EXTERNAL_FRAMES_DETECTION_NOVELTY_AREA_THRESHOLD = 0.20
    # 用于判定检测结果具有新颖性的姿态角变化阈值，单位度
    EXTERNAL_FRAMES_DETECTION_NOVELTY_POSE_THRESHOLD_DEG = 18.0
    # 空场景样本之间的最小时间间隔，单位秒
    EXTERNAL_FRAMES_EMPTY_SCENE_MIN_GAP_SEC = 1.0
    # 是否导出空场景帧
    EXTERNAL_FRAMES_INCLUDE_EMPTY_FRAMES = False
    # 是否允许用跟踪结果补齐标签
    EXTERNAL_FRAMES_ALLOW_TRACKER_FILL_LABELS = False
    # 是否开启代表帧裁剪，减少重复样本
    EXTERNAL_FRAMES_ENABLE_REPRESENTATIVE_PRUNING = True
    # 成对样本筛选时是否优先参考 RGB 结果
    EXTERNAL_FRAMES_PAIRED_RGB_REFERENCE = True
    # 成对参考时是否包含空目标场景
    EXTERNAL_FRAMES_PAIRED_REFERENCE_INCLUDE_EMPTY_TARGET = True
    # 代表帧筛选时全图哈希聚类阈值
    EXTERNAL_FRAMES_REPRESENTATIVE_GLOBAL_HASH_THRESHOLD = 14
    # 代表帧筛选时目标区域哈希聚类阈值
    EXTERNAL_FRAMES_REPRESENTATIVE_TARGET_HASH_THRESHOLD = 14
    # 代表帧筛选时运动变化阈值
    EXTERNAL_FRAMES_REPRESENTATIVE_MOTION_THRESHOLD = 0.55
    # 代表帧筛选时面积变化阈值
    EXTERNAL_FRAMES_REPRESENTATIVE_AREA_THRESHOLD = 0.35
    # 单个代表帧聚类允许覆盖的最大时间跨度，单位秒
    EXTERNAL_FRAMES_REPRESENTATIVE_MAX_CLUSTER_SPAN_SEC = 6.0
    # 相邻代表帧之间要求的最小时间间隔，单位秒
    EXTERNAL_FRAMES_REPRESENTATIVE_MIN_TIME_GAP_SEC = 0.6
    # 计算姿态角时目标区域所需的最小像素尺寸
    EXTERNAL_FRAMES_POSE_MIN_PIXELS = 12
    # 计算姿态角时轮廓点数量下限
    EXTERNAL_FRAMES_POSE_MIN_POINTS = 20
    # 计算姿态角时目标最小长宽比要求
    EXTERNAL_FRAMES_POSE_MIN_ASPECT_RATIO = 1.15
    # 计算姿态角时目标最小面积占比要求
    EXTERNAL_FRAMES_POSE_MIN_AREA_RATIO = 0.01

    # 数据集批处理任务的输入根目录
    DATASET_INPUT_ROOT = r"D:\Desktop\烟台项目数据\原始数据集\视频"
    # 数据集批处理结果输出目录
    DATASET_OUTPUT_ROOT = os.path.join(OUTPUT_DIR, "video_dataset_analysis")
    # 是否在批处理任务中启用断点续跑
    DATASET_ENABLE_RESUME = False
    # 数据集批处理默认并行进程数
    DATASET_MAX_WORKERS = 2
    # 场景分析时间窗长度，单位秒
    DATASET_WINDOW_SECONDS = 1.0
    # 一个时间窗被视为有效所需的最少帧数
    DATASET_MIN_FRAMES_PER_WINDOW = 8
    # 低帧率时间窗允许扩展到的最大时长，单位秒
    DATASET_LOW_FPS_WINDOW_MAX_SECONDS = 4.0

    # 海天场景中天空占比的判定阈值
    DATASET_SCENE_SKY_THRESHOLD = 0.20
    # 近岸场景下岸线占比的低阈值
    DATASET_SCENE_SHORELINE_LOW_THRESHOLD = 0.10
    # 近岸场景下岸线占比的高阈值
    DATASET_SCENE_SHORELINE_HIGH_THRESHOLD = 0.40
    # 场景分析前对图像进行缩放时允许的最大边长
    DATASET_SCENE_ANALYSIS_MAX_DIM = 320

    # 红外场景判定时主导区域占比的最小要求
    DATASET_T_SCENE_DOMINANT_RATIO_MIN = 0.50
    # 红外场景判定时整帧亮度方差下限
    DATASET_T_SCENE_MIN_INTENSITY_VARIANCE = 150.0
    # 红外场景判定时地平线置信度下限
    DATASET_T_SCENE_MIN_HORIZON_CONFIDENCE = 1.35

    # 目标尺度分桶时“小目标”的面积占比阈值
    DATASET_SCALE_SMALL_THRESHOLD = 0.0025
    # 目标尺度分桶时“中目标”的面积占比阈值
    DATASET_SCALE_MEDIUM_THRESHOLD = 0.02

    # 红外目标尺度默认校准系数
    DATASET_T_SCALE_DEFAULT = 1.5
    # 执行 RGB / IR 尺度校准所需的最少重叠样本数
    DATASET_T_CALIB_MIN_OVERLAP_COUNT = 50
    # 配对窗口被视为对齐所需的最小重叠比例
    DATASET_ALIGNMENT_MIN_OVERLAP_RATIO = 0.5

    # 视频被判定为可用所需的最小可用时间窗比例
    DATASET_USABLE_WINDOW_RATIO_THRESHOLD = 0.8
    # 视频被判定为不可用前允许连续出现的最大不可用时间窗数
    DATASET_MAX_CONSECUTIVE_UNUSABLE_WINDOWS = 3
