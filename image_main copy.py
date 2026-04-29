import sys

# Windows 控制台默认 GBK 编码，无法输出 UTF-8 中文，强制切换为 UTF-8
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")

import cv2
from cv_utils import imread_unicode, imwrite_unicode
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import visualization as vis
import os
import time
import logging
import uuid
import atexit
# Fix cuDNN 9 discovery: add PyTorch's lib dir to PATH so ONNX Runtime finds cudnn64_9.dll
import torch as _torch
_torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")

# 尝试导入CORS，如果失败则使用None
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
    print("Warning: Flask-CORS not available. Install with: pip install Flask-CORS")
# 导入MQ消息发布模块
try:
    import messaging.mq_publisher as mq
    MQ_PUBLISHER_AVAILABLE = True
    print("成功导入MQ发布模块")
except ImportError as e:
    MQ_PUBLISHER_AVAILABLE = False
    print(f"❌ 导入MQ发布模块失败: {e}")
try:
    import target_module.image_detect_module.target_detection as td
    IMAGE_DETECTION_AVAILABLE = True
    print("✅ 成功导入image_detection模块")
except ImportError as e:
    IMAGE_DETECTION_AVAILABLE = False
    print(f"❌ 导入image_detection模块失败: {e}")

try:
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
except Exception as e:
    print(f"日志配置失败: {e}")
    logger = None

# Flask应用初始化
app = Flask(__name__)

# 条件性地启用CORS
if CORS_AVAILABLE:
    CORS(app)
    print("CORS enabled")

# 配置
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB最大文件大小
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['RESULT_FOLDER'] = './results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
# 确保目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)


class APIResponse:
    """标准API响应格式"""

    @staticmethod
    def success(data=None, message="Success"):
        """成功响应"""
        # return {
        #     'success': True,
        #     'message': message,
        #     'data': data,
        #     'timestamp': int(time.time() * 1000)
        # }

    @staticmethod
    def error(message, error_code=400, details=None):
        """错误响应"""
        return {
            'success': False,
            'message': message,
            'error_code': error_code,
            'details': details,
            'timestamp': int(time.time() * 1000)
        }


def allowed_file(filename):
    """检查文件类型是否被允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def save_uploaded_file(file):
    """保存上传的文件"""
    if not file or not allowed_file(file.filename):
        return None

    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

    try:
        file.save(filepath)
        return filepath
    except Exception as e:
        logger.error(f"File save failed: {e}")
        return None


@app.route('/api/v1/detect/image', methods=['POST'])
@app.route('/api/detect/image', methods=['POST'])  # 兼容性路由
def detect_image():
    """单张图像目标检测 - 检测结果仅发送到MQ队列，HTTP只返回接收确认"""
    try:
        print("Received image detection request")
        if not IMAGE_DETECTION_AVAILABLE:
            return jsonify(APIResponse.error("Detection service not available")), 503

        filepath = None

        # 检查请求类型
        if 'file' in request.files:
            # 文件上传模式
            file = request.files['file']
            if not file or file.filename == '':
                return jsonify(APIResponse.error("No file provided")), 400

            # 保存文件
            filepath = save_uploaded_file(file)
            if not filepath:
                return jsonify(APIResponse.error("Invalid file type or save failed")), 400
            file.seek(0)  # 重置文件指针

        elif request.is_json:
            # JSON模式 - Base64图像数据或图像路径
            data = request.get_json()
            if not data:
                return jsonify(APIResponse.error("No JSON data provided")), 400

            # 检查是否提供了图像路径
            image_path = data.get('image_path')
            if image_path:
                # 直接使用提供的图像路径
                if not os.path.exists(image_path):
                    return jsonify(APIResponse.error(f"Image file not found: {image_path}")), 400
                filepath = image_path

        else:
            # 添加调试信息
            logger.warning(f"Request content type: {request.content_type}")
            logger.warning(f"Request files: {list(request.files.keys())}")
            logger.warning(f"Request form: {dict(request.form)}")
            logger.warning(f"Request is_json: {request.is_json}")

            return jsonify(APIResponse.error(
                "No valid input provided. Use file upload (multipart/form-data) or JSON with image_path/image_data. "
                f"Received content-type: {request.content_type}"
            )), 400

        # 使用image_detection模块进行检测
        try:
            logger.info(f"开始检测图像: {filepath}")
            detection_result = td.detect_targets(filepath, "./", enable_tracking=True)
            # 发送检测结果到MQ
            if MQ_PUBLISHER_AVAILABLE and detection_result:
                mq.send_image_result_to_mq(detection_result)
            # 最终检测结果可视化并保存
            img = imread_unicode(filepath)
            if img is None:
                print(f"无法读取图像: {filepath}")
                return None
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            vis_image = vis.visualize_detections(img_rgb, 'video', detection_result)

            # 保存可视化结果图像到统一结果目录
            result_folder = app.config['RESULT_FOLDER']
            os.makedirs(result_folder, exist_ok=True)

            # 生成唯一的结果文件名，并回写 result_id 供 GET 接口读取
            result_id = str(uuid.uuid4())
            result_filename = f"result_{result_id}.jpg"
            result_path = os.path.join(result_folder, result_filename)

            # 保存可视化图像
            vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            imwrite_unicode(result_path, vis_image_bgr)
            logger.info(f"图片处理结果已保存到: {result_path}")

            # 将结果图片路径和 result_id 写回检测结果
            detection_result['data']['result_image_path'] = os.path.abspath(result_path)
            detection_result['data']['result_id'] = result_id

            # 检查检测结果
            if not detection_result.get('success'):
                logger.error(f"检测失败: {detection_result.get('message')}")
                # 不返回数据，仅返回接收成功
                return jsonify(APIResponse.success(None, "Detection request received, results sent to MQ"))

            logger.info(
                f"检测完成: 发现 {detection_result.get('count', 0)} 个目标")

            # 不返回检测数据，仅确认接收成功并已发送到MQ
            return jsonify(APIResponse.success(None, "Detection request received, results sent to MQ"))

        except Exception as detection_error:
            logger.error(f"Detection execution failed: {detection_error}")
            # 不返回检测数据，仅确认接收成功
            return jsonify(APIResponse.success(None, "Detection request received but failed, error sent to MQ"))

    except Exception as e:
        logger.error(f"Image detection failed: {e}")
        return jsonify(APIResponse.error(f"Detection failed: {str(e)}")), 500

    finally:
        # 清理上传的文件（如果是文件上传模式）
        if filepath and 'file' in request.files:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup uploaded file: {cleanup_error}")


@app.route('/api/v1/results/<result_id>', methods=['GET'])
def get_result_image(result_id):
    """获取检测结果图像"""
    try:
        result_filename = f"result_{result_id}.jpg"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)

        if not os.path.exists(result_path):
            return jsonify(APIResponse.error("Result image not found")), 404

        return send_file(result_path, mimetype='image/jpeg')

    except Exception as e:
        return jsonify(APIResponse.error(f"Failed to get result image: {str(e)}")), 500

# ================ 错误处理 ================

@app.errorhandler(413)
def request_entity_too_large(error):
    """文件过大错误处理"""
    return jsonify(APIResponse.error("File too large. Maximum size is 100MB.")), 413


@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return jsonify(APIResponse.error("Endpoint not found")), 404


@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    return jsonify(APIResponse.error("Internal server error")), 500


# ================ 应用启动 ================

if __name__ == '__main__':
    # 记录启动时间
    app.config['start_time'] = time.time()

    # 初始化检测模块
    if IMAGE_DETECTION_AVAILABLE:
        try:
            def print_processing_stats():
                """输出处理统计信息"""
                detector = td.get_detector()
                if hasattr(detector, 'processor') and hasattr(detector.processor, 'processing_times'):
                    times = detector.processor.processing_times
                    if times:
                        avg_time = sum(times) / len(times)
                        logger.info(f"📊 图片处理统计:")
                        logger.info(f"   - 总处理图片数: {len(times)}")
                        logger.info(f"   - 平均处理时间: {avg_time:.3f}秒")
                        logger.info(f"   - 最快处理时间: {min(times):.3f}秒")
                        logger.info(f"   - 最慢处理时间: {max(times):.3f}秒")
            # 注册程序退出时的统计输出
            atexit.register(print_processing_stats)
        except Exception as e:
            logger.error(f"❌ image_detection模块测试失败: {e}")
    else:
        logger.error("❌ 没有可用的检测模块")

    # 检查端口配置
    port = int(os.environ.get('PORT', 8080))

    # 启动Flask应用
    logger.info(f"Starting Image Detection API server on port {port}...")
    logger.info(f"Image Detection Module: {'✅' if IMAGE_DETECTION_AVAILABLE else '❌'}")
    logger.info(f"MQ Publisher Module: {'✅' if MQ_PUBLISHER_AVAILABLE else '❌'}")

    # API行为说明
    logger.info("📡 API行为说明:")
    logger.info("   - POST /api/v1/detect/image: 默认仅发送结果到MQ队列")

    if MQ_PUBLISHER_AVAILABLE:
        logger.info("🔄 图片检测结果将发送到MQ队列: analyzeImageQueue")
        logger.info("📡 使用路由键: analyzeImageQueue.routing.key")
    else:
        logger.warning("⚠️ MQ发布器不可用，检测结果仅通过API返回")

    # ==========================================
    # 手动测试模式 (修改此处以切换模式)
    # ==========================================
    TEST_MODE = True
    TEST_IMAGE_PATH = r"uploads\DJI_20250812103257_0003_2_T.jpg"
    
    if TEST_MODE:
        logger.info("🔧 启动手动测试模式 (MQ发送已禁用)")
        
        if not os.path.exists(TEST_IMAGE_PATH):
            logger.error(f"❌ 测试图片不存在: {TEST_IMAGE_PATH}")
        else:
            try:
                # 1. 运行检测
                logger.info(f"📸 开始检测文件: {TEST_IMAGE_PATH}")
                # 可以在这里指定 output_dir，默认 './'
                detection_result = td.detect_targets(TEST_IMAGE_PATH, "./")
                
                # 2. 可视化
                img = imread_unicode(TEST_IMAGE_PATH)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    vis_image = vis.visualize_detections(img_rgb, 'video', detection_result)

                    result_path = "manual_test_result.jpg"
                    vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
                    imwrite_unicode(result_path, vis_image_bgr)
                    detection_result['data']['result_image_path'] = os.path.abspath(result_path)
                    logger.info(f"💾 可视化结果已保存至: {os.path.abspath(result_path)}")

                # 3. 打印检测结果（在可视化保存之后，路径已填充）
                logger.info("-" * 30)
                import json
                print("检测结果 JSON:")
                print(json.dumps(detection_result, indent=4, ensure_ascii=False))
                logger.info("-" * 30)

                logger.info("✅ 手动测试完成")
                
            except Exception as e:
                logger.error(f"❌ 手动测试失败: {e}")
                import traceback
                traceback.print_exc()
    else:
        try:
            logger.info(f"🚀 正在启动Flask应用，监听 0.0.0.0:{port}")
            app.run(
                host='0.0.0.0',
                port=port,
                debug=False,
                threaded=True,
                use_reloader=False
            )
        except Exception as e:
            logger.error(f"❌ Flask应用启动失败: {e}")
            import traceback
    
            traceback.print_exc()
