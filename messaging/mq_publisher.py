#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的RabbitMQ消息发布器
用于图片和视频检测结果的消息发布
"""

import pika
from pika.exceptions import AMQPConnectionError, ChannelClosedByBroker, ConnectionClosed
import json
import time
import logging
import threading

# 导入配置管理器
try:
    from .mq_config import get_mq_config
except ImportError:
    try:
        from mq_config import get_mq_config
    except ImportError:
        print("⚠️ 无法导入配置管理器")
        get_mq_config = None

logger = logging.getLogger(__name__)


class UnifiedMQPublisher:
    """统一的MQ消息发布器"""

    def __init__(self, host=None, port=None, username=None, password=None):
        """
        初始化MQ连接参数
        
        Args:
            host (str): MQ服务器地址，如果为None则从配置文件读取
            port (int): MQ服务端口，如果为None则从配置文件读取
            username (str): 用户名，如果为None则从配置文件读取
            password (str): 密码，如果为None则从配置文件读取
        """
        # 从配置文件加载设置
        if get_mq_config:
            config = get_mq_config()
            rabbitmq_config = config.get_rabbitmq_config()
            image_config = config.get_image_queue_config()
            video_usv_config = config.get_video_queue_config()  # 无人船
            video_uav_config = config.get_video_uav_queue_config()  # 无人机
            
            self.host = host or rabbitmq_config['host']
            self.port = port or rabbitmq_config['port']
            self.username = username or rabbitmq_config['username']
            self.password = password or rabbitmq_config['password']
            self.exchange = rabbitmq_config['exchange']
            
            # 队列配置
            self.image_queue = image_config['queue']
            self.image_routing_key = image_config['routing_key']
            self.video_usv_queue = video_usv_config['queue']
            self.video_usv_routing_key = video_usv_config['routing_key']
            self.video_uav_queue = video_uav_config['queue']
            self.video_uav_routing_key = video_uav_config['routing_key']

        self.credentials = pika.PlainCredentials(self.username, self.password)
        self.connection = None
        self.channel = None

    def connect(self):
        """建立到RabbitMQ的连接"""
        try:
            # 先关闭现有连接
            self._close_connection()
            
            connection_params = pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                credentials=self.credentials,
                heartbeat=60,  # 心跳检测时间(秒)
                blocked_connection_timeout=5,  # 减少阻塞连接超时
                socket_timeout=5,  # 减少套接字超时
                retry_delay=2,  # 增加重试延迟
                connection_attempts=1  # 减少连接重试次数
            )
            self.connection = pika.BlockingConnection(connection_params)
            self.channel = self.connection.channel()
            logger.info(f"成功连接到RabbitMQ服务器 {self.host}:{self.port}")

            logger.info(f"已配置交换机: {self.exchange}")
            logger.info(f"图片队列: {self.image_queue}, 路由键: {self.image_routing_key}")
            logger.info(f"无人船视频队列: {self.video_usv_queue}, 路由键: {self.video_usv_routing_key}")
            logger.info(f"无人机视频队列: {self.video_uav_queue}, 路由键: {self.video_uav_routing_key}")
            return True

        except AMQPConnectionError as e:
            logger.error(f"MQ连接失败: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"MQ连接过程中发生未知错误: {str(e)}")
            return False

    def _close_connection(self):
        """关闭现有连接"""
        try:
            if self.channel and not self.channel.is_closed:
                self.channel.close()
            if self.connection and not self.connection.is_closed:
                self.connection.close()
        except Exception as e:
            logger.warning(f"关闭连接时发生错误: {e}")
        finally:
            self.channel = None
            self.connection = None

    def _ensure_connection(self):
        """确保连接可用"""
        if not self.connection or self.connection.is_closed:
            logger.info("连接已断开，正在重新连接...")
            return self.connect()
        
        if not self.channel or self.channel.is_closed:
            logger.info("通道已关闭，正在重新创建通道...")
            try:
                self.channel = self.connection.channel()
                return True
            except Exception as e:
                logger.error(f"重新创建通道失败: {e}")
                return self.connect()
        
        return True

    def _convert_detection_result_format(self, detection_result):
        """
        转换检测结果格式，从输入格式转换为MQ输出格式
        
        Args:
            detection_result (dict): 输入的检测结果
            
        Returns:
            dict: 转换后的检测结果
        """
        converted_result = {
            "data": {
                "boxes": [],
                "count": detection_result.get("data", {}).get("count", 0),
                "result_image_path": detection_result.get("data", {}).get("result_image_path", "")
            },
            "message": detection_result.get("message", "Success"),
            "success": detection_result.get("success") == "true" or detection_result.get("success") is True,
            "timestamp": detection_result.get("timestamp", int(time.time() * 1000)),
            "type": detection_result.get("type", "infrared")
        }
        
        # 转换boxes格式
        input_boxes = detection_result.get("data", {}).get("boxes", [])
        for i, box in enumerate(input_boxes):
            converted_box = {
                "confidence": box.get("confidence", 0.0),
                "h": box.get("h", 0),
                "id": i + 1,  # 添加id字段，从1开始递增
                "w": box.get("w", 0),
                "x": box.get("x", 0),
                "y": box.get("y", 0)
            }
            # 不再包含class和class_confidence字段
            converted_result["data"]["boxes"].append(converted_box)
        
        return converted_result

    def _send_detection_result_to_queue(self, detection_result, routing_key, queue_name, message_type="检测结果"):
        """
        通用的检测结果发送方法
        
        Args:
            detection_result (dict): 检测结果数据
            routing_key (str): 路由键
            queue_name (str): 队列名称
            message_type (str): 消息类型（用于日志记录）
            
        Returns:
            bool: 发送是否成功
        """
        max_retries = 1  # 减少重试次数
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 确保连接可用
                if not self._ensure_connection():
                    logger.error("无法建立MQ连接")
                    retry_count += 1
                    time.sleep(1)
                    continue

                # 转换检测结果格式
                converted_result = self._convert_detection_result_format(detection_result)
                message_json = json.dumps(converted_result, ensure_ascii=False)

                # 发布消息到指定队列
                self.channel.basic_publish(
                    exchange=self.exchange,
                    routing_key=routing_key,
                    body=message_json.encode('utf-8'),
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # 使消息持久化
                        content_type='application/json'
                    )
                )

                logger.info(f"{message_type}已发送到MQ队列 '{queue_name}'")
                logger.debug(f"消息内容: {message_json}")
                return True

            except ChannelClosedByBroker as e:
                logger.warning(f"通道被代理关闭 - {str(e)}，正在重试... ({retry_count + 1}/{max_retries})")
                self.channel = None
                retry_count += 1
                time.sleep(1)
                
            except pika.exceptions.ConnectionClosed as e:
                logger.warning(f"连接被关闭 - {str(e)}，正在重试... ({retry_count + 1}/{max_retries})")
                self.connection = None
                self.channel = None
                retry_count += 1
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"发送{message_type}失败: {str(e)}，正在重试... ({retry_count + 1}/{max_retries})")
                retry_count += 1
                time.sleep(1)
        
        logger.error(f"发送{message_type}最终失败，已重试 {max_retries} 次")
        return False

    def send_image_detection_result(self, detection_result):
        """
        发送图片检测结果到MQ
        
        Args:
            detection_result (dict): 检测结果数据
            image_info (dict): 图片信息（可选）
        """
        # 转换检测结果格式并打印消息
        converted_result = self._convert_detection_result_format(detection_result)
        message_json = json.dumps(converted_result, ensure_ascii=False)
        print("发送到MQ的消息为：")
        print(message_json)
        
        # 使用通用方法发送消息
        return self._send_detection_result_to_queue(
            detection_result, 
            self.image_routing_key, 
            self.image_queue, 
            "图片检测结果"
        )

    def send_video_detection_result(self, detection_result, equipment_type="usv"):
        """
        发送视频检测结果到MQ
        
        Args:
            detection_result (dict): 检测结果数据
            equipment_type (str): 设备类型，"uav"表示无人机，其他表示无人船
        """
        # 转换检测结果格式并打印消息
        converted_result = self._convert_detection_result_format(detection_result)
        message_json = json.dumps(converted_result, ensure_ascii=False)
        print("输出到视频队列的消息：")
        print(message_json)

        # 根据设备类型选择路由键和队列名
        if equipment_type == "uav":
            routing_key = self.video_uav_routing_key
            queue_name = self.video_uav_queue
        else:
            routing_key = self.video_usv_routing_key
            queue_name = self.video_usv_queue
        
        print(f"📤 发送视频检测结果到MQ:")
        print(f"   设备类型: {equipment_type}")
        print(f"   目标队列: {queue_name}")
        print(f"   路由键: {routing_key}")
        print(f"   交换机: {self.exchange}")

        # 使用通用方法发送消息
        return self._send_detection_result_to_queue(
            detection_result, 
            routing_key, 
            queue_name, 
            "视频检测结果"
        )

    def close(self):
        """关闭RabbitMQ连接"""
        try:
            self._close_connection()
            logger.info("RabbitMQ连接已关闭")
        except Exception as e:
            logger.error(f"关闭MQ连接时发生错误: {e}")


# 全局MQ发布器实例（单例模式）
_mq_publisher = None
_publisher_lock = threading.Lock()


def get_mq_publisher():
    """获取MQ发布器实例（单例模式，线程安全）"""
    global _mq_publisher
    if _mq_publisher is None:
        with _publisher_lock:
            if _mq_publisher is None:  # 双重检查
                _mq_publisher = UnifiedMQPublisher()
                # 预先连接
                _mq_publisher.connect()
    return _mq_publisher


# 便捷函数
def send_image_result_to_mq(detection_result):
    """发送图片检测结果到MQ的便捷函数"""
    publisher = get_mq_publisher()
    return publisher.send_image_detection_result(detection_result)


def send_video_result_to_mq(detection_result):
    """发送视频检测结果到MQ的便捷函数"""
    publisher = get_mq_publisher()
    return publisher.send_video_detection_result(detection_result)


if __name__ == "__main__":
    # 测试MQ发布器
    import logging
    logging.basicConfig(level=logging.INFO)
    
    publisher = get_mq_publisher()
    
    # 测试图片检测结果发送
    test_image_result = {
        "data": {
            "boxes": [
                {
                    "id": 1,                 # 检测目标唯一ID
                    "confidence": 0.95,      # 置信度
                    "h": 19,                 # 标记矩形高度
                    "w": 43,                 # 标记矩形宽度
                    "x": 676,                # 标记矩形左上角像素坐标x
                    "y": 497,                # 标记矩形左上角像素坐标y
                    "class": "USV",          # 可选：目标分类（转换时会被移除）
                    "class_confidence": 0.8  # 可选：分类置信度（转换时会被移除）
                }
            ],
            "count": 1,                      # 识别到数量
            "result_image_path": "./results/result_uuid.jpg"  # 保存路径
        },
        "message": "Success",
        "success": "true",
        "timestamp": 1753774440304,
        "type": "infrared"                   # infrared：红外, light：可见光
    }

    
    print("测试发送图片检测结果...")
    if send_image_result_to_mq(test_image_result):
        print("✅ 图片检测结果发送成功")
    else:
        print("❌ 图片检测结果发送失败")
    
    publisher.close()
