#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MQ配置加载器
"""

import configparser
import os
import logging

logger = logging.getLogger(__name__)

class MQConfig:
    """MQ配置管理类"""
    
    def __init__(self):
        """初始化配置"""
        self.config = configparser.ConfigParser()
        self._set_default_config()

    def _set_default_config(self):
        """设置默认配置"""
        self.config['rabbitmq'] = {
            'host': '168.1.3.100',
            'port': '5672',
            'username': 'admin',
            'password': 'admin',
            'exchange': 'uavExchange'
        }
        
        self.config['image_analysis'] = {
            'queue': 'analyzeImageQueue',
            'routing_key': 'analyzeImageQueue.routing.key'
        }
        
        self.config['video_analysis1'] = {
            'queue': 'analyzeVideoQueue1',
            'routing_key': 'analyzeVideoQueueUsv.routing.key'
        }

        self.config['video_analysis2'] = {
            'queue': 'analyzeVideoQueue2',
            'routing_key': 'analyzeVideoQueueUav.routing.key'
        }
        
        self.config['api'] = {
            'return_detection_results': 'false',
            'max_file_size': '100',
            'allowed_extensions': 'png,jpg,jpeg,gif,bmp,tiff'
        }
        
        self.config['logging'] = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    
    def get_rabbitmq_config(self):
        """获取RabbitMQ连接配置"""
        return {
            'host': self.config.get('rabbitmq', 'host', fallback='168.1.3.100'),
            'port': self.config.getint('rabbitmq', 'port', fallback=5672),
            'username': self.config.get('rabbitmq', 'username', fallback='admin'),
            'password': self.config.get('rabbitmq', 'password', fallback='admin'),
            'exchange': self.config.get('rabbitmq', 'exchange', fallback='uavExchange')
        }
    
    def get_image_queue_config(self):
        """获取图片分析队列配置"""
        return {
            'queue': self.config.get('image_analysis', 'queue', fallback='analyzeImageQueue'),
            'routing_key': self.config.get('image_analysis', 'routing_key', fallback='analyzeImageQueue.routing.key')
        }
    
    def get_video_queue_config(self):
        """获取视频分析队列配置（无人船）"""
        return {
            'queue': self.config.get('video_analysis1', 'queue', fallback='analyzeVideoQueue1'),
            'routing_key': self.config.get('video_analysis1', 'routing_key', fallback='analyzeVideoQueueUsv.routing.key')
        }
    
    def get_video_uav_queue_config(self):
        """获取视频分析队列配置（无人机）"""
        return {
            'queue': self.config.get('video_analysis2', 'queue', fallback='analyzeVideoQueue2'),
            'routing_key': self.config.get('video_analysis2', 'routing_key', fallback='analyzeVideoQueueUav.routing.key')
        }
    
    def get_api_config(self):
        """获取API配置"""
        return {
            'return_detection_results': self.config.getboolean('api', 'return_detection_results', fallback=False),
            'max_file_size': self.config.getint('api', 'max_file_size', fallback=100),
            'allowed_extensions': self.config.get('api', 'allowed_extensions', fallback='png,jpg,jpeg,gif,bmp,tiff').split(',')
        }
    
    def should_return_results(self):
        """是否应该通过API返回检测结果"""
        return self.config.getboolean('api', 'return_detection_results', fallback=False)

# 全局配置实例
_mq_config = None

def get_mq_config():
    """获取MQ配置实例（单例模式）"""
    global _mq_config
    if _mq_config is None:
        _mq_config = MQConfig()
    return _mq_config

if __name__ == "__main__":
    # 测试配置加载
    import logging
    logging.basicConfig(level=logging.INFO)
    
    config = get_mq_config()
    
    print("RabbitMQ配置:", config.get_rabbitmq_config())
    print("图片队列配置:", config.get_image_queue_config())
    print("视频队列配置:", config.get_video_queue_config())
    print("API配置:", config.get_api_config())
    print("是否返回结果:", config.should_return_results())
