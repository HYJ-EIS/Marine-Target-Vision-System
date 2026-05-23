#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频输出管理器 - 推流和保存功能

从video_detect_module中剥离出来的推流和视频保存功能
支持：
1. RTSP推流输出
2. 本地视频文件保存
3. 高质量编码配置
4. 错误恢复机制
"""

import os
import sys
import time
import logging
import cv2
import numpy as np
import subprocess
import shutil
from typing import Optional, Dict, Any
from datetime import datetime


class VideoOutputManager:
    """视频输出管理器"""
    
    def __init__(self, 
                 rtsp_url: Optional[str] = None,
                 video_save_path: Optional[str] = None,
                 fps: int = 25,
                 enable_rtsp: bool = True,
                 enable_save: bool = True,
                 logger: Optional[logging.Logger] = None):
        """
        初始化视频输出管理器
        
        Args:
            rtsp_url: RTSP推流地址，如 "rtsp://localhost:8554/detection_output"
            video_save_path: 视频保存路径，如 "output.mp4"
            fps: 帧率
            enable_rtsp: 是否启用RTSP推流
            enable_save: 是否启用视频保存
            logger: 日志记录器
        """
        self.rtsp_url = rtsp_url
        self.video_save_path = video_save_path
        self.fps = fps
        self.enable_rtsp = enable_rtsp
        self.enable_save = enable_save
        self.logger = logger or self._setup_logger()
        
        # 输出组件
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.rtsp_writer: Optional[subprocess.Popen] = None
        
        # 状态标志
        self.is_initialized = False
        self.frame_width = 0
        self.frame_height = 0
        
        # 错误恢复相关
        self._rtsp_failure_count = 0
        self._max_rtsp_failures = 5
        
        self.logger.info("🎬 视频输出管理器已创建")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(f"{__name__}.VideoOutputManager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def initialize(self, init_frame) -> bool:
        """
        初始化输出流
        Returns:  bool: 初始化是否成功
        """
        try:
            self.frame_width = init_frame.shape[1]
            self.frame_height = init_frame.shape[0]
            success = True
            
            # 初始化视频文件保存
            if self.enable_save and self.video_save_path:
                success &= self._initialize_video_writer()
            
            # 初始化RTSP推流
            if self.enable_rtsp and self.rtsp_url:
                success &= self._initialize_rtsp_stream()
            
            self.is_initialized = success
            
            if success:
                self.logger.info("✅ 视频输出管理器初始化成功")
            else:
                self.logger.error("❌ 视频输出管理器初始化失败")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ 初始化视频输出管理器时出错: {e}")
            return False
    
    def _initialize_video_writer(self) -> bool:
        """初始化视频文件写入器"""
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(self.video_save_path), exist_ok=True)
            
            # 配置编码器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.video_save_path,
                fourcc,
                self.fps,
                (self.frame_width, self.frame_height)
            )
            
            if self.video_writer.isOpened():
                self.logger.info(f"✅ 视频文件写入器已初始化: {self.video_save_path}")
                return True
            else:
                self.logger.error(f"❌ 无法初始化视频文件写入器: {self.video_save_path}")
                self.video_writer = None
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 初始化视频文件写入器失败: {e}")
            return False
    
    def _initialize_rtsp_stream(self) -> bool:
        """初始化RTSP推流"""
        try:
            # 查找FFmpeg路径
            ffmpeg_path = self._find_ffmpeg()
            if not ffmpeg_path:
                self.logger.error("❌ 未找到FFmpeg，无法启用RTSP推流")
                return False
            
            # 构建FFmpeg命令
            ffmpeg_cmd = self._build_ffmpeg_command(ffmpeg_path)
            
            # 启动FFmpeg进程 - 增大缓冲区避免卡顿
            self.rtsp_writer = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                # stderr=subprocess.PIPE,
                bufsize=1024*1024  # 1MB缓冲区，避免管道阻塞
            )
            
            # 等待进程启动
            time.sleep(0.1)
            
            if self.rtsp_writer.poll() is None:
                self.logger.info(f"✅ RTSP推流已初始化: {self.rtsp_url}")
                self._rtsp_failure_count = 0
                return True
            else:
                stderr_output = self.rtsp_writer.stderr.read().decode('utf-8', errors='ignore')
                self.logger.error(f"❌ RTSP推流初始化失败: {stderr_output}")
                self.rtsp_writer = None
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 初始化RTSP推流失败: {e}")
            self.rtsp_writer = None
            return False
    
    def _find_ffmpeg(self) -> Optional[str]:
        """查找FFmpeg可执行文件"""
        # 首先在系统PATH中查找
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            return ffmpeg_path
        
        # 在模块目录下查找
        module_dir = os.path.dirname(__file__)
        
        # 检查ffmpeg/bin/目录
        ffmpeg_bin = os.path.join(module_dir, 'ffmpeg', 'bin', 'ffmpeg.exe')
        if os.path.exists(ffmpeg_bin):
            return ffmpeg_bin
        
        # 检查模块根目录
        local_ffmpeg = os.path.join(module_dir, 'ffmpeg.exe')
        if os.path.exists(local_ffmpeg):
            return local_ffmpeg
        
        return None
    
    def _build_ffmpeg_command(self, ffmpeg_path: str) -> list:
        """构建FFmpeg命令 - 优化版本减少卡顿"""
        return [
            ffmpeg_path,
            '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.frame_width}x{self.frame_height}',
            '-r', str(self.fps), '-i', '-',
            
            # 强化错误处理
            '-err_detect', 'ignore_err',
            '-fflags', '+igndts+ignidx+genpts+discardcorrupt',
            '-avoid_negative_ts', 'make_zero',
            '-max_error_rate', '0.9',  # 提高容错率
            
            # 优化的H.264编码配置 - 减少延迟
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',  # 改为ultrafast以减少编码延迟
            '-tune', 'zerolatency',
            '-profile:v', 'baseline',  # 改为baseline以获得更好的兼容性
            '-level', '3.1',
            
            # 减少关键帧间隔以提高流畅度
            '-g', '1',  # 每秒一个关键帧（假设15fps）
            '-keyint_min', '5',
            '-sc_threshold', '40',
            '-bf', '0',  # 不使用B帧，减少延迟
            
            # 质量设置 - 平衡质量和延迟
            '-crf', '28',  # 稍微降低质量以减少编码时间
            '-maxrate', '4000k',  # 降低最大码率
            '-bufsize', '80000k',   # 相应降低缓冲区大小
            
            # 编码优化 - 减少复杂度
            '-refs', '1',  # 减少参考帧
            '-me_method', 'dia',  # 使用简单的运动估计
            '-subq', '3',  # 降低子像素精度
            '-trellis', '0',  # 关闭trellis量化
            '-deblock', '0:0:0',  # 关闭去块滤波器
            
            # 减少延迟的同步设置
            '-vsync', 'passthrough',  # 改为passthrough模式
            '-copytb', '1',
            '-fflags', '+genpts',
            
            # RTSP输出优化
            '-f', 'rtsp',
            '-rtsp_transport', 'tcp',  # 改为TCP，更稳定
            '-flush_packets', '1',
            '-max_delay', '1000000',  # 100ms最大延迟
            '-muxdelay', '0.1',
            self.rtsp_url
        ]
    
    def write_frame(self, frame: np.ndarray) -> bool:
        """
        写入视频帧到输出流
        
        Args:
            frame: 要写入的视频帧
            
        Returns:
            bool: 写入是否成功
        """
        if not self.is_initialized:
            self.logger.warning("⚠️ 输出管理器未初始化，跳过帧写入")
            return False
        
        success = True
        
        # 写入视频文件
        if self.enable_save and self.video_writer:
            success &= self._write_to_file(frame)
        
        print("writed a frame")

        # 写入RTSP流
        if self.enable_rtsp and self.rtsp_writer:
            success &= self._write_to_rtsp(frame)
            print("push a frame to RTSP")
        return success
    
    def _write_to_file(self, frame: np.ndarray) -> bool:
        """写入视频文件"""
        try:
            if self.video_writer and self.video_writer.isOpened():
                self.video_writer.write(frame)
                return True
            else:
                self.logger.warning("⚠️ 视频文件写入器未打开")
                return False
        except Exception as e:
            self.logger.warning(f"⚠️ 视频文件写入失败: {e}")
            return False
    
    def _write_to_rtsp(self, frame: np.ndarray) -> bool:
        """写入RTSP流 - 优化版本减少卡顿"""
        try:
            # 检查FFmpeg进程状态
            if self.rtsp_writer.poll() is not None:
                self.logger.warning("⚠️ FFmpeg进程已停止，尝试重新启动RTSP推流")
                self._restart_rtsp_stream(frame)
                return False
            
            # 写入帧数据
            frame_bytes = frame.tobytes()
            
            try:
                print("write frame RTSP")
                self.rtsp_writer.stdin.write(frame_bytes)
                # 重要：立即刷新缓冲区，减少延迟
                print("flush RTSP")
                self.rtsp_writer.stdin.flush()
                return True
            except BlockingIOError:
                # 如果写入被阻塞，记录警告但不重启，避免频繁重启导致更严重的卡顿
                self.logger.debug("⚠️ RTSP写入暂时阻塞，跳过当前帧")
                return False
            except BrokenPipeError:
                # 管道破损，重启推流
                self.logger.warning("⚠️ RTSP推流管道破损，尝试重启")
                self._restart_rtsp_stream(frame)
                return False
                
        except Exception as e:
            self.logger.warning(f"⚠️ RTSP推流写入失败: {e}")
            return False
    
    def _restart_rtsp_stream(self, frame: np.ndarray):
        """重启RTSP推流"""
        try:
            self._rtsp_failure_count += 1
            
            if self._rtsp_failure_count > self._max_rtsp_failures:
                self.logger.error(f"❌ RTSP推流失败次数过多({self._rtsp_failure_count})，停止重试")
                self._cleanup_rtsp_stream()
                return
            
            self.logger.info(f"🔄 重启RTSP推流 (尝试 {self._rtsp_failure_count}/{self._max_rtsp_failures})")
            
            # 清理旧的推流
            self._cleanup_rtsp_stream()
            
            # 等待一下
            time.sleep(0.5)
            
            # 重新初始化
            if self._initialize_rtsp_stream():
                self.logger.info("✅ RTSP推流重启成功")
            else:
                self.logger.error("❌ RTSP推流重启失败")
                
        except Exception as e:
            self.logger.error(f"❌ 重启RTSP推流失败: {e}")
    
    def _cleanup_rtsp_stream(self):
        """清理RTSP推流资源"""
        try:
            if self.rtsp_writer:
                if self.rtsp_writer.stdin:
                    self.rtsp_writer.stdin.close()
                self.rtsp_writer.terminate()
                try:
                    self.rtsp_writer.wait(timeout=3)
                except:
                    self.rtsp_writer.kill()
                self.rtsp_writer = None
        except Exception as e:
            self.logger.debug(f"清理RTSP推流时出错: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取输出状态"""
        return {
            'initialized': self.is_initialized,
            'rtsp_enabled': self.enable_rtsp,
            'save_enabled': self.enable_save,
            'rtsp_url': self.rtsp_url,
            'video_save_path': self.video_save_path,
            'fps': self.fps,
            'frame_size': f"{self.frame_width}x{self.frame_height}",
            'video_writer_active': self.video_writer is not None and self.video_writer.isOpened() if self.video_writer else False,
            'rtsp_writer_active': self.rtsp_writer is not None,
            'rtsp_failure_count': self._rtsp_failure_count
        }
    
    def stop(self):
        """停止输出管理器"""
        self.logger.info("🛑 停止视频输出管理器...")
        
        # 关闭视频文件写入器
        if self.video_writer:
            try:
                self.video_writer.release()
                self.video_writer = None
                self.logger.info("📹 视频文件写入器已关闭")
            except Exception as e:
                self.logger.error(f"关闭视频文件写入器时出错: {e}")
        
        # 清理RTSP推流
        self._cleanup_rtsp_stream()
        if self.rtsp_writer is None:
            self.logger.info("📡 RTSP推流已关闭")
        
        self.is_initialized = False
        self.logger.info("⏹️ 视频输出管理器已停止")


def create_video_output_manager(rtsp_url: Optional[str] = None,
                               video_save_path: Optional[str] = None,
                               fps: int = 25,
                               enable_rtsp: bool = True,
                               enable_save: bool = True) -> VideoOutputManager:
    """
    创建视频输出管理器的便捷函数
    
    Args:
        rtsp_url: RTSP推流地址
        video_save_path: 视频保存路径
        fps: 帧率
        enable_rtsp: 是否启用RTSP推流
        enable_save: 是否启用视频保存
        
    Returns:
        VideoOutputManager: 视频输出管理器实例
    """
    return VideoOutputManager(
        rtsp_url=rtsp_url,
        video_save_path=video_save_path,
        fps=fps,
        enable_rtsp=enable_rtsp,
        enable_save=enable_save
    )


# 使用示例
if __name__ == "__main__":
    import time
    
    # 创建输出管理器
    output_manager = create_video_output_manager(
        rtsp_url="rtsp://localhost:8554/test_output",
        video_save_path="results/test_output.mp4",
        fps=25,
        enable_rtsp=True,
        enable_save=True
    )
    
    # 创建测试帧
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame[:] = (100, 150, 200)  # 填充颜色
    
    # 初始化
    if output_manager.initialize(test_frame):
        print("✅ 输出管理器初始化成功")
        
        # 写入一些测试帧
        for i in range(100):
            # 修改帧内容以显示变化
            cv2.putText(test_frame, f"Frame {i}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 写入帧
            output_manager.write_frame(test_frame)
            time.sleep(1/25)  # 模拟25fps
            
            if i % 25 == 0:
                print(f"已写入 {i} 帧")
        
        # 停止
        output_manager.stop()
        print("✅ 测试完成")
    else:
        print("❌ 输出管理器初始化失败")
