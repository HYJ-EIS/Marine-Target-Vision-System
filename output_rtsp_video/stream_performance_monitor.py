#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频流性能监控器

用于监控和诊断视频流输出性能，帮助识别卡顿原因
"""

import time
import threading
import queue
import logging
from collections import deque
from typing import Dict, List, Optional
import psutil
import numpy as np

class StreamPerformanceMonitor:
    """视频流性能监控器"""
    
    def __init__(self, window_size: int = 100):
        """
        初始化性能监控器
        
        Args:
            window_size: 性能统计窗口大小（帧数）
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.write_times = deque(maxlen=window_size)
        self.success_count = 0
        self.failure_count = 0
        self.total_frames = 0
        
        # 系统资源监控
        self.cpu_usage = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        
        # 延迟统计
        self.frame_intervals = deque(maxlen=window_size)
        self.processing_delays = deque(maxlen=window_size)
        
        # 监控线程
        self.monitor_thread = None
        self.stop_monitoring = False
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
    def start_monitoring(self):
        """开始性能监控"""
        self.stop_monitoring = False
        self.monitor_thread = threading.Thread(target=self._monitor_system_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring_thread(self):
        """停止性能监控"""
        self.stop_monitoring = True
        if self.monitor_thread:
            self.monitor_thread.join()
            
    def _monitor_system_resources(self):
        """监控系统资源使用情况"""
        while not self.stop_monitoring:
            try:
                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.cpu_usage.append(cpu_percent)
                
                # 内存使用率
                memory = psutil.virtual_memory()
                self.memory_usage.append(memory.percent)
                
            except Exception as e:
                self.logger.warning(f"系统资源监控错误: {e}")
                
            time.sleep(0.5)
    
    def record_frame_processing(self, processing_time: float, write_success: bool):
        """
        记录帧处理性能
        
        Args:
            processing_time: 处理时间（秒）
            write_success: 写入是否成功
        """
        current_time = time.time()
        
        # 记录处理时间
        self.write_times.append(processing_time)
        
        # 记录帧间隔
        if len(self.frame_times) > 0:
            interval = current_time - self.frame_times[-1]
            self.frame_intervals.append(interval)
            
        self.frame_times.append(current_time)
        
        # 统计成功/失败
        if write_success:
            self.success_count += 1
        else:
            self.failure_count += 1
            
        self.total_frames += 1
        
    def get_performance_stats(self) -> Dict:
        """获取性能统计信息"""
        if len(self.frame_times) < 2:
            return {"status": "insufficient_data"}
            
        stats = {
            "total_frames": self.total_frames,
            "success_rate": self.success_count / self.total_frames if self.total_frames > 0 else 0,
            "failure_count": self.failure_count,
        }
        
        # 帧率统计
        if len(self.frame_intervals) > 0:
            intervals = list(self.frame_intervals)
            avg_interval = np.mean(intervals)
            stats["actual_fps"] = 1.0 / avg_interval if avg_interval > 0 else 0
            stats["fps_variance"] = np.var(intervals)
            stats["fps_std"] = np.std(intervals)
            
        # 处理时间统计
        if len(self.write_times) > 0:
            write_times = list(self.write_times)
            stats["avg_write_time"] = np.mean(write_times)
            stats["max_write_time"] = np.max(write_times)
            stats["write_time_std"] = np.std(write_times)
            
        # 系统资源统计
        if len(self.cpu_usage) > 0:
            stats["avg_cpu_usage"] = np.mean(list(self.cpu_usage))
            stats["max_cpu_usage"] = np.max(list(self.cpu_usage))
            
        if len(self.memory_usage) > 0:
            stats["avg_memory_usage"] = np.mean(list(self.memory_usage))
            stats["max_memory_usage"] = np.max(list(self.memory_usage))
            
        return stats
    
    def detect_performance_issues(self) -> List[str]:
        """检测性能问题"""
        issues = []
        stats = self.get_performance_stats()
        
        if "actual_fps" not in stats:
            return ["数据不足，无法分析"]
            
        # 检测帧率问题
        if stats.get("fps_variance", 0) > 0.01:  # 帧率方差过大
            issues.append("⚠️ 帧率不稳定，可能导致卡顿")
            
        if stats.get("fps_std", 0) > 5:  # 帧率标准差过大
            issues.append("⚠️ 帧率波动过大")
            
        # 检测处理延迟问题
        if stats.get("max_write_time", 0) > 0.1:  # 写入时间超过100ms
            issues.append("⚠️ 存在写入延迟过大的情况")
            
        if stats.get("write_time_std", 0) > 0.05:  # 写入时间标准差过大
            issues.append("⚠️ 写入时间不稳定")
            
        # 检测系统资源问题
        if stats.get("avg_cpu_usage", 0) > 80:
            issues.append("⚠️ CPU使用率过高")
            
        if stats.get("avg_memory_usage", 0) > 90:
            issues.append("⚠️ 内存使用率过高")
            
        # 检测成功率问题
        if stats.get("success_rate", 1) < 0.95:
            issues.append("⚠️ 写入成功率过低")
            
        return issues if issues else ["✅ 未检测到明显的性能问题"]
    
    def print_performance_report(self):
        """打印性能报告"""
        stats = self.get_performance_stats()
        issues = self.detect_performance_issues()
        
        print("\n" + "="*50)
        print("📊 视频流性能报告")
        print("="*50)
        
        if "actual_fps" in stats:
            print(f"总帧数: {stats['total_frames']}")
            print(f"成功率: {stats['success_rate']:.2%}")
            print(f"失败帧数: {stats['failure_count']}")
            print(f"实际帧率: {stats['actual_fps']:.2f} fps")
            print(f"帧率标准差: {stats.get('fps_std', 0):.3f}")
            print(f"平均写入时间: {stats.get('avg_write_time', 0)*1000:.2f} ms")
            print(f"最大写入时间: {stats.get('max_write_time', 0)*1000:.2f} ms")
            
            if 'avg_cpu_usage' in stats:
                print(f"平均CPU使用率: {stats['avg_cpu_usage']:.1f}%")
                print(f"最大CPU使用率: {stats['max_cpu_usage']:.1f}%")
                
            if 'avg_memory_usage' in stats:
                print(f"平均内存使用率: {stats['avg_memory_usage']:.1f}%")
        
        print("\n🔍 性能问题检测:")
        for issue in issues:
            print(f"  {issue}")
            
        print("="*50)


def create_performance_monitor() -> StreamPerformanceMonitor:
    """创建性能监控器的便捷函数"""
    return StreamPerformanceMonitor()


if __name__ == "__main__":
    # 测试示例
    monitor = create_performance_monitor()
    monitor.start_monitoring()
    
    try:
        # 模拟帧处理
        for i in range(100):
            start_time = time.time()
            time.sleep(0.04)  # 模拟处理时间
            processing_time = time.time() - start_time
            
            # 模拟偶尔的写入失败
            success = True if i % 20 != 0 else False
            monitor.record_frame_processing(processing_time, success)
            
        # 打印报告
        monitor.print_performance_report()
        
    finally:
        monitor.stop_monitoring_thread()
