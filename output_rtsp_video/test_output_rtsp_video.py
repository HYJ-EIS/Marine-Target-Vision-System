import cv2
import sys
import time
import requests

from video_output_manager import create_video_output_manager
from stream_performance_monitor import create_performance_monitor

def init_video_output_manager(frame, input_fps):
    """初始化视频输出管理器，需要传入第一帧来确定视频尺寸"""
    # 创建输出管理器，使用输入流的实际帧率
    output_fps = max(int(input_fps), 15)  # 输出流帧率至少15fps，避免太低的帧率
    output_manager = create_video_output_manager(
        rtsp_url="rtsp://localhost:8554/demo_class",
        video_save_path="results/demo_class_output.mp4",
        fps=output_fps,  # 使用实际帧率
        enable_rtsp=True,
        enable_save=True
    )
    
    # 设置帧尺寸
    height, width = frame.shape[:2]
    output_manager.frame_width = width
    output_manager.frame_height = height
    
    if output_manager.initialize():
        print(f"✅ VideoOutputManager初始化成功 (输出FPS: {output_fps})")
        return output_manager
    else:
        print("❌ VideoOutputManager初始化失败")
        return None

def is_valid_stream(stream_url):
    """检查视频流地址是否有效"""
    try:
        # 对于RTSP流
        if stream_url.startswith('rtsp://'):
            cap = cv2.VideoCapture(stream_url)
            if cap.isOpened():
                cap.release()
                return True
            return False
        
        # 对于HTTP流，尝试发送HEAD请求
        elif stream_url.startswith(('http://', 'https://')):
            try:
                response = requests.head(stream_url, timeout=5)
                return response.status_code == 200
            except:
                # 如果HEAD请求失败，尝试GET请求
                try:
                    response = requests.get(stream_url, timeout=5, stream=True)
                    return response.status_code == 200
                except:
                    return False
        
        return False
    except:
        return False

def annotate_stream_frames(stream_url, reconnect_delay=5):
    """
    读取视频流并在每一帧上标注当前帧数
    
    参数:
        stream_url: 视频流地址 (RTSP, HTTP等)
        output_path: 输出视频路径(可选)
        reconnect_delay: 重连延迟时间(秒)
    """

    # 创建性能监控器
    performance_monitor = create_performance_monitor()
    performance_monitor.start_monitoring()

    print(f"尝试连接视频流: {stream_url}")
    
    # 验证流地址
    if not is_valid_stream(stream_url):
        print(f"错误：无法访问视频流 {stream_url}")
        print("请检查流地址是否正确且可访问")
        return False
    
    # 打开视频流
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        print(f"错误：无法打开视频流 {stream_url}")
        return False
    
    print("成功连接到视频流")
    
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频流信息: {width}x{height}, {fps:.2f} FPS")
    
    frame_count = 0
    last_frame_time = time.time()
    frame_interval = 1.0 / fps if fps > 0 else 0.04  # 默认25fps
    output_manager = None  # 延迟初始化
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("获取帧失败，尝试重新连接...")
                cap.release()
                time.sleep(reconnect_delay)
                
                # 尝试重新连接
                cap = cv2.VideoCapture(stream_url)
                if not cap.isOpened():
                    print("重新连接失败")
                    break
                print("重新连接成功")
                continue
                
            # 第一帧时初始化输出管理器
            if output_manager is None:
                output_manager = init_video_output_manager(frame, fps)
                if output_manager is None:
                    print("❌ 无法初始化输出管理器，退出测试")
                    break
                
                # 显示初始状态
                status = output_manager.get_status()
                print(f"📊 初始状态: {status}")
                
            # 自适应帧率控制 - 减少不必要的延迟
            current_time = time.time()
            elapsed = current_time - last_frame_time
            
            # 只有当处理速度太快时才添加延迟
            if elapsed < frame_interval * 0.8:  # 允许20%的缓冲
                sleep_time = frame_interval - elapsed
                if sleep_time > 0.001:  # 只有当需要延迟超过1ms时才睡眠
                    time.sleep(sleep_time)
            last_frame_time = time.time()
            
            # 在帧上标注帧数和时间戳
            cv2.putText(frame, f"Frame: {frame_count}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            cv2.putText(frame, f"Time: {timestamp}", (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # # 显示帧
            # cv2.imshow('Video Stream with Frame Annotation', frame)

            # 记录写入性能
            write_start_time = time.time()
            success = output_manager.write_frame(frame)
            write_time = time.time() - write_start_time
            
            # 记录到性能监控器
            performance_monitor.record_frame_processing(write_time, success)
            
            if success:
                # 显示处理进度
                if frame_count % 30 == 0:
                    print(f"已处理 {frame_count} 帧")
                    
                    # 每30帧显示一次性能统计
                    if frame_count > 0 and frame_count % 150 == 0:  # 每150帧显示详细统计
                        stats = performance_monitor.get_performance_stats()
                        if "actual_fps" in stats:
                            print(f"📈 性能统计: 实际FPS={stats['actual_fps']:.2f}, "
                                  f"成功率={stats['success_rate']:.2%}, "
                                  f"平均写入时间={stats.get('avg_write_time', 0)*1000:.2f}ms")
            else:
                print(f"⚠️ 第 {frame_count} 帧写入失败")
            
            frame_count += 1
            
            # 按'q'键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # 限制测试帧数，避免无限运行
            if frame_count >= 300:  # 约20秒的视频（15fps）
                print(f"达到测试帧数限制 ({frame_count} 帧)，结束测试")
                break
    
    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
    finally:
        # 清理资源
        cap.release()
        cv2.destroyAllWindows()
        
        if output_manager:
            # 最终状态
            final_status = output_manager.get_status()
            print(f"📊 最终状态: {final_status}")
            
            # 停止管理器
            output_manager.stop()
            print("✅ VideoOutputManager演示完成")
        
        # 停止性能监控并显示报告
        performance_monitor.stop_monitoring_thread()
        performance_monitor.print_performance_report()
    
    print(f"处理完成! 共处理 {frame_count} 帧")
    return True

if __name__ == "__main__":
    # 使用方法示例
    
    stream_url = "rtsp://localhost:8554/input_stream"
    
    # 执行视频流帧标注
    annotate_stream_frames(stream_url)