import os
import shutil
import datetime

from ..config import Config

def is_target_file(filename):
    """
    判断是否为需要处理的目标文件（支持_T红外、_S/_V可见光）
    
    参数:
        filename (str): 文件名
        
    返回:
        bool: 如果是目标文件返回True，否则返回False
    """
    # 获取文件扩展名
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    
    # 检查扩展名是否在支持的文件类型中
    is_supported_ext = ext in {'.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi', '.mov'}
    
    # 检查文件名是否符合目标模式
    # 支持以"_T"（红外）、"_S"（可见光静态）、"_V"（可见光视频）结尾的文件
    name_only = os.path.splitext(filename)[0]
    is_target_pattern = (name_only.endswith("_T") or 
                        name_only.endswith("_S") or
                        name_only.endswith("_V"))
    
    # 返回结果
    return is_supported_ext and is_target_pattern

def get_file_type(filename):
    """
    获取文件类型（红外或可见光）
    
    参数:
        filename (str): 文件名
        
    返回:
        str: 文件类型（'infrared'红外, 'visible'可见光, 'unknown'未知）
    """
    name_only = os.path.splitext(filename)[0]
    if name_only.endswith("_T"):
        return "infrared"
    elif name_only.endswith("_S") or  name_only.endswith("_V"):
        return "visible"
    else:
        return "unknown"

def ensure_dir(path):
    """
    确保目录存在，如果不存在则创建
    
    参数:
        path (str): 目录路径
    """
    if path and not os.path.exists(path):
        try:
            os.makedirs(path)
            print(f"创建目录: {path}")
        except Exception as e:
            print(f"创建目录失败 {path}: {e}")

def generate_output_path(input_path, output_dir, suffix):
    """
    生成输出文件路径
    
    参数:
        input_path (str): 输入文件路径
        output_dir (str): 输出目录
        suffix (str): 文件名后缀
        
    返回:
        str: 输出文件路径
    """
    # 获取输入文件名（不含扩展名）
    input_filename = os.path.basename(input_path)
    name, ext = os.path.splitext(input_filename)
    
    # 处理特定情况：如果文件名以"_T"结尾，去掉"_T"
    if name.endswith("_T") or name.endswith("_t"):
        name = name[:-2]
    
    # 生成输出文件名
    output_filename = f"{name}_{suffix}{ext}"
    
    # 组合输出路径
    output_path = os.path.join(output_dir, output_filename)
    
    return output_path

def get_processed_files(output_dir):
    """
    获取已处理的文件列表
    
    参数:
        output_dir (str): 输出目录
        
    返回:
        list: 已处理文件列表
    """
    processed_files = []
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.endswith("_result.jpg") or filename.endswith("_result.mp4"):
                processed_files.append(filename)
    return processed_files

def create_log_file(log_dir="logs", prefix="process_log"):
    """
    创建日志文件
    
    参数:
        log_dir (str): 日志目录
        prefix (str): 日志文件名前缀
        
    返回:
        tuple: (日志文件路径, 文件对象)
    """
    ensure_dir(log_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{prefix}_{timestamp}.txt"
    log_path = os.path.join(log_dir, log_filename)
    
    try:
        log_file = open(log_path, "w", encoding="utf-8")
        log_file.write(f"处理日志 - {timestamp}\n")
        log_file.write("=" * 50 + "\n")
        return log_path, log_file
    except Exception as e:
        print(f"创建日志文件失败: {log_path}: {e}")
        return None, None

def get_file_size(file_path):
    """
    获取文件大小（人类可读格式）
    
    参数:
        file_path (str): 文件路径
        
    返回:
        str: 文件大小字符串（如 "2.5 MB"）
    """
    try:
        size_bytes = os.path.getsize(file_path)
        # 转换为更友好的格式
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} GB"
    except:
        return "未知大小"

def backup_config(output_dir):
    """
    备份配置文件到输出目录
    
    参数:
        output_dir (str): 输出目录
    """
    # 假设配置文件名为 "config.py"
    config_path = "config.py"
    if os.path.exists(config_path):
        backup_dir = os.path.join(output_dir, "config_backups")
        ensure_dir(backup_dir)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"config_{timestamp}.py")
        
        try:
            shutil.copy2(config_path, backup_path)
            print(f"配置文件已备份到: {backup_path}")
        except Exception as e:
            print(f"配置文件备份失败: {e}")

def clean_output_dir(output_dir, max_files=100):
    """
    清理输出目录，保留最近的文件
    
    参数:
        output_dir (str): 输出目录
        max_files (int): 最大保留文件数
    """
    if not os.path.exists(output_dir):
        return
    
    # 获取所有文件及其修改时间
    files = []
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if os.path.isfile(file_path):
            mtime = os.path.getmtime(file_path)
            files.append((file_path, mtime))
    
    # 如果文件数量不超过限制，直接返回
    if len(files) <= max_files:
        return
    
    # 按修改时间排序（旧文件在前）
    files.sort(key=lambda x: x[1])
    
    # 删除多余的文件
    for i in range(len(files) - max_files):
        try:
            os.remove(files[i][0])
            print(f"清理旧文件: {files[i][0]}")
        except Exception as e:
            print(f"删除文件失败 {files[i][0]}: {e}")

def get_target_files(input_folder):
    """
    获取文件夹中所有符合条件的目标文件
    
    参数:
        input_folder (str): 输入文件夹路径
        
    返回:
        list: 目标文件路径列表
    """
    target_files = []
    
    if not os.path.exists(input_folder):
        return target_files
    
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        # 只处理文件，跳过目录
        if os.path.isfile(file_path) and is_target_file(filename):
            target_files.append(file_path)
    
    # 按文件名排序
    target_files.sort()
    return target_files