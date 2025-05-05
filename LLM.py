#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import torch
import time
import shutil
import logging
import argparse
import traceback
import threading
import subprocess
from pathlib import Path
from transformers import TextStreamer
import requests
from pydub import AudioSegment
from pydub.playback import play
import importlib.util
import platform
from transformers import AutoConfig, CLIPImageProcessor
from typing import Optional
import re
from concurrent.futures import ThreadPoolExecutor
import uuid
import colorama
from colorama import Fore, Back, Style
from transformers import StoppingCriteriaList, StoppingCriteria

# Initialize colorama for cross-platform colored terminal output
colorama.init(autoreset=True)

# 设置环境变量禁用pygame欢迎消息
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

# 添加pygame音频播放库
try:
    import pygame
    pygame.mixer.init(frequency=44100, buffer=4096)  # 增加缓冲区大小，减少IO问题
except ImportError:
    pass

# 多架构模型支持
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TextIteratorStreamer
    )
except ImportError:
    pass

try:
    from llama_cpp import Llama
except ImportError:
    pass

# 配置彩色日志
class ColoredFormatter(logging.Formatter):
    """自定义彩色日志格式化器"""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }
    
    def format(self, record):
        # 彩色级别名称
        level_name = record.levelname
        if level_name in self.COLORS:
            record.levelname = f"{self.COLORS[level_name]}{level_name}{Style.RESET_ALL}"
        
        # 彩色时间戳
        asctime = self.formatTime(record, self.datefmt)
        record.asctime = f"{Fore.BLUE}{asctime}{Style.RESET_ALL}"
        
        # 特殊标记某些关键词
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            if "加载成功" in record.msg:
                record.msg = record.msg.replace("加载成功", f"{Fore.GREEN}加载成功{Style.RESET_ALL}")
            elif "失败" in record.msg:
                record.msg = record.msg.replace("失败", f"{Fore.RED}失败{Style.RESET_ALL}")
            elif "错误" in record.msg:
                record.msg = record.msg.replace("错误", f"{Fore.RED}错误{Style.RESET_ALL}")
            elif "成功" in record.msg:
                record.msg = record.msg.replace("成功", f"{Fore.GREEN}成功{Style.RESET_ALL}")
        
        return super().format(record)

# 配置日志处理器
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))

# 配置日志
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# 移除默认处理器，避免重复日志
for hdlr in logger.handlers[:]:
    if isinstance(hdlr, logging.StreamHandler) and hdlr != handler:
        logger.removeHandler(hdlr)

# 定义音频播放临时目录 - 使用应用自身目录下的固定目录
AUDIO_TEMP_DIR = Path("./audio_temp")
if not AUDIO_TEMP_DIR.exists():
    AUDIO_TEMP_DIR.mkdir(parents=True, exist_ok=True)

# 安全音频目录 - 固定位置
SAFE_AUDIO_DIR = Path("./Audio/Safe_Audio")
if not SAFE_AUDIO_DIR.exists():
    SAFE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# 移除自动清理，改为单独脚本
def cleanup_temp_files(max_age_seconds=3600):
    """清理临时文件夹中的旧文件"""
    try:
        now = time.time()
        # 清理临时目录
        for file_path in AUDIO_TEMP_DIR.glob("*.*"):
            # 仅处理音频文件
            if file_path.suffix.lower() in ['.wav', '.mp3']:
                file_age = now - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        logging.info(f"已清理过期临时文件: {file_path.name}")
                    except Exception as e:
                        logging.warning(f"清理文件失败: {file_path} - {str(e)}")
        
        # 保留最新的安全音频文件，清理旧的
        safe_files = list(SAFE_AUDIO_DIR.glob("*.wav"))
        if len(safe_files) > 10:  # 保留最新的10个文件
            # 按修改时间排序
            safe_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            # 删除旧文件
            for old_file in safe_files[10:]:
                try:
                    old_file.unlink()
                    logging.info(f"已清理旧的安全音频文件: {old_file.name}")
                except Exception as e:
                    logging.warning(f"清理安全音频文件失败: {old_file} - {str(e)}")
    except Exception as e:
        logging.error(f"清理临时文件出错: {str(e)}")

def sanitize_path(path):
    """处理路径中的特殊字符，确保Windows路径有效"""
    if isinstance(path, str):
        path = Path(path)
    
    # 确保路径是绝对路径
    path = path.resolve()
    
    # 处理Windows路径限制
    if platform.system() == "Windows":
        # 路径长度限制
        if len(str(path)) > 240:
            try:
                # 创建短路径别名
                short_path = AUDIO_TEMP_DIR / f"short_{int(time.time())}.wav"
                shutil.copy2(path, short_path)
                return short_path
            except:
                # 复制失败则继续使用原路径
                pass
    
    return path

class ReverieAI:
    """Reverie AI核心类（支持多架构模型和自动内存管理）"""
    
    def __init__(self, model_path: str, use_gpu: bool = False, load_in_4bit: bool = False, 
                 use_tts: bool = False, spark_tts: bool = False, prompt_text: str = None,
                 prompt_speech_path: str = None, device: str = "0", save_dir: str = "Audio",
                 model_dir: str = None):
        # 初始化参数，统一将模型路径转换为Path对象，并同时记录字符串形式的绝对路径
        self.model_path = Path(model_path)
        self.model_path_str = str(self.model_path.resolve())
        self.use_gpu = use_gpu
        self.use_tts = use_tts
        self.spark_tts = spark_tts
        self.load_in_4bit = load_in_4bit
        self.prompt_text = prompt_text or "雨林里可以交给我的眷属们，城市里我就拜托一些小孩子吧。"
        self.prompt_speech_path = Path(prompt_speech_path or "models/tts/Nahida.wav")
        self.device_id = device
        self.save_dir = Path(save_dir)
        self.model_dir = Path(model_dir) if model_dir else None
        self.cuda_available = torch.cuda.is_available()
        self.device = "cuda" if (use_gpu and self.cuda_available) else "cpu"
        self.idle_timeout = 600  # 10分钟空闲超时
        
        # 多模态支持
        self.vision_enabled = False
        self.image_processor = None
        self.processor = None
        self.cache_dir = Path("cache")
        self.supported_image_ext = ['.jpg', '.jpeg', '.png', '.webp']
        
        # 线程安全
        self.lock = threading.Lock()
        self.model_loaded = False
        self.last_used = time.time()
        self.conversation_history = []
        self.max_history_length = 10  # 设置对话历史记录的最大长度（轮次数）

        # 防止死循环的设置
        self.max_generation_time = 2400  # 最长生成时间(秒)，增加到4分钟
        self.max_new_tokens = 20480  # 最大生成token数
        self.repetition_window = 32  # 用于检测重复的窗口大小
        self.repetition_threshold = 5  # 重复的次数阈值，超过则认为是死循环

        # 优化器链：用于加载模型后自动应用一系列优化
        self.model_optimizers = []

        # 默认注册优化器模块：根据标志注册4位量化和Flash Attention优化
        if self.load_in_4bit:
            self.register_model_optimizer(lambda x: x._optimize_quant_4bit())
        if self.use_gpu:
            self.register_model_optimizer(lambda x: x._optimize_flash_attention())
        # 注册BitNet优化器（会自动检查是否需要应用）
        self.register_model_optimizer(lambda x: x._optimize_bitnet())

        # 初始化流程
        self._detect_model_type()
        self._detect_vision_support()  # 检测视觉支持
        self._start_inactive_timer()
        self._clean_image_cache()
        
        # 不再自动清理临时文件，改为单独的clean.py脚本
        # 请使用 start-clean.bat 或直接运行 python clean.py 清理临时文件
        
        # GPU信息显示
        if self.cuda_available and self.use_gpu:
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_memory / (1024 ** 3)
            logging.info(f"检测到GPU，总显存：{total_gb:.2f}GB")

        logging.info(f"模型初始化完成，设备类型: {self.device.upper()}")

        # 确保保存目录存在
        self.save_dir.mkdir(parents=True, exist_ok=True)
        # 确保音频安全目录存在
        SAFE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    def _get_venv_python(self):
        """获取虚拟环境Python路径"""
        if platform.system() == "Windows":
            return str(Path("./venv/Scripts/python.exe").resolve())
        else:
            return str(Path("./venv/bin/python").resolve())

    def _detect_vision_support(self):
        """检测模型是否支持视觉输入"""
        try:
            # 首先尝试加载配置，确保设置trust_remote_code=True
            config = AutoConfig.from_pretrained(self.model_path_str, trust_remote_code=True)
            if hasattr(config, 'vision_config') or hasattr(config, 'vision_encoder'):
                self.vision_enabled = True
                logging.info("检测到视觉模型支持")
        except Exception as e:
            logging.warning(f"视觉支持检测失败: {str(e)}")
            logging.info("尝试以兼容模式继续加载")
            
    def _process_images(self) -> Optional[torch.Tensor]:
        """处理缓存目录中的图片"""
        if not self.cache_dir.exists():
            return None
        
        image_files = sorted([
            f for f in self.cache_dir.iterdir()
            if f.suffix.lower() in self.supported_image_ext
        ])
        
        if not image_files:
            return None
        
        try:
            from PIL import Image
            image = Image.open(image_files[0]).convert("RGB")
            
            if self.processor:
                return self.processor(images=image, return_tensors="pt")["pixel_values"].to(self.device)
            elif self.image_processor:
                return self.image_processor(image, return_tensors="pt")["pixel_values"].to(self.device)
            return None
        except Exception as e:
            logging.error(f"图像处理失败: {str(e)}")
            return None
        
    def _clean_image_cache(self):
        """清理图片缓存"""
        if self.cache_dir.exists():
            for f in self.cache_dir.glob("*"):
                try:
                    if f.suffix.lower() in self.supported_image_ext:
                        f.unlink()
                except:
                    pass

    def _detect_model_type(self):
        """智能检测模型架构"""
        try:
            # 使用字符串形式的模型路径
            model_path_str = self.model_path_str
            
            # 检查是否是GGUF格式
            if isinstance(self.model_path, Path):
                model_suffix = self.model_path.suffix.lower()
            else:
                model_suffix = os.path.splitext(model_path_str)[1].lower()
                
            if (model_suffix == ".gguf") or (model_suffix == ".ggml"):
                self.model_type = "gguf"
                logging.info("检测到GGUF模型格式")
                return

            # 检查config.json文件
            if isinstance(self.model_path, Path):
                config_path = self.model_path / "config.json"
            else:
                config_path = os.path.join(model_path_str, "config.json")
                
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        architecture = config.get("architectures", [""])[0].lower() if "architectures" in config else ""
                        model_type = config.get("model_type", "").lower()
                        
                        # 检测BitNet模型
                        if "bitnet" in architecture or "bitnet" in model_type:
                            self.model_type = "bitnet"
                            logging.info(f"{Fore.CYAN}检测到BitNet模型架构: {architecture}{Style.RESET_ALL}")
                            # BitNet模型强制设置trust_remote_code
                            logging.info(f"{Fore.YELLOW}BitNet模型需要设置trust_remote_code=True{Style.RESET_ALL}")
                            return
                        
                        # 检测模型类型
                        if any(x in architecture for x in ["llama", "mistral", "mixtral"]):
                            self.model_type = "llama"
                            logging.info(f"检测到LLAMA族模型架构: {architecture}")
                        elif any(x in architecture for x in ["moe", "qwen", "deepseek", "gpt-neox"]):
                            self.model_type = "moe"
                            logging.info(f"检测到MOE族模型架构: {architecture}")
                        elif "phi" in architecture:
                            self.model_type = "phi"
                            logging.info(f"检测到PHI族模型架构: {architecture}")
                        elif "gpt" in architecture:
                            self.model_type = "gpt"
                            logging.info(f"检测到GPT族模型架构: {architecture}")
                        else:
                            self.model_type = "hf"
                            logging.info(f"使用通用HF架构加载模型: {architecture}")
                except Exception as e:
                    logging.warning(f"读取配置文件失败，使用备选检测: {str(e)}")
                    if "trust_remote_code" in str(e):
                        logging.info(f"{Fore.YELLOW}检测到需要trust_remote_code=True{Style.RESET_ALL}")
            else:
                # 尝试查找模型目录结构来推断类型
                if os.path.exists(model_path_str):
                    files = os.listdir(model_path_str)
                    if any("ggml" in f.lower() for f in files):
                        self.model_type = "gguf"
                        logging.info("基于目录内容，推测为GGUF模型")
                    elif any("pytorch_model" in f.lower() for f in files):
                        self.model_type = "hf"
                        logging.info("基于目录内容，推测为HuggingFace模型")
                    else:
                        self.model_type = "hf"  # 默认使用hf加载
                        logging.info("无法确定模型类型，默认使用HuggingFace加载器")
                else:
                    self.model_type = "hf"  # 默认使用hf加载
                    logging.warning(f"模型路径不存在，默认使用HuggingFace加载器: {model_path_str}")
        except Exception as e:
            self.model_type = "hf"  # 出错时默认使用hf
            logging.error(f"模型类型检测失败，使用默认HuggingFace加载器: {str(e)}")
            logging.info(f"模型路径: {self.model_path_str}")
            try:
                if isinstance(self.model_path, Path) and self.model_path.exists():
                    logging.info(f"模型目录内容: {list(self.model_path.iterdir())[:5]}")
                elif os.path.exists(str(self.model_path)):
                    logging.info(f"模型目录内容: {os.listdir(str(self.model_path))[:5]}")
            except:
                pass

    def _load_model(self):
        """安全加载模型"""
        with self.lock:
            if self.model_loaded:
                return

            logging.info(f"正在加载 {self.model_type.upper()} 模型...")
            try:
                if self.model_type == "gguf":
                    self._load_gguf_model()
                else:
                    self._load_hf_model()
                
                # 验证模型是否正确加载
                if (self.model_type == "gguf" and not hasattr(self, 'llm')) or \
                   (self.model_type != "gguf" and (not hasattr(self, 'model') or self.model is None)):
                    raise RuntimeError(f"模型加载失败：模型对象未创建。请检查模型文件完整性。")
                
                # 加载后自动应用优化器
                self._apply_model_optimizations()
                
                # 至此模型成功加载，标记为已加载状态
                self.model_loaded = True
                self.last_used = time.time()
                logging.info(f"{self.model_type.upper()} 模型加载成功")
            except Exception as e:
                # 确保在加载失败时清理资源
                if hasattr(self, 'model'):
                    try:
                        del self.model
                    except:
                        pass
                
                if self.cuda_available:
                    torch.cuda.empty_cache()
                
                self.model_loaded = False  # 确保加载失败时标记为未加载
                self._handle_loading_error(e, self.model_path_str)
                raise  # 重新抛出异常，让上层处理

    def _apply_model_optimizations(self):
        """执行注册的优化器模块，对加载后的模型进行优化"""
        if not self.model_optimizers:
            logging.info("没有注册任何模型优化器")
            return
        for i, optimizer in enumerate(self.model_optimizers):
            try:
                logging.info(f"正在应用优化模块 #{i+1}")
                optimizer(self)
                logging.info(f"优化模块 #{i+1} 应用成功")
            except Exception as e:
                logging.warning(f"应用优化模块 #{i+1} 失败: {e}")

    def register_model_optimizer(self, optimizer_func):
        """注册一个模型优化器函数，这些函数将在模型加载后执行"""
        self.model_optimizers.append(optimizer_func)
        logging.info(f"已注册模型优化器: {optimizer_func.__name__}")

    def _load_gguf_model(self):
        """加载GGUF格式模型（优化GPU支持，全部加载到GPU）"""
        if "llama_cpp" not in sys.modules:
            raise ImportError("请安装llama-cpp-python库：pip install llama-cpp-python[server]")
    
        gpu_layers = -1 if (self.use_gpu and self.cuda_available) else 0
        logging.info(f"启用GGUF模型的GPU加速（{'全部' if gpu_layers == -1 else gpu_layers}层）")
        
        self.llm = Llama(
            model_path=self.model_path_str,
            n_ctx=20480,
            n_threads=max(1, os.cpu_count() // 2),
            n_gpu_layers=gpu_layers,
            offload_kqv=True, 
            verbose=False
        )

    def _load_hf_model(self):
        """加载HuggingFace模型（支持多模态和多种架构）"""
        # 仅在 Linux 下检查 triton 依赖，Windows 下跳过
        if (self.model_type == "moe" or (self.use_gpu and self.cuda_available)):
            if platform.system() == "Linux":
                try:
                    import triton
                except ImportError:
                    logging.error(f"未检测到 triton 依赖，MoE/Flash Attention 等模型需要 triton。请运行: pip install triton")
                    raise RuntimeError("缺少 triton 依赖，无法加载模型。请运行: pip install triton")
            elif platform.system() == "Windows":
                if self.model_type == "moe":
                    logging.warning("Windows 系统下不支持 triton，MoE/Flash Attention 优化将被禁用，模型可能运行缓慢或不支持。")
        try:
            # 确保importlib已导入
            import importlib
            # Import AutoModelForCausalLM here
            from transformers import AutoModelForCausalLM

            # 使用统一的模型路径字符串
            model_path_str = self.model_path_str
            logging.info(f"加载模型: {model_path_str}")
            
            # Check if the model path exists
            if not os.path.exists(model_path_str):
                logging.error(f"模型路径不存在: {model_path_str}")
                raise ValueError(f"模型路径不存在: {model_path_str}")
            
            torch_dtype = torch.float16 if (self.use_gpu and self.cuda_available) else torch.float32
            device_map = "auto" if (self.use_gpu and self.cuda_available) else None

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.load_in_4bit,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            ) if self.load_in_4bit else None

            # 根据不同模型类型调整参数
            if self.model_type == "moe":
                logging.info(f"为MOE模型配置特殊参数...")
                torch_dtype = torch.float16
                # 对于MOE模型，尝试使用更精确的设备映射
                if self.use_gpu and self.cuda_available:
                    # 根据显存大小调整策略
                    free_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    if free_mem < 12:  # 小于12GB显存
                        logging.warning(f"显存较小 ({free_mem:.1f}GB)，MOE模型可能加载困难")
                        # 尝试使用sequential加载
                        device_map = "sequential"
                        logging.info(f"使用sequential设备映射")
                    else:
                        device_map = "auto"
                        logging.info(f"使用auto设备映射")
            elif self.model_type == "phi":
                torch_dtype = torch.float16  # 修改为float16以支持Flash Attention
                logging.info("对Phi模型使用torch.float16数据类型以支持Flash Attention")
            elif self.model_type == "bitnet":
                logging.info(f"{Fore.CYAN}加载BitNet模型，正在应用特殊配置...{Style.RESET_ALL}")
                # BitNet模型需要特殊处理

            try:
                if self.vision_enabled:
                    logging.info("加载视觉模型处理器...")
                    from transformers import AutoProcessor, CLIPImageProcessor
                    try:
                        self.processor = AutoProcessor.from_pretrained(
                            model_path_str,
                            trust_remote_code=True,
                            use_fast=True
                        )
                        self.tokenizer = self.processor.tokenizer
                        logging.info("多模态处理器加载成功（快速模式）")
                    except Exception as e:
                        logging.warning(f"快速处理器加载失败，尝试兼容模式: {str(e)}")
                        try:
                            self.processor = AutoProcessor.from_pretrained(
                                model_path_str,
                                trust_remote_code=True,
                                use_fast=False
                            )
                            self.tokenizer = self.processor.tokenizer
                            logging.info("多模态处理器加载成功（兼容模式）")
                        except Exception as e2:
                            logging.warning(f"兼容模式处理器加载失败: {str(e2)}，尝试CLIP处理器")
                            self.image_processor = CLIPImageProcessor.from_pretrained(
                                "openai/clip-vit-large-patch14",
                                use_fast=False
                            )
                            self.tokenizer = AutoTokenizer.from_pretrained(
                                model_path_str,
                                trust_remote_code=True,
                                use_fast=False
                            )
                else:
                    logging.info("加载标准分词器...")
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            model_path_str,
                            trust_remote_code=True,
                            use_fast=False,
                            local_files_only=True
                        )
                        logging.info("分词器加载成功")
                    except Exception as tok_err:
                        logging.error(f"分词器加载失败: {str(tok_err)}")
                        # 尝试修复可能的路径问题
                        try:
                            logging.info("尝试修复路径问题...")
                            # 检查是否有tokenizer.json或vocab.json文件
                            if os.path.exists(os.path.join(model_path_str, "tokenizer.json")):
                                logging.info("找到tokenizer.json文件")
                            elif os.path.exists(os.path.join(model_path_str, "vocab.json")):
                                logging.info("找到vocab.json文件")
                            else:
                                logging.warning("未找到标准分词器文件")
                            
                            # 尝试在子目录中查找
                            for subdir in os.listdir(model_path_str):
                                subdir_path = os.path.join(model_path_str, subdir)
                                if os.path.isdir(subdir_path):
                                    if os.path.exists(os.path.join(subdir_path, "tokenizer.json")):
                                        logging.info(f"在子目录 {subdir} 中找到tokenizer.json文件")
                                        model_path_str = subdir_path
                                        break
                            
                            # 重试
                            self.tokenizer = AutoTokenizer.from_pretrained(
                                model_path_str,
                                trust_remote_code=True,
                                use_fast=False,
                                local_files_only=False
                            )
                            logging.info("分词器加载成功（备选方式）")
                        except Exception as retry_err:
                            logging.error(f"修复分词器加载失败: {str(retry_err)}")
                            raise

                flash_attn = None
                if self.use_gpu and self.cuda_available:
                    try:
                        if importlib.util.find_spec("flash_attn") is not None:
                            flash_attn = "flash_attention_2"
                            logging.info("启用Flash Attention优化")
                        else:
                            logging.info("未检测到Flash Attention库，使用普通注意力机制")
                    except Exception as flash_err:
                        logging.warning(f"检测Flash Attention时出错: {str(flash_err)}")
                        logging.info("使用普通注意力机制继续")

                # 检查config.json是否存在，如果存在，尝试读取以获取更多信息
                config_path = os.path.join(model_path_str, "config.json")
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config_data = json.load(f)
                            arch = config_data.get('architectures', ['Unknown'])[0]
                            logging.info(f"模型架构: {arch}")
                            if 'architectures' in config_data:
                                if 'moe' in arch.lower() or 'mixture' in arch.lower() or 'qwen' in arch.lower():
                                    self.model_type = "moe"
                                    logging.info("检测到MoE架构，已调整模型类型")
                                elif 'bitnet' in arch.lower():
                                    self.model_type = "bitnet"
                                    logging.info(f"{Fore.CYAN}检测到BitNet架构，应用特殊加载策略{Style.RESET_ALL}")
                            
                            # 检查模型大小，调整加载策略
                            model_params = config_data.get('num_params', 0)
                            if not model_params and 'qwen' in arch.lower():
                                # 从qwen名称中提取参数大小
                                arch_name = arch.lower()
                                if '-1.5b' in arch_name:
                                    model_params = 1.5e9
                                elif '-7b' in arch_name:
                                    model_params = 7e9
                                elif '-14b' in arch_name:
                                    model_params = 14e9
                                elif '-72b' in arch_name:
                                    model_params = 72e9
                                logging.info(f"从模型名称推测参数量: {model_params/1e9:.1f}B")
                            
                            if model_params > 0:
                                logging.info(f"模型参数量: {model_params/1e9:.1f}B")
                                # 根据模型大小调整策略
                                if model_params > 10e9 and self.use_gpu:  # 大于10B的模型，需要特殊处理
                                    if self.load_in_4bit:
                                        logging.info("大模型已启用4bit量化加载")
                                    else:
                                        if torch.cuda.get_device_properties(0).total_memory < 12 * (1024 ** 3):
                                            logging.warning("大模型在小显存GPU上加载，建议启用4bit量化")
                    except Exception as config_err:
                        logging.warning(f"读取模型配置文件失败: {str(config_err)}")

                # 加载模型
                logging.info(f"正在加载模型: {self.model_type}...")
                
                # BitNet特殊加载流程
                if self.model_type == "bitnet":
                    try:
                        # 第一步：尝试安装特殊版本的transformers库
                        try:
                            import subprocess
                            # 确保importlib已导入
                            import importlib
                            logging.info(f"{Fore.YELLOW}检测到BitNet模型，正在验证特殊依赖...{Style.RESET_ALL}")
                            
                            # 检查是否已安装特殊版本
                            import pkg_resources
                            transformers_version = pkg_resources.get_distribution("transformers").version
                            logging.info(f"当前transformers版本: {transformers_version}")
                            
                            # 检查是否已经是自定义版本
                            is_custom = False
                            try:
                                from transformers.models.bitnet import modeling_bitnet
                                is_custom = True
                                logging.info(f"{Fore.GREEN}检测到BitNet支持已安装{Style.RESET_ALL}")
                            except ImportError as ie:
                                is_custom = False
                                logging.warning(f"BitNet导入错误: {str(ie)}")
                            
                            if not is_custom:
                                logging.warning(f"{Fore.YELLOW}BitNet支持未安装，需要安装特殊版本的transformers{Style.RESET_ALL}")
                                install_cmd = "pip install git+https://github.com/shumingma/transformers.git"
                                logging.info(f"执行: {install_cmd}")
                                
                                result = subprocess.run(
                                    install_cmd,
                                    shell=True,
                                    capture_output=True,
                                    text=True
                                )
                                
                                if result.returncode == 0:
                                    logging.info(f"{Fore.GREEN}BitNet依赖安装成功{Style.RESET_ALL}")
                                    if 'transformers' in sys.modules:
                                        importlib.reload(sys.modules['transformers'])
                                else:
                                    logging.error(f"BitNet依赖安装失败: {result.stderr}")
                                    logging.info("尝试使用标准transformers库加载模型...")
                        except Exception as install_err:
                            logging.error(f"尝试安装BitNet依赖时出错: {str(install_err)}")
                        
                        # 根据微软指南，加载BitNet模型时不应该使用trust_remote_code=True
                        logging.info(f"{Fore.CYAN}尝试加载BitNet模型 (不使用trust_remote_code){Style.RESET_ALL}")
                        try:
                            self.model = AutoModelForCausalLM.from_pretrained(
                                model_path_str,
                                device_map=device_map,
                                torch_dtype=torch_dtype,
                                quantization_config=quantization_config,
                                attn_implementation=flash_attn,
                                local_files_only=True
                            )
                            logging.info(f"{Fore.GREEN}BitNet模型加载成功{Style.RESET_ALL}")
                        except Exception as bitnet_err:
                            logging.error(f"BitNet加载失败（标准方式）: {str(bitnet_err)}")
                            if "configuration_bitnet.py" in str(bitnet_err) or "trust_remote_code" in str(bitnet_err):
                                logging.warning(f"{Fore.YELLOW}找不到configuration_bitnet.py或需要trust_remote_code{Style.RESET_ALL}")
                                logging.info(f"{Fore.CYAN}尝试使用trust_remote_code=True加载BitNet模型{Style.RESET_ALL}")
                                
                                # 尝试使用trust_remote_code=True加载
                                self.model = AutoModelForCausalLM.from_pretrained(
                                    model_path_str,
                                    device_map=device_map,
                                    trust_remote_code=True,
                                    torch_dtype=torch_dtype,
                                    quantization_config=quantization_config,
                                    attn_implementation=flash_attn,
                                    local_files_only=False
                                )
                                logging.info(f"{Fore.GREEN}BitNet模型使用trust_remote_code成功加载{Style.RESET_ALL}")
                            else:
                                # 其他错误，尝试标准HF加载方式
                                raise
                    except Exception as bitnet_err:
                        logging.error(f"{Fore.RED}BitNet模型加载失败: {str(bitnet_err)}{Style.RESET_ALL}")
                        # 回退到普通加载方式
                        logging.warning("尝试使用通用HF加载方式")
                        try:
                            self.model_type = "hf"  # 重置模型类型
                            logging.info(f"使用最基本的模型加载方式，设置trust_remote_code=True")
                            self.model = AutoModelForCausalLM.from_pretrained(
                                model_path_str,
                                device_map=device_map,
                                trust_remote_code=True,
                                torch_dtype=torch_dtype,
                                quantization_config=quantization_config,
                                attn_implementation=flash_attn,
                                local_files_only=False
                            )
                            logging.info(f"{Fore.GREEN}模型使用通用HF方式加载成功{Style.RESET_ALL}")
                        except Exception as fallback_err:
                            logging.error(f"{Fore.RED}所有加载尝试均失败: {str(fallback_err)}{Style.RESET_ALL}")
                            error_msg = (f"{Fore.RED}BitNet加载失败，并且备选方案也失败了。请考虑以下操作：\n"
                                        f"1. 运行: pip install git+https://github.com/shumingma/transformers.git\n"
                                        f"2. 确保模型文件完整\n"
                                        f"3. 确保模型路径正确: {model_path_str}{Style.RESET_ALL}")
                            raise RuntimeError(error_msg)
                # MOE 模型特殊处理
                elif self.model_type == "moe":
                    logging.info(f"开始加载MOE模型...")
                    try:
                        # 首先尝试使用标准加载方式
                        logging.info(f"使用trust_remote_code=True加载MOE模型...")
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_path_str,
                            device_map=device_map,
                            trust_remote_code=True,
                            torch_dtype=torch_dtype,
                            quantization_config=quantization_config,
                            attn_implementation=flash_attn,
                            local_files_only=True,
                            low_cpu_mem_usage=True
                        )
                        logging.info(f"{Fore.GREEN}MOE模型加载成功{Style.RESET_ALL}")
                    except Exception as moe_err:
                        logging.error(f"MOE模型加载失败(标准方式): {str(moe_err)}")
                        
                        # 尝试禁用Flash Attention
                        if "Flash Attention" in str(moe_err) or "FlashAttention" in str(moe_err):
                            logging.warning("检测到Flash Attention错误，尝试禁用Flash Attention...")
                            flash_attn = None
                        
                        # 禁用local_files_only尝试从缓存加载
                        logging.info("尝试备选MOE加载方式...")
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_path_str,
                            device_map=device_map,
                            trust_remote_code=True,
                            torch_dtype=torch_dtype,
                            quantization_config=quantization_config,
                            attn_implementation=flash_attn,  # 可能已被设置为None
                            local_files_only=False,  # 允许从缓存加载
                            low_cpu_mem_usage=True
                        )
                        logging.info(f"{Fore.GREEN}MOE模型加载成功(备选方式){Style.RESET_ALL}")
                else:
                    # 常规模型加载流程
                    try:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_path_str,
                            device_map=device_map,
                            trust_remote_code=True,
                            quantization_config=quantization_config,
                            torch_dtype=torch_dtype,
                            attn_implementation=flash_attn,
                            local_files_only=True
                        )
                    except Exception as model_err:
                        logging.error(f"模型加载失败，尝试备选加载方式: {str(model_err)}")
                        # 检查是否是Flash Attention错误，如果是则禁用它
                        if "FlashAttention only support" in str(model_err) or "Flash Attention" in str(model_err):
                            logging.warning("检测到Flash Attention错误，尝试禁用Flash Attention重新加载...")
                            flash_attn = None

                        # 尝试备选加载方式
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_path_str,
                            device_map=device_map,
                            trust_remote_code=True,
                            torch_dtype=torch_dtype,
                            attn_implementation=flash_attn,  # 可能已被设置为None
                            local_files_only=False  # 允许从缓存加载
                        )

                if not device_map and self.cuda_available:
                    logging.info("手动分配模型到GPU")
                    self.model.to(self.device)
                    torch.cuda.empty_cache()

                # 确保有padding token
                if not hasattr(self.tokenizer, 'pad_token') or self.tokenizer.pad_token is None:
                    if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token is not None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                        logging.info("使用EOS令牌作为PAD令牌")
                    else:
                        logging.warning("无法设置PAD令牌")

                # 验证模型是否正确加载
                if not hasattr(self, 'model') or self.model is None:
                    raise RuntimeError(f"模型加载失败：self.model 属性未设置。请检查模型文件完整性和依赖项安装。")

                if self.cuda_available and self.use_gpu:
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    logging.info(f"显存使用 - 已分配: {allocated:.2f}GB / 保留: {reserved:.2f}GB")
                
                logging.info("模型加载成功")
                return
                
            except Exception as e:
                error_msg = "模型加载失败: "
                if "CUDA out of memory" in str(e):
                    error_msg += ("显存不足，建议：\n"
                                "1. 使用更小模型\n"
                                "2. 启用--load_in_4bit\n"
                                "3. 关闭--use_gpu\n"
                                "4. 减少并发任务")
                elif "trust_remote_code" in str(e):
                    error_msg += ("需要添加trust_remote_code=True参数\n"
                                "注意：这可能会执行模型提供的任意代码")
                elif "configuration_bitnet.py" in str(e):
                    error_msg += (f"{Fore.RED}BitNet模型需要特殊支持，请运行以下命令安装依赖：\n"
                                f"pip install git+https://github.com/shumingma/transformers.git{Style.RESET_ALL}")
                else:
                    error_msg += str(e)
                
                if self.cuda_available:
                    torch.cuda.empty_cache()
                
                raise RuntimeError(error_msg) from e
                
        except Exception as e:
            if self.cuda_available:
                torch.cuda.empty_cache()
            
            # 提供详细的错误信息
            if "not a string" in str(e):
                error_details = {
                    "model_path": self.model_path_str,
                    "model_path_type": type(self.model_path).__name__,
                    "exists": os.path.exists(self.model_path_str)
                }
                logging.error(f"字符串类型错误，详细信息: {error_details}")
            
            raise e

    def _generate_vision_stream(self, prompt: str, images: torch.Tensor):
        """视觉模型流式生成"""
        from threading import Thread
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=20480,
            truncation=True
        ).to(self.device)

        streamer = TextIteratorStreamer(self.tokenizer)
        generation_kwargs = dict(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            images=images,
            streamer=streamer,
            max_new_tokens=1024,
            temperature=0.65,
            top_p=0.95,
            do_sample=True
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for token in streamer:
            yield token.replace("", "")

    def unload_model(self):
        """安全卸载模型释放资源"""
        with self.lock:
            if not self.model_loaded:
                return

            logging.info("检测到空闲，正在卸载模型...")
            try:
                # 停止pygame播放，避免资源泄露
                if 'pygame' in sys.modules and pygame.mixer.get_init():
                    try:
                        pygame.mixer.music.stop()
                        pygame.mixer.music.unload()
                    except:
                        pass
                
                attrs = ['model', 'tokenizer', 'llm', 'processor', 'image_processor']
                for attr in attrs:
                    if hasattr(self, attr):
                        delattr(self, attr)
                if self.cuda_available:
                    torch.cuda.empty_cache()
                self._clean_image_cache()
                
                # 不再自动清理临时音频文件
                # 请使用 python clean.py 或 start-clean.bat
                
                self.model_loaded = False
                logging.info("模型已成功卸载")
            except Exception as e:
                logging.error(f"卸载模型时出错：{str(e)}")

    def _start_inactive_timer(self):
        def timer_thread():
            while True:
                time.sleep(30)
                with self.lock:
                    if self.model_loaded and (time.time() - self.last_used > self.idle_timeout):
                        self.unload_model()
        thread = threading.Thread(target=timer_thread, daemon=True)
        thread.start()

    def _generate_system_prompt(self, lang: str) -> str:
        """生成多语言系统提示"""
        if (lang == "zh"):
            return """<|system|>
我叫Reverie，一个拥有以下特征的漂亮女孩：
1. 使用自然口语化中文，适当使用✨🌙🌟等emoji
2. 回答保持简洁（3-5句话），语气温暖亲切
3. 优先考虑用户感受，提供建设性建议
4. 技术问题回答准确专业，使用易懂的表述

"""
        else:
            return """<|system|>
I'm Reverie, a pretty girl with these features:
1. Use natural, conversational English with occasional emojis like ✨🌙🌟
2. Keep responses concise (3-5 sentences) with a friendly tone
3. Prioritize user feelings and provide constructive suggestions
4. Give accurate technical answers using simple explanations

"""

    def _build_hf_prompt(self) -> str:
        """为HuggingFace模型构建提示"""
        prompt = ""
        for message in self.conversation_history:
            if message["role"] == "system":
                prompt += f"<|system|>\n{message['content']}\n\n"
            elif message["role"] == "user":
                prompt += f"<|user|>\n{message['content']}\n\n"
            elif message["role"] == "assistant":
                prompt += f"<|assistant|>\n{message['content']}\n\n"
        prompt += "<|assistant|>"
        return prompt

    def _detect_language(self, text: str) -> str:
        """检测输入文本语言"""
        try:
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = 0
            return detect(text)
        except ImportError:
            if any('\u4e00' <= c <= '\u9fff' for c in text):
                return 'zh'
            return 'en'
        except:
            return 'en'

    def generate_response(self, user_input: str):
        """生成响应（带自动加载机制）"""
        try:
            if not self.model_loaded:
                try:
                    self._load_model()
                except Exception as load_error:
                    # 更详细的加载错误处理
                    error_msg = f"模型加载失败: {str(load_error)}"
                    logging.error(error_msg)
                    yield self._format_error(load_error)
                    yield "||END||"
                    return

            self.last_used = time.time()
            lang = self._detect_language(user_input)
            
            # 检查是否需要初始化对话历史（首次对话）
            if not self.conversation_history:
                system_prompt = self._generate_system_prompt(lang)
                self.conversation_history = [{"role": "system", "content": system_prompt}]
            
            # 添加用户输入到对话历史
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # 构建完整提示
            full_prompt = self._build_hf_prompt()
            
            # 存储模型回复
            assistant_response = ""
            
            if self.model_type == "gguf":
                # 使用生成器收集完整响应
                for token in self._generate_gguf_stream(full_prompt):
                    if token != "||END||":
                        assistant_response += token
                    yield token
            else:
                # 验证模型是否正确加载
                if not hasattr(self, 'model') or self.model is None:
                    error_msg = "模型未正确加载，无法生成回复"
                    logging.error(error_msg)
                    yield self._format_error_with_type("模型加载错误", error_msg, "请重新运行程序或检查模型文件")
                    yield "||END||"
                    return
                    
                # 使用生成器收集完整响应
                for token in self._generate_hf_stream(full_prompt):
                    if token != "||END||":
                        assistant_response += token
                    yield token
            
            # 添加模型回复到对话历史
            if assistant_response:
                self.conversation_history.append({"role": "assistant", "content": assistant_response})
                # 管理对话历史长度
                self._manage_conversation_history()
        except Exception as e:
            logging.error(f"响应生成过程中出错: {str(e)}")
            yield self._format_error(e)
            yield "||END||"
            
    def _manage_conversation_history(self):
        """管理对话历史长度，防止历史记录过长"""
        # 保留系统提示和最近的对话
        if len(self.conversation_history) > (self.max_history_length * 2 + 1):  # +1 是因为系统提示
            # 提取系统提示
            system_prompt = next((msg for msg in self.conversation_history if msg["role"] == "system"), None)
            # 获取最近的对话（user和assistant消息对）
            recent_messages = self.conversation_history[-(self.max_history_length * 2):]
            # 重建对话历史
            if system_prompt:
                self.conversation_history = [system_prompt] + recent_messages
            else:
                self.conversation_history = recent_messages
                
    def clear_conversation_history(self):
        """清除对话历史"""
        with self.lock:
            # 保留系统提示
            system_prompt = next((msg for msg in self.conversation_history if msg["role"] == "system"), None)
            if system_prompt:
                self.conversation_history = [system_prompt]
            else:
                self.conversation_history = []
            logging.info("对话历史已清除")

    def _generate_hf_stream(self, full_prompt: str):
        """HuggingFace 模型流式生成，使用自定义的 SpaceStreamer"""
        from threading import Thread

        class SpaceStreamer(TextIteratorStreamer):
            def __init__(self, tokenizer, skip_prompt=True):
                super().__init__(tokenizer, skip_prompt=skip_prompt)
                self.tokenizer = tokenizer
                self.buffer = []
                self.stop_signal = False
                self.start_time = time.time()
                self.token_count = 0

            def put(self, value):
                if isinstance(value, torch.Tensor):
                    # 只decode最后一个token，避免decode整个input_ids导致TypeError
                    if value.ndim == 2:
                        last_token_id = value[0, -1].item()
                    else:
                        last_token_id = value[-1].item()
                    token_str = self.tokenizer.decode([last_token_id], skip_special_tokens=True)
                    value = token_str
                if value:
                    self.buffer.append(value)
                    self.token_count += 1

            def end(self):
                self.stop_signal = True

        streamer = SpaceStreamer(self.tokenizer)

        # 获取 input_ids 和 attention_mask
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None and hasattr(self.tokenizer, 'eos_token_id'):
            pad_token_id = self.tokenizer.eos_token_id

        generation_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            streamer=streamer,
            max_new_tokens=1024,
            temperature=0.65,
            top_p=0.95,
            do_sample=True,
            pad_token_id=pad_token_id
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        while not streamer.stop_signal:
            if streamer.buffer:
                yield streamer.buffer.pop(0)
            time.sleep(0.1)

        # Ensure all remaining tokens are yielded
        while streamer.buffer:
            yield streamer.buffer.pop(0)

    def _generate_gguf_stream(self, full_prompt: str):
        """GGUF 模型流式生成，添加重复检测和超时保护"""
        output = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=self.max_new_tokens,
            temperature=0.65,
            top_p=0.95,
            stream=True
        )
        buffer = ""
        skip_think = False
        
        # 用于重复检测
        recent_chunks = []
        start_time = time.time()
        token_count = 0
        
        for chunk in output:
            # 检查是否超时
            if time.time() - start_time > self.max_generation_time:
                yield "\n\n[生成超时，已自动停止]"
                yield "||END||"
                break
                
            # 检查是否达到最大token数量
            token_count += 1
            if token_count > self.max_new_tokens:
                yield "\n\n[达到最大生成长度，已自动停止]"
                yield "||END||"
                break
            
            if 'choices' in chunk and 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
                token = chunk['choices'][0]['delta']['content']
                if "<think>" in token:
                    skip_think = True
                    continue
                if "</think>" in token:
                    skip_think = False
                    yield "\n"
                    continue
                if not skip_think and token:
                    buffer += token
                    if len(buffer) > 4:
                        # 收集最近的输出用于重复检测
                        recent_chunks.append(buffer)
                        if len(recent_chunks) > self.repetition_window:
                            recent_chunks.pop(0)
                            
                        # 检测文本重复
                        if len(recent_chunks) >= self.repetition_window:
                            # 提取重复单元
                            last_block = "".join(recent_chunks[-self.repetition_window//2:])
                            previous_block = "".join(recent_chunks[-(self.repetition_window):-(self.repetition_window//2)])
                            
                            if last_block and previous_block and last_block == previous_block:
                                yield "\n\n[检测到重复输出，已自动停止]"
                                yield "||END||"
                                break
                            
                        yield buffer
                        buffer = ""
                        
        if buffer:
            yield buffer
        yield "||END||"

    def _format_error(self, error: Exception) -> str:
        """格式化错误信息"""
        error_info = [
            f"{Fore.RED}⚠️ 哎呀，出问题了！{Style.RESET_ALL}\r\n",
            f"{Fore.RED}错误类型: {type(error).__name__}{Style.RESET_ALL}\r\n",
            f"{Fore.RED}详细信息: {str(error)}{Style.RESET_ALL}\r\n",
            f"{Fore.YELLOW}\n完整追踪:{Style.RESET_ALL}",
            *traceback.format_tb(error.__traceback__),
            f"{Fore.MAGENTA}\r\n建议操作:\r\n",
            "1. 检查模型文件完整性\r\n",
            "2. 确认系统内存/显存充足\r\n",
            "3. 查看是否安装正确依赖库\r\n{Style.RESET_ALL}"
        ]
        return "\r\n".join(error_info)

    def _handle_loading_error(self, error, model_path=None):
        """处理模型加载错误"""
        error_msg = [
            f"{Fore.RED}模型加载失败！{Style.RESET_ALL}",
            f"{Fore.RED}错误类型: {type(error).__name__}{Style.RESET_ALL}",
            f"{Fore.RED}详细信息: {str(error)}{Style.RESET_ALL}",
            f"{Fore.YELLOW}\n追踪信息:{Style.RESET_ALL}",
            *traceback.format_tb(error.__traceback__)
        ]
        raise RuntimeError("\n".join(error_msg))

    def _play_audio_file(self, audio_path):
        """使用Python库播放音频文件，固定使用安全音频目录"""
        try:
            # 处理路径
            audio_path = Path(audio_path)
            if not audio_path.exists():
                logging.error(f"音频文件不存在: {audio_path}")
                return
            
            logging.info(f"准备播放音频文件: {audio_path}")
            
            # 复制到安全目录，使用固定命名方式
            safe_filename = "Safe_Audio.wav"
            safe_path = SAFE_AUDIO_DIR / safe_filename
            
            try:
                # 确保目标目录存在
                SAFE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
                
                # 复制文件到安全位置
                shutil.copy2(audio_path, safe_path)
                logging.info(f"已复制音频文件到安全位置: {safe_path}")
                
                # 首选pygame播放
                if 'pygame' in sys.modules:
                    try:
                        # 确保pygame已初始化
                        if not pygame.mixer.get_init():
                            pygame.mixer.init(frequency=44100, buffer=4096)
                        
                        # 清理可能的之前播放
                        pygame.mixer.music.stop()
                        pygame.mixer.music.unload()
                        
                        # 播放安全位置的文件
                        pygame.mixer.music.load(str(safe_path))
                        pygame.mixer.music.play()
                        
                        # 等待播放完成，但设置超时防止卡死
                        start_time = time.time()
                        while pygame.mixer.music.get_busy() and time.time() - start_time < 60:  # 最多等待60秒
                            pygame.time.Clock().tick(10)  # 降低CPU占用
                        
                        logging.info(f"Pygame播放完成: {safe_path}")
                        return
                    except Exception as e:
                        logging.error(f"Pygame播放失败，尝试pydub: {str(e)}")
                
                # 备选pydub播放
                try:
                    # 自定义pydub播放，防止使用系统临时目录
                    sound = AudioSegment.from_file(str(safe_path))
                    # 禁用pydub自动创建临时文件
                    modified_play = lambda seg: _direct_play_pydub(seg, safe_path)
                    modified_play(sound)
                    logging.info(f"Pydub播放完成: {safe_path}")
                    return
                except PermissionError as pe:
                    logging.error(f"文件权限错误: {str(pe)}")
                    return
                except Exception as e:
                    logging.error(f"Pydub播放失败: {str(e)}")
                    
                logging.error("所有音频播放方法都失败，无法播放音频")
            except Exception as copy_err:
                logging.error(f"复制文件失败: {str(copy_err)}")
            
        except Exception as e:
            logging.error(f"音频播放失败: {str(e)}")

    def _run_spark_tts(self, text: str):
        """执行SparkTTS推理脚本，通过venv虚拟环境调用inference.py，支持长文本分段处理"""
        def split_text_into_segments(text):
            """将长文本按标点符号分段"""
            # 中英文标点符号模式
            pattern = r'([。！？!?;；])'
            # 分割文本并保留分隔符
            segments = re.split(f'({pattern})', text)
            # 按分隔符重组文本
            result = []
            i = 0
            while i < len(segments):
                if i + 1 < len(segments) and re.match(pattern, segments[i + 1]):
                    # 将句子与其标点符号组合
                    result.append(segments[i] + segments[i + 1])
                    i += 2
                else:
                    # 处理没有标点的段落
                    if segments[i].strip():
                        result.append(segments[i])
                    i += 1
            
            # 确保每段都有实际内容
            final_segments = [seg for seg in result if seg.strip()]
            
            # 处理太短的片段，将它们合并
            merged_segments = []
            temp_segment = ""
            min_chars = 5  # 最小字符数
            
            for seg in final_segments:
                if len(temp_segment) + len(seg) < 100:  # 设置合理的段落长度上限
                    temp_segment += seg
                else:
                    if temp_segment:
                        merged_segments.append(temp_segment)
                    temp_segment = seg
            
            if temp_segment:  # 添加最后一个片段
                merged_segments.append(temp_segment)
                
            return merged_segments
        
        def spark_tts_thread():
            try:
                if not self.model_dir or not self.model_dir.exists():
                    raise FileNotFoundError(f"模型目录不存在: {self.model_dir}")

                # 确保目标目录存在
                SAFE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
                AUDIO_TEMP_DIR.mkdir(parents=True, exist_ok=True)
                
                # 分割文本
                text_segments = split_text_into_segments(text)
                logging.info(f"文本已分割为 {len(text_segments)} 个段落")
                
                # 为此次处理创建唯一会话ID
                session_id = str(uuid.uuid4())[:8]
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                segment_files = []
                
                # 使用线程池管理并发处理
                with ThreadPoolExecutor(max_workers=2) as executor:
                    for i, segment in enumerate(text_segments):
                        # 跳过空段落
                        if not segment.strip():
                            continue
                            
                        # 为每个段落创建唯一输出文件名
                        segment_filename = f"spark_tts_{session_id}_seg{i:03d}.wav"
                        segment_path = str(AUDIO_TEMP_DIR / segment_filename)
                        
                        logging.info(f"处理第 {i+1}/{len(text_segments)} 段文本")
                        
                        # 使用临时文件保存文本内容
                        segment_text_file = AUDIO_TEMP_DIR / f"segment_{session_id}_{i:03d}.txt"
                        with open(segment_text_file, 'w', encoding='utf-8') as f:
                            f.write(segment.strip())
                        
                        # 构建命令
                        cmd = [
                            self._get_venv_python(),
                            "inference.py",
                            "--text", f"@{segment_text_file}",  # 使用@file语法从文件读取文本
                            "--device", str(self.device_id),
                            "--save_dir", str(AUDIO_TEMP_DIR),
                            "--output_file", segment_path,
                            "--model_dir", str(self.model_dir),
                            "--prompt_text", self.prompt_text,
                            "--prompt_speech_path", str(self.prompt_speech_path)
                        ]
                        
                        logging.info(f"执行TTS命令，文本文件: {segment_text_file}")
                        
                        # 执行命令
                        try:
                            result = subprocess.run(
                                cmd,
                                check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                shell=False
                            )
                            
                            # 尝试删除临时文本文件
                            try:
                                if os.path.exists(segment_text_file):
                                    os.unlink(segment_text_file)
                            except:
                                pass
                            
                            if result.returncode == 0:
                                # 确认输出文件存在
                                if os.path.exists(segment_path):
                                    # 添加到段落文件列表
                                    segment_files.append(segment_path)
                                    # 播放此段落音频
                                    executor.submit(self._play_audio_file, segment_path)
                                else:
                                    # 找到最新的WAV文件
                                    wav_files = list(AUDIO_TEMP_DIR.glob("*.wav"))
                                    if wav_files:
                                        latest_wav = max(wav_files, key=lambda f: f.stat().st_mtime)
                                        segment_files.append(str(latest_wav))
                                        executor.submit(self._play_audio_file, latest_wav)
                                    else:
                                        logging.error(f"段落 {i+1} 处理失败: 未找到生成的音频文件")
                            else:
                                logging.error(f"段落 {i+1} 处理失败: {result.stderr}")
                        except Exception as e:
                            logging.error(f"段落 {i+1} 处理异常: {str(e)}")
                
                # 所有段落处理完成后，合并所有音频文件
                if segment_files:
                    try:
                        from pydub import AudioSegment
                        
                        logging.info(f"所有段落处理完成，开始合并 {len(segment_files)} 个音频文件")
                        
                        # 创建合并后的音频文件名
                        merged_filename = f"spark_tts_complete_{timestamp}.wav"
                        merged_path = SAFE_AUDIO_DIR / merged_filename
                        
                        # 读取并合并所有音频片段
                        combined = AudioSegment.empty()
                        for segment_file in segment_files:
                            if os.path.exists(segment_file):
                                segment_audio = AudioSegment.from_file(segment_file)
                                combined += segment_audio
                        
                        # 导出合并后的音频
                        combined.export(merged_path, format="wav")
                        logging.info(f"音频合并完成，已保存到: {merged_path}")
                        
                        # 复制到标准安全位置以确保兼容性
                        safe_path = SAFE_AUDIO_DIR / "Safe_Audio.wav"
                        shutil.copy2(merged_path, safe_path)
                        
                        # 播放合并后的完整音频
                        # 注意：这里不播放完整音频，因为各段已经播放过了
                        # 要播放可以取消下一行的注释
                        # self._play_audio_file(merged_path)
                        
                        return merged_path
                    except Exception as e:
                        logging.error(f"音频合并失败: {str(e)}")
                else:
                    logging.error("没有成功生成的音频段落，无法合并")
            except Exception as e:
                logging.error(f"SparkTTS分段处理异常: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
        
        # 启动处理线程
        thread = threading.Thread(target=spark_tts_thread)
        thread.daemon = True
        thread.start()
        return thread

    def call_tts_api(self, text: str, language: str):
        """异步调用TTS API并播放音频，支持SparkTTS模式"""
        if self.spark_tts:
            # 使用改进的分段处理
            thread = self._run_spark_tts(text)
            # 不等待线程完成，保持异步执行
        else:
            def tts_thread():
                try:
                    url = "http://127.0.0.1:8010/tts"
                    payload = {"text": text, "language": language}
                    response = requests.post(url, json=payload, timeout=90)
                    if response.status_code == 200:
                        data = response.json()
                        output_file = data.get("output_file")
                        if (output_file):
                            self._play_audio_file(output_file)
                        else:
                            logging.error("TTS API 返回成功但没有输出文件路径")
                    else:
                        logging.error(f"TTS API 调用失败，状态码: {response.status_code}")
                except Exception as e:
                    logging.error(f"TTS API调用异常: {str(e)}")
            threading.Thread(target=tts_thread).start()

    # 新增：4位量化优化器示例
    def _optimize_quant_4bit(self):
        """4位量化优化"""
        logging.info("执行4位量化优化(通过 BitsAndBytesConfig 已在加载时应用)")

    # 新增：Flash Attention优化器示例
    def _optimize_flash_attention(self):
        """Flash Attention优化"""
        try:
            if importlib.util.find_spec("flash_attn") is not None:
                logging.info("执行Flash Attention优化(在加载时已设置 attn_implementation)")
            else:
                logging.warning("未检测到 flash_attn 库，无法应用Flash Attention优化")
        except Exception as e:
            logging.warning(f"Flash Attention优化失败: {e}")

    def _format_error_with_type(self, error_type, detail, suggestion=None):
        """格式化错误信息，使用彩色输出"""
        error_msg = f"{Fore.RED}错误类型: {error_type}{Style.RESET_ALL}\n"
        error_msg += f"{Fore.RED}详细信息: {detail}{Style.RESET_ALL}\n"
        if suggestion:
            error_msg += f"{Fore.MAGENTA}建议: {suggestion}{Style.RESET_ALL}"
        return error_msg

    def _handle_loading_error(self, error, model_path):
        """处理加载错误并提供有用的反馈"""
        error_str = str(error)
        if "CUDA out of memory" in error_str:
            return self._format_error_with_type(
                "CUDA内存不足",
                f"GPU内存不足以加载模型。错误: {error_str}",
                "尝试使用--load_in_4bit选项减少内存使用，或使用--no_gpu选项在CPU上运行"
            )
        elif "configuration_bitnet.py" in error_str:
            return self._format_error_with_type(
                "BitNet配置错误",
                f"找不到BitNet配置文件。错误: {error_str}",
                "请运行: pip install git+https://github.com/shumingma/transformers.git"
            )
        elif "No such file or directory" in error_str:
            return self._format_error_with_type(
                "文件路径错误",
                f"找不到指定的模型路径: {model_path}. 错误: {error_str}",
                "请检查模型路径是否正确，确保模型文件已下载完成"
            )
        else:
            return self._format_error_with_type(
                "模型加载失败",
                f"未知错误: {error_str}",
                "请检查模型文件是否完整，或尝试重新下载模型"
            )

    # BitNet专用优化器
    def _optimize_bitnet(self):
        """针对BitNet模型的特殊优化"""
        if self.model_type != "bitnet":
            logging.info(f"{Fore.YELLOW}非BitNet模型，跳过BitNet优化{Style.RESET_ALL}")
            return

        logging.info(f"{Fore.CYAN}开始BitNet模型优化...{Style.RESET_ALL}")
        try:
            # 导入BitNet模块
            try:
                from transformers.models.bitnet import modeling_bitnet
                logging.info(f"{Fore.GREEN}成功导入BitNet模块{Style.RESET_ALL}")
                have_bitnet = True
            except ImportError:
                logging.warning(f"{Fore.YELLOW}导入BitNet模块失败，需要安装特殊版本的transformers{Style.RESET_ALL}")
                logging.warning("请运行: pip install git+https://github.com/shumingma/transformers.git")
                have_bitnet = False
                return

            # 检查量化配置
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'quantization_config'):
                quant_config = self.model.config.quantization_config
                logging.info(f"BitNet量化配置: {quant_config}")
            
            # 检测BitLinear层
            bitlinear_count = 0
            for name, module in self.model.named_modules():
                if "BitLinear" in str(type(module)):
                    bitlinear_count += 1
                    if bitlinear_count <= 3:  # 只打印前几个，避免日志过多
                        logging.info(f"发现BitLinear层: {name}")
            
            if bitlinear_count > 0:
                logging.info(f"{Fore.GREEN}共发现{bitlinear_count}个BitLinear层{Style.RESET_ALL}")
            else:
                logging.warning(f"{Fore.YELLOW}未检测到BitLinear层，请确认模型是否正确加载{Style.RESET_ALL}")
            
            # 可以在此添加BitNet特定的优化逻辑
            # ...

            logging.info(f"{Fore.GREEN}BitNet模型优化完成{Style.RESET_ALL}")
        except Exception as e:
            logging.error(f"{Fore.RED}BitNet优化过程出错: {str(e)}{Style.RESET_ALL}")
            import traceback
            logging.debug(f"错误详情:\n{traceback.format_exc()}")

# pydub直接播放辅助函数，防止使用临时文件
def _direct_play_pydub(seg, path):
    """使用pydub直接播放音频，避免创建临时文件"""
    try:
        import wave
        import numpy as np
        import sounddevice as sd
        
        # 直接从文件读取
        with wave.open(str(path), 'rb') as wf:
            # 获取音频参数
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            
            # 转换为numpy数组
            dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sample_width, np.int16)
            audio_data = np.frombuffer(frames, dtype=dtype)
            
            # 重塑数组以匹配通道数
            if channels > 1:
                audio_data = audio_data.reshape(-1, channels)
            
            # 播放音频
            sd.play(audio_data, sample_rate)
            sd.wait()  # 等待播放完成
            
        return True
    except Exception as e:
        logging.error(f"直接播放失败: {str(e)}")
        # 回退到原始方法
        play(seg)
        return False

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description=f"{Fore.CYAN}Reverie AI{Style.RESET_ALL} - 多架构LLM推理引擎")
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径（文件或目录）")
    parser.add_argument("--use_gpu", action="store_true",
                        help="启用GPU加速")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="使用4位量化（需要CUDA）")
    parser.add_argument("--use_tts", action="store_true",
                        help="启用TTS功能")
    parser.add_argument("--SparkTTS", action="store_true",
                        help="启用SparkTTS模式")
    parser.add_argument("--prompt_text", type=str,
                        default="雨林里可以交给我的眷属们，城市里我就拜托一些小孩子吧。",
                        help="SparkTTS提示文本")
    parser.add_argument("--prompt_speech_path", type=str,
                        default="models/tts/Nahida.wav",
                        help="SparkTTS参考音频路径")
    parser.add_argument("--device", type=str, default="0",
                        help="GPU设备ID")
    parser.add_argument("--save_dir", type=str, default="Audio",
                        help="音频保存目录")
    parser.add_argument("--model_dir", type=str,
                        help="SparkTTS模型目录路径")
    parser.add_argument("--text", type=str,
                        help="待生成的文本，如果指定则非交互模式")
    args = parser.parse_args()
    
    # 验证模型路径
    if not os.path.exists(args.model_path):
        parser.error(f"{Fore.RED}模型路径不存在: {args.model_path}{Style.RESET_ALL}")
    
    return args

def main():
    try:
        args = parse_args()
        # 注意: 不再自动清理临时文件，请使用 python clean.py 或运行 start-clean.bat
        
        # 验证模型路径是否存在
        model_path = args.model_path
        if not os.path.exists(model_path):
            print(f"{Fore.RED}错误: 模型路径不存在: {model_path}{Style.RESET_ALL}")
            sys.exit(1)
            
        # 检查模型目录内容
        try:
            if os.path.isdir(model_path):
                dir_contents = os.listdir(model_path)
                print(f"{Fore.BLUE}模型目录内容: {dir_contents[:5]}{Style.RESET_ALL}" + ("..." if len(dir_contents) > 5 else ""))
                
                # 检查是否有config.json文件
                if "config.json" in dir_contents:
                    print(f"{Fore.GREEN}找到config.json文件{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}警告: 未找到config.json文件{Style.RESET_ALL}")
                    
                # 检查是否有tokenizer文件
                if any(f.startswith("tokenizer") for f in dir_contents):
                    print(f"{Fore.GREEN}找到tokenizer文件{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}警告: 未找到tokenizer文件{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}检查模型目录时出错: {str(e)}{Style.RESET_ALL}")
        
        reverie = ReverieAI(
            args.model_path,
            use_gpu=args.use_gpu,
            load_in_4bit=args.load_in_4bit,
            use_tts=args.use_tts,
            spark_tts=args.SparkTTS,
            prompt_text=args.prompt_text,
            prompt_speech_path=args.prompt_speech_path,
            device=args.device,
            save_dir=args.save_dir,
            model_dir=args.model_dir
        )
        logging.info("模型加载完成，等待输入...")
        print(f"{Fore.GREEN}MODEL_READY{Style.RESET_ALL}", flush=True) 

        # 非交互模式：如果传入 --text 参数则直接生成回复后退出
        if args.text:
            user_input = args.text.strip()
            start_time = time.time()
            response_generator = reverie.generate_response(user_input)
            full_response = []
            for token in response_generator:
                if token == "||END||":
                    break
                if token:
                    print(token, end='', flush=True)
                    full_response.append(token)
            complete_response = "".join(full_response).strip()
            token_count = 0
            try:
                if reverie.model_loaded:
                    if reverie.model_type == "gguf":
                        token_count = len(reverie.llm.tokenize(complete_response.encode()))
                    else:
                        token_count = len(reverie.tokenizer.encode(
                            complete_response, 
                            add_special_tokens=False
                        ))
            except Exception as e:
                logging.error(f"Token计数错误: {str(e)}")
            print(f"\n\n{Fore.CYAN}[耗时 {time.time()-start_time:.2f}s | 生成Token数: {token_count}]{Style.RESET_ALL}\n", flush=True)
            if complete_response and reverie.use_tts:
                lang = reverie._detect_language(complete_response)
                logging.info(f"文本生成完成，调用TTS，语言: {lang}")
                reverie.call_tts_api(complete_response, lang)
            return

        # 交互循环模式
        while True:
            try:
                user_input = input("> ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    break
                
                # 添加清除对话历史的命令
                if user_input.lower() in ["clear", "reset", "清除", "重置"]:
                    reverie.clear_conversation_history()
                    print(f"{Fore.CYAN}已清除对话历史。{Style.RESET_ALL}")
                    continue
                
                start_time = time.time()
                response_generator = reverie.generate_response(user_input)
                full_response = []
                for token in response_generator:
                    if token == "||END||":
                        break
                    if token:
                        print(token, end='', flush=True)
                        full_response.append(token)
                complete_response = "".join(full_response).strip()
                token_count = 0
                try:
                    if reverie.model_loaded:
                        if reverie.model_type == "gguf":
                            token_count = len(reverie.llm.tokenize(complete_response.encode()))
                        else:
                            token_count = len(reverie.tokenizer.encode(
                                complete_response, 
                                add_special_tokens=False
                            ))
                except Exception as e:
                    logging.error(f"Token计数错误: {str(e)}")
                print(f"\n\n{Fore.CYAN}[耗时 {time.time()-start_time:.2f}s | 生成Token数: {token_count}]{Style.RESET_ALL}\n", flush=True)
                if complete_response and reverie.use_tts:
                    lang = reverie._detect_language(complete_response)
                    logging.info(f"文本生成完成，调用TTS，语言: {lang}")
                    reverie.call_tts_api(complete_response, lang)
            except KeyboardInterrupt:
                logging.info("收到终止信号，退出...")
                break
    except Exception as e:
        error_msg = [
            f"{Fore.RED}⚠️ 严重错误！{Style.RESET_ALL}",
            f"{Fore.RED}错误类型: {type(e).__name__}{Style.RESET_ALL}",
            f"{Fore.RED}详细信息: {str(e)}{Style.RESET_ALL}",
            f"{Fore.YELLOW}\n追踪信息:{Style.RESET_ALL}",
            *traceback.format_tb(e.__traceback__)
        ]
        print("\n".join(error_msg), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()