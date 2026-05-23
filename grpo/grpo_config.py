#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO格律诗训练系统配置文件
包含所有全局配置参数和GPU适配设置
"""

import os
import torch


class Config:
    """全局配置类"""
    
    # GPU适配兼容性配置
    GPU_FLAG = 'RTX2080Ti'  # 设置为 'RTX2080Ti' 启用RTX2080Ti适配，其他值则使用默认配置，对应AutoDL

    # 训练日志记录
    ENABLE_LOGGING = True  # 设置为 True 启用训练日志记录，False 则不启用
    
    # 全局配置参数
    MAX_SEQ_LENGTH = 8192  # 最大序列长度
    MAX_PROMPT_LENGTH = 4096  # 最大提示长度
    MAX_COMPLETION_LENGTH = 512  # 最大生成长度

    # 环境变量设置
    HF_ENDPOINT = "https://hf-mirror.com"
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

    # 数据集参数
    IDX_START = 0  # 数据起始索引（包含）
    IDX_END = 10000  # 数据结束索引（不包含）
    
    # 批量评分配置
    MAX_CONCURRENT_REQUESTS = 32  # 最大并发请求数

    # 训练参数
    LORA_RANK = 16
    TRAINING_EPOCHS = 1
    SAVE_STEPS = 50
    RANDOM_SEED = 42
    
    @classmethod
    def _apply_gpu_config(cls):
        """根据GPU类型应用配置"""
        # 设置环境变量
        os.environ['HF_ENDPOINT'] = cls.HF_ENDPOINT
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if cls.GPU_FLAG == 'RTX2080Ti':
            # RTX2080Ti 专用优化配置
            print("检测到RTX2080Ti，启用专用优化配置...")
            os.environ["FLASH_ATTENTION_2_DISABLE"] = "1" # 禁用 FA2
            os.environ["TRITON_SHARED_MEMORY_LIMIT"] = "65536"
            cls.MODEL_NAME = os.getenv(
                "MODEL_NAME", "unsloth/Qwen3-4B-Instruct-2507"
            )
            cls.CHAT_TEMPLATE = "qwen3-instruct"

            # RTX2080Ti 专用训练参数
            cls.PER_DEVICE_TRAIN_BATCH_SIZE = 32
            cls.GRADIENT_ACCUMULATION_STEPS = 1
            cls.LOAD_IN_4BIT = True
            cls.USE_MIXED_PRECISION = False
            cls.USE_BF16 = False  # RTX2080Ti不支持bfloat16
            cls.NUM_GENERATIONS = 4
            
            # RTX2080Ti 专用CUDA后端配置
            if torch.cuda.is_available():
                torch.backends.cuda.sdp_kernel(
                    enable_flash=False,        # 禁用 Flash Attention
                    enable_math=True,          # 关键：用 math 实现（最保守、共享内存占用最低）
                    enable_mem_efficient=False, # 禁用内存高效实现
                )
        elif cls.GPU_FLAG == 'RTX4090':
            # RTX4090 专用优化配置
            print("检测到RTX4090，启用专用优化配置...")
            cls.MODEL_NAME = os.getenv(
                "MODEL_NAME", "unsloth/Qwen3-4B-Instruct-2507"
            )
            cls.CHAT_TEMPLATE = "qwen3-instruct"
            
            # RTX4090 专用训练参数
            cls.PER_DEVICE_TRAIN_BATCH_SIZE = 1
            cls.GRADIENT_ACCUMULATION_STEPS = 1
            cls.LOAD_IN_4BIT = True
            cls.USE_MIXED_PRECISION = True
            cls.USE_BF16 = True  # RTX4090支持bfloat16
            cls.NUM_GENERATIONS = 4

            # RTX4090 专用CUDA后端配置
            if torch.cuda.is_available():
                torch.backends.cuda.sdp_kernel(
                    enable_flash=True,         # 启用 Flash Attention
                    enable_math=True,          # 启用数学实现
                    enable_mem_efficient=True, # 启用内存高效实现
                )

        cls.TRAINING_STEPS = int(cls.TRAINING_EPOCHS * (cls.IDX_END - cls.IDX_START) / (cls.PER_DEVICE_TRAIN_BATCH_SIZE/cls.NUM_GENERATIONS) / cls.GRADIENT_ACCUMULATION_STEPS)
    
    @classmethod
    def initialize(cls):
        """初始化配置"""
        cls._apply_gpu_config()


# 创建全局配置实例
config = Config()

# 初始化配置
config.initialize()
