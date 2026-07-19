#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO诗词训练系统
基于律诗合集_refine_fin.csv中的Mental_Journey_Simple，使用Qwen3 4B Instruct模型生成诗词，
通过DeepSeek评分，再用GRPO算法优化模型

GPU适配兼容性说明：
- 当 gpu_flag = 'RTX2080Ti' 时，启用RTX2080Ti专用优化配置：
  * 环境变量优化：禁用torch.compile、unsloth编译、Flash Attention 2
  * CUDA后端配置：使用math实现，禁用Flash和内存高效实现
  * 模型量化：启用4bit量化以节省显存
  * Triton共享内存限制：设置为65536字节
  * 训练参数：批次大小1，梯度累积1，生成数量4，禁用混合精度
- 当 gpu_flag = 'RTX4090' 时，启用RTX4090专用优化配置：
  * CUDA后端配置：启用Flash Attention和内存高效实现
  * 模型量化：禁用4bit量化，充分利用24GB显存
  * 训练参数：批次大小4，梯度累积4，生成数量8，启用混合精度
- 当 gpu_flag 为其他值时，使用默认配置，适用于其他GPU
"""

import os
import sys
import logging
import torch
import warnings
from pathlib import Path
from typing import List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 屏蔽PyTorch相关的FutureWarning警告
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", message=".*torch.backends.cuda.sdp_kernel.*")
warnings.filterwarnings("ignore", message=".*torch.nn.attention.sdpa_kernel.*")

from poetry_core.poetry_data_loader import PoetryData, PoetryDataLoader
from poetry_core.poetry_evaluator import PoetryEvaluator
from poetry_core.poetry_generator import PoetryGenerator
from poetry_core.poetry_logger import Logger, set_global_logger
try:
    from grpo.grpo_config import config
except ImportError:
    from grpo_config import config

# 导入必要的库
from unsloth import FastModel
from trl import GRPOConfig, GRPOTrainer as TRLGRPOTrainer

# 设置CUDA设备
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GRPOTrainer:
    """GRPO训练器"""
    
    def __init__(self, model, tokenizer, evaluator, generator, data_loader):
        """初始化训练器"""
        self.model = model
        self.tokenizer = tokenizer
        self.evaluator = evaluator
        self.generator = generator
        self.data_loader = data_loader
        
        # 初始化训练日志记录器
        if config.ENABLE_LOGGING:
            self.training_logger = Logger(log_dir="grpo_logs")
            set_global_logger(self.training_logger)
            logger.info("训练日志记录功能已启用")
        else:
            self.training_logger = None
            logger.info("训练日志记录功能已禁用")
    
    def train(self, data_list: List[PoetryData], max_steps: int = 100):
        """开始GRPO训练"""
        logger.info("开始GRPO训练...")
        
        # 格式化数据 - 使用数据加载器
        dataset = self.data_loader.format_dataset_for_grpo(data_list, self.generator)

        # 创建奖励函数 - 使用评估器
        reward_funcs = self.evaluator.create_reward_functions(config)
        
        # 配置训练参数
        training_args = GRPOConfig(
            learning_rate=5e-6,
            weight_decay=0.01,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            logging_steps=1,
            per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
            num_generations=config.NUM_GENERATIONS,
            max_prompt_length=config.MAX_PROMPT_LENGTH,
            max_completion_length=config.MAX_COMPLETION_LENGTH,
            max_steps=config.TRAINING_STEPS,
            save_steps=config.SAVE_STEPS,
            save_total_limit=2,  # 最多只保留save_total_limit个模型检查点文件
            report_to="swanlab",
            run_name="qwen3-4B-Instruct-GRPO",
            output_dir="grpo_outputs",
            seed=config.RANDOM_SEED,
            # GSPO/GRPO 算法配置
            importance_sampling_level=config.IMPORTANCE_SAMPLING_LEVEL,
            loss_type=config.LOSS_TYPE,
            num_iterations=config.NUM_ITERATIONS,
            # 按顺序采样配置 - 不随机打乱数据
            shuffle_dataset=False,  # 禁用数据集的随机打乱
            dataloader_drop_last=False,  # 不丢弃最后一个不完整的批次
            # 混合精度训练配置
            bf16=config.USE_BF16,  # 根据GPU类型设置精度
            # 显式设置生成参数
            temperature=0.7,  # 与模型默认值一致
            top_p=0.8,        # 与模型默认值一致
        )
        
        # 创建训练器
        grpo_trainer = TRLGRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset,
        )
        
        # 显示训练前内存状态
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        logger.info(f"{start_gpu_memory} GB of memory reserved.")
        
        # 开始训练
        trainer_stats = grpo_trainer.train()
        
        # 显示训练后内存状态
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        
        logger.info(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        logger.info(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
        logger.info(f"Peak reserved memory = {used_memory} GB.")
        logger.info(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        logger.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
        logger.info(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
        
        logger.info("GRPO训练完成！")
        logger.info(f"训练统计信息：{trainer_stats}")
        
        # 保存模型
        try:
            # 保存LoRA适配器
            self.model.save_pretrained("./grpo_lora_adapters")
            self.tokenizer.save_pretrained("./grpo_lora_adapters")
            logger.info("LoRA适配器已保存到 ./grpo_lora_adapters")
            
            # 合并并保存完整模型
            logger.info("正在合并LoRA适配器到基础模型...")
            self.model.save_pretrained_merged("grpo_merged_model", self.tokenizer, save_method="merged_16bit")
            logger.info("合并后的模型已保存到 grpo_merged_model")
            
        except Exception as e:
            logger.error(f"保存模型时出现错误: {str(e)}")
            logger.info("尝试使用备用保存方法...")
            try:
                # 备用方法：使用lora_merged保存
                self.model.save_pretrained_merged("grpo_merged_model", self.tokenizer, save_method="lora_merged")
                logger.info("使用备用方法保存成功: grpo_merged_model")
            except Exception as e2:
                logger.error(f"备用保存方法也失败: {str(e2)}")
                logger.info("请检查模型状态和磁盘空间")

def main():
    try:
        # 0. 验证配置
        logger.info("=== 步骤0: 验证配置 ===")
        logger.info(f"GPU标志: {config.GPU_FLAG}")
        logger.info(f"模型名称: {config.MODEL_NAME}")
        
        # 1. 加载数据
        logger.info("=== 步骤1: 加载数据 ===")
        
        # 检查数据文件是否存在
        data_file = os.getenv("GRPO_DATA_PATH", "data/dataset_grpo.csv")
        if not os.path.exists(data_file):
            logger.error(f"数据文件不存在: {data_file}")
            logger.error("请确保数据文件在当前目录下")
            raise FileNotFoundError(f"数据文件不存在: {data_file}")
        
        data_loader = PoetryDataLoader(data_file)
        data_list = data_loader.load_data(idx_start=config.IDX_START, idx_end=config.IDX_END)
        logger.info(f"成功加载 {len(data_list)} 条数据")
        
        # 2. 初始化模型
        logger.info("=== 步骤2: 初始化模型 ===")
        model, tokenizer = FastModel.from_pretrained(
            model_name=config.MODEL_NAME,
            max_seq_length=config.MAX_SEQ_LENGTH,
            load_in_4bit=config.LOAD_IN_4BIT,   # 根据GPU类型动态设置量化
            device_map={"": local_rank},
            token=config.HUGGINGFACE_TOKEN,
        )     

        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
        # ids = tokenizer("hello", return_tensors="pt").input_ids.to(model.device)
        # out = model(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=4)
        # print("probe", tuple(out.logits.shape))  # 期望 (..., 2560)

        # 设置聊天模板
        import swanlab
        swanlab.init(project="qwen3-4B-Instruct-GRPO", config={"algorithm": config.RL_ALGORITHM})

        from unsloth.chat_templates import get_chat_template
        tokenizer = get_chat_template(
            tokenizer,
            chat_template=config.CHAT_TEMPLATE,
        )
        
        model = FastModel.get_peft_model(
            model,
            r=config.LORA_RANK,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=config.LORA_RANK*2,
            use_gradient_checkpointing="unsloth",
            random_state=config.RANDOM_SEED,
        )

        # 显式禁用缓存以避免警告
        model.config.use_cache = False
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.max_length = None
        if getattr(model.config, "max_length", None):
            model.config.max_length = None
        logger.info("模型初始化完成")
        
        # 3. 初始化组件
        logger.info("=== 步骤3: 初始化组件 ===")
        evaluator = PoetryEvaluator(
            max_concurrent_requests=config.MAX_CONCURRENT_REQUESTS
        )
        generator = PoetryGenerator()
        grpo_trainer = GRPOTrainer(model, tokenizer, evaluator, generator, data_loader)
        
        # 4. 开始GRPO训练
        logger.info("=== 步骤4: 开始GRPO训练 ===")
        grpo_trainer.train(data_list, max_steps=config.TRAINING_STEPS)
        
        logger.info("=== GRPO训练完成 ===")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise

if __name__ == "__main__":
    """
    使用说明：
    1. GPU类型标识：
       - 'RTX2080Ti': 启用RTX2080Ti专用优化（保守配置）
       - 'RTX4090': 启用RTX4090专用优化（高性能配置）
       - 其他值: 使用默认配置
    2. 确保数据文件 '律诗合集_refine_fin.csv' 存在
    3. 设置环境变量 DEEPSEEK_API_KEY 用于诗词评分
    4. 运行脚本开始GRPO训练
    """

    main()
