#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诗歌评估系统
基于测试集数据，使用Qwen3 4B Instruct模型生成诗歌，
通过DeepSeek评分，记录生成的诗作内容和评分结果

GPU适配兼容性说明：
- 当 gpu_flag = 'RTX2080Ti' 时，启用RTX2080Ti专用优化配置：
  * 环境变量优化：禁用torch.compile、unsloth编译、Flash Attention 2
  * CUDA后端配置：使用math实现，禁用Flash和内存高效实现
  * 模型量化：启用4bit量化以节省显存
  * Triton共享内存限制：设置为65536字节
- 当 gpu_flag = 'RTX4090' 时，启用RTX4090专用优化配置：
  * CUDA后端配置：启用Flash Attention和内存高效实现
  * 模型量化：启用4bit量化以节省显存
"""

import os
import sys
import logging
import torch
import warnings
import time
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
from poetry_core.poetry_generator import PoetryGenerator, DeepSeekGenerator
from poetry_core.poetry_logger import Logger, set_global_logger
try:
    from eval.eval_config import config
except ImportError:
    from eval_config import config

# 导入必要的库
from unsloth import FastModel

# 设置CUDA设备
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)

class PoetryEvalSystem:
    """诗歌评估系统"""
    
    def __init__(self, model, tokenizer, evaluator, generator, data_loader):
        """初始化评估系统"""
        self.model = model
        self.tokenizer = tokenizer
        self.evaluator = evaluator
        self.generator = generator
        self.data_loader = data_loader
        
        # 检查是否为DeepSeek模式
        self.is_deepseek_mode = config.GPU_FLAG == "DeepSeek"
        
        # 初始化评估日志记录器
        if config.ENABLE_LOGGING:
            self.eval_logger = Logger(log_dir="eval_logs")
            set_global_logger(self.eval_logger)
            logger.info("评估日志记录功能已启用")
        else:
            self.eval_logger = None
            logger.info("评估日志记录功能已禁用")
    
    def generate_poetry_batch(self, queries: List[str]) -> List[str]:
        """批量生成诗歌"""
        try:
            if self.is_deepseek_mode:
                # DeepSeek API模式
                logger.info("使用DeepSeek API模式生成诗歌")
                return self.generator.generate_poetry_batch(queries)
            else:
                # 本地模型模式
                logger.info("使用本地模型模式生成诗歌")
                # 1. 批量构建提示词
                prompts = []
                for query in queries:
                    prompt = self.generator._get_create_prompt(query)
                    prompts.append(prompt)
                
                # 2. 批量格式化输入
                messages_list = []
                for prompt in prompts:
                    messages = [{"role": "user", "content": prompt}]
                    messages_list.append(messages)
                
                # 3. 批量tokenize - 使用正确的字典格式
                inputs_list = []
                for messages in messages_list:
                    # 使用tokenizer.encode_plus来获取字典格式
                    encoded = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors=None
                    )
                    # 将token ID列表转换为字典格式
                    inputs_dict = {
                        "input_ids": encoded,
                        "attention_mask": [1] * len(encoded)  # 创建attention mask
                    }
                    inputs_list.append(inputs_dict)
                
                # 4. Padding到相同长度 - 使用left padding避免decoder-only架构警告
                inputs_batch = self.tokenizer.pad(
                    inputs_list,
                    padding=True,
                    padding_side='left',  # 明确指定使用left padding
                    return_tensors="pt"
                ).to(self.model.device)
                
                # 5. 批量生成诗歌
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs_batch.input_ids,
                        attention_mask=inputs_batch.attention_mask,
                        max_new_tokens=config.MAX_COMPLETION_LENGTH,
                        temperature=0.7,
                        top_p=0.8,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                # 6. 批量解码输出
                generated_texts = []
                for i, output in enumerate(outputs):
                    # 获取输入长度，只解码新生成的部分
                    input_length = inputs_batch.input_ids[i].shape[0]
                    generated_text = self.tokenizer.decode(
                        output[input_length:], 
                        skip_special_tokens=True
                    ).strip()
                    generated_texts.append(generated_text)
                
                return generated_texts
            
        except Exception as e:
            logger.error(f"批量生成诗歌失败: {e}")
            # 返回与输入相同长度的空字符串列表
            return [""] * len(queries)
    
    def evaluate_test_dataset(self, data_list: List[PoetryData]):
        """评估测试集数据（批量处理模式）"""
        logger.info("开始评估测试集数据（批量处理模式）...")
        
        total_samples = len(data_list)
        batch_size = config.EVAL_BATCH_SIZE
        logger.info(f"总共需要评估 {total_samples} 个样本，批量大小: {batch_size}")
        
        # 初始化进度统计
        start_time = time.time()
        successful_samples = 0
        failed_samples = 0
        processed_samples = 0
        
        # 批量处理循环
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_data = data_list[batch_start:batch_end]
            batch_size_actual = len(batch_data)
            
            batch_start_time = time.time()
            processed_samples += batch_size_actual
            current_progress = processed_samples / total_samples * 100
            
            # 只在每100个样本或最后一个batch时打印进度信息
            if processed_samples % 100 == 0 or batch_end == total_samples:
                elapsed_time = time.time() - start_time
                if processed_samples > 0:
                    avg_time_per_sample = elapsed_time / processed_samples
                    remaining_samples = total_samples - processed_samples
                    estimated_remaining_time = avg_time_per_sample * remaining_samples
                    estimated_total_time = elapsed_time + estimated_remaining_time
                    
                    logger.info(f"进度: {processed_samples}/{total_samples} ({current_progress:.1f}%) | "
                              f"已用时间: {elapsed_time/60:.1f}分钟 | "
                              f"预估剩余: {estimated_remaining_time/60:.1f}分钟 | "
                              f"预估总计: {estimated_total_time/60:.1f}分钟")
                else:
                    logger.info(f"进度: {processed_samples}/{total_samples} ({current_progress:.1f}%) | 开始处理...")
            
            try:
                # 批量生成诗歌（并行）
                generation_start = time.time()
                queries = [data.query for data in batch_data]
                generated_poems = self.generate_poetry_batch(queries)
                generation_time = time.time() - generation_start
                
                logger.info(f"批量生成完成，耗时: {generation_time:.2f}秒，平均每样本: {generation_time/batch_size_actual:.2f}秒")
                
                # 批量评估诗歌（并发）
                evaluation_start = time.time()
                valid_poems = []
                valid_data = []
                
                # 过滤掉生成失败的诗歌
                for data, generated_poetry in zip(batch_data, generated_poems):
                    if generated_poetry:
                        valid_poems.append(generated_poetry)
                        valid_data.append(data)
                    else:
                        logger.warning(f"样本 {data.idx} 生成诗歌失败")
                        failed_samples += 1
                
                # 批量评分
                if valid_poems:
                    evaluation_results = self.evaluator.evaluate_poetry_batch(valid_poems)
                    evaluation_time = time.time() - evaluation_start
                    
                    logger.info(f"批量评分完成，耗时: {evaluation_time:.2f}秒，平均每样本: {evaluation_time/len(valid_poems):.2f}秒")
                    
                    # 记录结果
                    for data, generated_poetry, evaluation_result in zip(valid_data, valid_poems, evaluation_results):
                        try:
                            if self.eval_logger:
                                completions_data = [{
                                    "content": generated_poetry,
                                    "evaluation_result": {
                                        "体裁判断": evaluation_result['体裁判断'],
                                        "详细评审": evaluation_result['详细评审'].model_dump(),
                                        "得分": evaluation_result['得分'].model_dump(),
                                    }
                                }]
                                
                                self.eval_logger.log_sample(
                                    sample_index=data.idx,
                                    query=data.query,
                                    reference=data.reference,
                                    completions_data=completions_data
                                )
                            
                            successful_samples += 1
                            
                        except Exception as e:
                            logger.error(f"记录样本 {data.idx} 时出现错误: {e}")
                            failed_samples += 1
                            continue
                
                batch_total_time = time.time() - batch_start_time
                logger.info(f"批次 {batch_start//batch_size + 1} 处理完成，耗时: {batch_total_time:.2f}秒")
                
            except Exception as e:
                logger.error(f"处理批次 {batch_start//batch_size + 1} 时出现错误: {e}")
                failed_samples += batch_size_actual
                continue
        
        # 输出最终统计信息
        total_time = time.time() - start_time
        avg_time_per_sample = total_time / total_samples if total_samples > 0 else 0
        
        logger.info("=" * 60)
        logger.info("测试集评估完成！")
        logger.info(f"统计信息:")
        logger.info(f"   • 总样本数: {total_samples}")
        logger.info(f"   • 成功处理: {successful_samples}")
        logger.info(f"   • 处理失败: {failed_samples}")
        logger.info(f"   • 成功率: {successful_samples/total_samples*100:.1f}%")
        logger.info(f"   • 总耗时: {total_time/60:.1f}分钟")
        logger.info(f"   • 平均每样本: {avg_time_per_sample:.2f}秒")
        logger.info("=" * 60)

def main():
    try:
        # 0. 验证配置
        logger.info("=== 步骤0: 验证配置 ===")
        logger.info(f"GPU标志: {config.GPU_FLAG}")
        logger.info(f"模型名称: {config.MODEL_NAME}")
        
        # 1. 加载测试集数据
        logger.info("=== 步骤1: 加载测试集数据 ===")
        
        # 检查测试集数据文件是否存在
        test_data_file = os.getenv("EVAL_DATA_PATH", "data/dataset_test.csv")
        if not os.path.exists(test_data_file):
            logger.error(f"测试集数据文件不存在: {test_data_file}")
            logger.error("请确保测试集数据文件在当前目录下")
            raise FileNotFoundError(f"测试集数据文件不存在: {test_data_file}")
        
        data_loader = PoetryDataLoader(test_data_file)
        data_list = data_loader.load_data(idx_start=config.IDX_START, idx_end=config.IDX_END)
        logger.info(f"成功加载索引 [{config.IDX_START}, {config.IDX_END}) 区间的 {len(data_list)} 条测试集数据")
        
        # 2. 初始化模型
        logger.info("=== 步骤2: 初始化模型 ===")
        if config.GPU_FLAG == "DeepSeek":
            # DeepSeek API模式，不需要加载本地模型
            logger.info("DeepSeek API模式，跳过本地模型加载")
            model = None
            tokenizer = None
        else:
            # 本地模型模式
            model, tokenizer = FastModel.from_pretrained(
                model_name=config.MODEL_NAME,
                max_seq_length=config.MAX_SEQ_LENGTH,
                load_in_4bit=config.LOAD_IN_4BIT,   # 根据GPU类型动态设置量化
                device_map={"": local_rank},
                token=config.HUGGINGFACE_TOKEN,
            )     
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"

            # 设置聊天模板
            from unsloth.chat_templates import get_chat_template
            tokenizer = get_chat_template(
                tokenizer,
                chat_template=config.CHAT_TEMPLATE,
            )

            # 显式禁用缓存以避免警告
            model.config.use_cache = False
            
            # 设置tokenizer的padding_side为'left'，避免decoder-only架构的right-padding警告
            tokenizer.padding_side = 'left'
            logger.info("模型初始化完成，已设置padding_side='left'")
        
        # 3. 初始化组件
        logger.info("=== 步骤3: 初始化组件 ===")
        evaluator = PoetryEvaluator(
            max_concurrent_requests=config.MAX_CONCURRENT_REQUESTS
        )
        
        # 根据模式选择生成器
        if config.GPU_FLAG == "DeepSeek":
            generator = DeepSeekGenerator(
                max_concurrent_requests=config.MAX_CONCURRENT_REQUESTS
            )
            logger.info("使用DeepSeek API生成器")
        else:
            generator = PoetryGenerator()
            logger.info("使用本地模型生成器")
            
        eval_system = PoetryEvalSystem(model, tokenizer, evaluator, generator, data_loader)
        
        # 4. 开始评估测试集
        logger.info("=== 步骤4: 开始评估测试集 ===")
        eval_system.evaluate_test_dataset(data_list)
        
        logger.info("=== 测试集评估完成 ===")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise

if __name__ == "__main__":
    """
    使用说明：
    1. GPU类型标识：
       - 'RTX2080Ti': 启用RTX2080Ti专用优化（保守配置）
       - 'RTX4090': 启用RTX4090专用优化（高性能配置）
    2. 确保测试集数据文件 'dataset_test.csv' 存在
    3. 设置环境变量 DEEPSEEK_API_KEY 用于诗歌评分
    4. 设置环境变量 HUGGINGFACE_API_KEY 用于模型下载
    5. 运行脚本开始测试集评估
    """

    main()
