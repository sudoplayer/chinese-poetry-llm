#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一日志记录模块
负责记录训练和评估过程中的query、standard和生成结果
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

# 设置日志
logger = logging.getLogger(__name__)


class Logger:
    """统一日志记录器"""
    
    def __init__(self, log_dir: str = "logs"):
        """
        初始化日志记录器
        
        参数:
            log_dir: 日志保存目录
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"日志目录: {self.log_dir}")
    
    def log_sample(self, sample_index: int, query: str, reference: str, 
                   completions_data: List[Dict[str, Any]]):
        """
        记录单个样本的数据
        
        参数:
            sample_index: 样本索引
            query: 心路历程
            reference: 参考诗作
            completions_data: 完成结果列表，每个元素包含:
                {
                    "content": str,
                    "evaluation_result": Dict (完整的评估结果)
                }
        """
        try:
            # 构造保存数据
            sample_data = {
                "sample_index": sample_index,
                "query": query,
                "reference": reference,
                "completions": completions_data,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 生成文件名（五位数字）
            filename = f"sample_{sample_index:05d}.json"
            filepath = self.log_dir / filename
            
            # 保存为JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"已保存样本 {sample_index} 的日志: {filepath}")
            
        except Exception as e:
            logger.error(f"保存样本 {sample_index} 的日志失败: {e}")

    def log_batch_samples(self, batch_data: List[Dict[str, Any]]):
        """
        批量记录多个样本的数据
        
        参数:
            batch_data: 批次数据列表，每个元素包含:
                {
                    "sample_index": int,
                    "query": str,
                    "reference": str,
                    "completions_data": List[Dict[str, Any]]
                }
        """
        try:
            success_count = 0
            for sample_data in batch_data:
                self.log_sample(
                    sample_index=sample_data["sample_index"],
                    query=sample_data["query"],
                    reference=sample_data["reference"],
                    completions_data=sample_data["completions_data"]
                )
                success_count += 1
            
            logger.info(f"批量保存完成，成功保存 {success_count}/{len(batch_data)} 个样本")
            
        except Exception as e:
            logger.error(f"批量保存样本日志失败: {e}")


# 全局logger实例
_global_logger = None


def set_global_logger(logger: Logger):
    """设置全局logger实例"""
    global _global_logger
    _global_logger = logger


def get_global_logger() -> Logger:
    """获取全局logger实例"""
    return _global_logger
