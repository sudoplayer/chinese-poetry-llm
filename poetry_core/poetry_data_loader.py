#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诗词数据加载模块
负责从CSV文件加载和预处理诗词数据
"""

import pandas as pd
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from datasets import Dataset

# 设置日志
logger = logging.getLogger(__name__)

class PoetryData(BaseModel):
    """诗词数据模型"""
    idx: int = Field(..., description="行索引")
    query: str = Field(..., description="查询内容")
    reference: str = Field(..., description="诗作范例")

class PoetryDataLoader:
    """诗词数据加载器"""
    
    def __init__(self, csv_path: str):
        """初始化数据加载器"""
        self.csv_path = csv_path
        
    def load_data(self, idx_start: Optional[int] = None, idx_end: Optional[int] = None) -> List[PoetryData]:
        """从CSV文件加载数据"""
        try:
            df = pd.read_csv(self.csv_path, encoding='utf-8')
            logger.info(f"成功读取CSV文件，共 {len(df)} 条记录")
            
            if idx_start is not None and idx_end is not None:
                # 确保索引在有效范围内
                idx_start = max(0, idx_start)
                idx_end = min(len(df), idx_end)
                df = df.iloc[idx_start:idx_end]
                logger.info(f"加载索引 [{idx_start}, {idx_end}) 区间的数据，共 {len(df)} 条记录")
            elif idx_start is not None:
                # 只指定开始索引，加载到末尾
                idx_start = max(0, idx_start)
                df = df.iloc[idx_start:]
                logger.info(f"加载索引 [{idx_start}, 末尾) 区间的数据，共 {len(df)} 条记录")
            elif idx_end is not None:
                # 只指定结束索引，从开头加载
                idx_end = min(len(df), idx_end)
                df = df.iloc[:idx_end]
                logger.info(f"加载索引 [0, {idx_end}) 区间的数据，共 {len(df)} 条记录")
            else:
                # 没有指定索引，加载全部数据
                logger.info("加载全部数据")

            data_list = []
            for idx, row in df.iterrows():
                data = PoetryData(
                    idx=idx,
                    query=row['Spark'],  # 使用spark作为query
                    reference=row['Content']
                )
                data_list.append(data)
            
            logger.info(f"成功加载 {len(data_list)} 条诗词数据")
            return data_list
            
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise
    
    def format_dataset_for_grpo(self, data_list: List[PoetryData], generator) -> Dataset:
        """将数据格式化为GRPO训练格式"""
        formatted_data = []
        
        for data in data_list:
            create_prompt = generator._get_create_prompt(data.query)
            prompt = [
                {"role": "user", "content": create_prompt},
            ]
            
            formatted_data.append({
                "sample_index": data.idx,  # 添加样本索引
                "query": data.query,  # 添加原始query
                "reference": data.reference,  # 添加参考诗作
                "prompt": prompt,
            })
        
        return Dataset.from_list(formatted_data)
