#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诗词生成模块
负责使用模型生成诗词内容
支持本地模型和DeepSeek API两种模式
"""

import os
import logging
import asyncio
from typing import Dict, List, Tuple, Optional
from openai import AsyncOpenAI

# 设置日志
logger = logging.getLogger(__name__)

def _get_create_prompt(query: str) -> str:
    """创建提示词（共享函数）"""
    prompt = (
        f"你是一名精通格律诗创作的诗人。请仔细阅读并深入体会下方【所见所闻所感】中描述的情境与情感，以此为核心灵感，创作一首符合以下要求的格律诗：\n\n"
        f"**诗歌创作要求：**\n"
        f"1. **格律规范**：严格依照《平水韵》的平仄谱，并押**平声韵**。\n"
        f"3. **对仗工整**：若为律诗，**颔联、颈联必须对仗工稳**。若为绝句，结构需体现完整的\"起承转合\"。\n"
        f"4. **语言精炼**：用词精准凝练，**重点锤炼动词与形容词**以打造\"诗眼\"，避免生硬凑韵。\n"
        f"5. **意境营造**：**选取2-3个核心意象**构建统一画面，情感表达要真切自然，通过诗句的递进自然流露，避免直白说教。\n\n"
        f"**输出格式要求：**\n"
        f"- **仅输出诗作正文**，每句一行。\n"
        f"- 严禁添加标题、作者署名、注释等任何额外文字。\n\n"
        f"【所见所闻所感】：\n{query}"
    )
    return prompt


class PoetryGenerator:
    """诗词生成器"""
    
    def __init__(self):
        """初始化生成器"""
        pass
    
    def _get_create_prompt(self, query: str) -> str:
        """创建提示词"""
        return _get_create_prompt(query)


class DeepSeekGenerator:
    """DeepSeek API诗词生成器"""
    
    def __init__(self, max_concurrent_requests: int = 10):
        """初始化DeepSeek生成器"""
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("请在环境变量中设置 DEEPSEEK_API_KEY")
        
        # 初始化DeepSeek异步客户端
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        
        # 批量生成配置
        self.max_concurrent_requests = max_concurrent_requests
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
    
    async def _generate_single_poetry_async(self, query: str) -> str:
        """异步生成单首诗词"""
        prompt = _get_create_prompt(query)
        
        async with self.semaphore:
            try:
                # 使用DeepSeek异步客户端
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                    temperature=1.5,
                    max_tokens=512
                )
                
                generated_text = response.choices[0].message.content.strip()
                return generated_text
                
            except Exception as e:
                logger.error(f"DeepSeek API调用异常: {e}")
                return ""
    
    async def _generate_poetry_batch_async(self, queries: List[str]) -> List[str]:
        """异步批量生成诗词"""
        if not queries:
            return []
        
        # 创建所有生成任务
        tasks = [
            self._generate_single_poetry_async(query)
            for query in queries
        ]
        
        # 并发执行所有任务
        results = await asyncio.gather(*tasks)
        
        return results
    
    def generate_poetry_batch(self, queries: List[str]) -> List[str]:
        """批量生成诗词（同步接口）"""
        if not queries:
            return []
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # 运行异步批量生成
        return loop.run_until_complete(self._generate_poetry_batch_async(queries))
    
