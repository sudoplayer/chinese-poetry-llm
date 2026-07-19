#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诗词评价模块
负责调用DeepSeek API进行诗词评分
支持批量并发评分以提升效率
"""

import os
import json
import re
import logging
import asyncio
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ValidationError
from openai import AsyncOpenAI

# 设置日志
logger = logging.getLogger(__name__)

class DetailedReview(BaseModel):
    """详细评审数据模型"""
    格律规范性评语: str = Field(..., description="格律规范性评语")
    对仗与结构评语: str = Field(..., description="对仗与结构评语")
    语言与锤炼评语: str = Field(..., description="语言与锤炼评语")
    意境与立意评语: str = Field(..., description="意境与立意评语")

class ScoringDetails(BaseModel):
    """评分详细数据模型"""
    格律规范性: int = Field(..., description="格律规范性得分")
    对仗与结构: int = Field(..., description="对仗与结构得分")
    语言与锤炼: int = Field(..., description="语言与锤炼得分")
    意境与立意: int = Field(..., description="意境与立意得分")
    总分: int = Field(..., description="总分，范围1-100")

class ScoringResponse(BaseModel):
    """评分响应数据模型"""
    体裁判断: str = Field(..., description="诗歌体裁判断")
    详细评审: DetailedReview = Field(..., description="详细评审信息")
    得分: ScoringDetails = Field(..., description="详细得分信息")

class PoetryEvaluator:
    """诗词评价器"""
    
    # 默认评分详情常量
    DEFAULT_SCORING_DETAILS = ScoringDetails(
        格律规范性=0, 对仗与结构=0, 
        语言与锤炼=0, 意境与立意=0,总分=0
    )
    
    # 默认详细评审常量
    DEFAULT_DETAILED_REVIEW = DetailedReview(
        格律规范性评语="评分失败",
        对仗与结构评语="评分失败", 
        语言与锤炼评语="评分失败",
        意境与立意评语="评分失败"
    )
    
    def __init__(self, max_concurrent_requests: int = 10):
        """初始化评价器"""
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("请在环境变量中设置 DEEPSEEK_API_KEY")
        
        # 初始化DeepSeek异步客户端
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        
        # 批量评分配置
        self.max_concurrent_requests = max_concurrent_requests
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
    
    def _get_score_prompt(self, content: str) -> str:
        """获取详细的评分标准和扣分细则"""

        prompt = (
            f"你是一位格律诗专家，职责是根据以下评分规则对格律诗作进行精准打分：\n"
            f"**评分规则（满分100分）：**\n"
            f"-   **体裁判断：** 仅依句数和每句字数判断体裁（支持：五绝、七绝、五律、七律）。**体裁不符则'格律规范性'项计0分。**\n"
            f"-   **格律规范性（40分）：**\n"
            f"    -   平仄： 20分。严格依照《平水韵》平仄谱。每处平仄错误扣5分（扣完20分为止）。**若某处不合平仄但构成了公认的“拗救”格，则视为正确，不扣分。**\n"
            f"    -   押韵： 20分。严格依照《平水韵》韵部，所有押韵句须用同一韵部字。每处押韵错误扣5分（扣完20分为止）。（注：首句押韵与否根据具体诗体判断，不作为错误。）\n"
            f"-   **对仗与结构（20分）：**\n"
            f"    -   律诗：颔联、颈联必须对仗。每联基础分为6分，**对仗工整、意境佳妙（如工对、流水对等），可酌情奖励1-4分，单联最高10分。每处词性不对应扣2分；若出现“合掌”（上下句意思重复），该联扣5分。（单联扣完基础分为止）\n"
            f"    -   绝句：考察“起承转合”结构。结构完整、转折自然得17-20分；结构基本合理但转折略生硬得13-16分；结构有明显缺陷得9-12分；结构混乱或无效得0-8分。\n"
            f"-   **语言与锤炼（20分）：**\n"
            f"-   评分标准：\n"
            f"    -   18-20分：语言精炼准确，意蕴丰厚，几乎无一字可易；句式灵活多变，富有表现力与张力；音韵和谐优美，节奏感强，朗朗上口，达到炉火纯青的境地。\n"
            f"    -   15-17分：语言通顺流畅，用词较为准确，能服务于主旨表达，无明显语病；句式有一定变化，不显单调；音韵较为和谐，整体语感良好。\n"
            f"    -   12-14分：语言基本通顺，能清晰表达核心意思；但个别词语的运用或句式结构尚可推敲，偶尔出现轻微拗口或不够精炼之处。整体框架无大碍。\n"
            f"    -   8-11分：语言存在少量明显问题，如用词不当、语意重复、句式单调或存在语病；音韵不甚流畅，在一定程度上影响了阅读的美感和顺畅度。\n"
            f"    -   0-7分：语言存在严重缺陷，如词语贫乏、用词陈腐、表达不清、逻辑混乱；音韵严重失调，文理不通，阅读体验差。\n"
            f"-   **意境与立意（20分）：**\n"
            f"-   **评分标准：**\n"
            f"    -   18-20分：立意高远，主旨深刻新颖，具有独特的思想价值或启发性；意境开阔或深邃，意象丰富且高度贴切，情景交融，浑然天成；情感真挚饱满，能引发读者强烈共鸣，余味无穷。\n"
            f"    -   15-17分：立意明确，主旨清晰且有一定深度；意境营造较为成功，画面感强，能够有力地烘托主旨；情感表达真切自然，能有效打动读者。\n"
            f"    -   12-14分：立意基本清晰，主旨表达无明显偏差；能够营造出特定意境，但可能不够突出或意象略显单一；情感表达基本真实，但感染力有限。\n"
            f"    -   8-11分：立意略显平庸或浅薄，主旨不够突出或模糊；意境营造较为勉强，或与主旨联系不紧密，意象选择不佳；情感表达较浅，缺乏深度和感染力。\n"
            f"    -   0-7分：立意不清、陈旧或存在谬误，主旨不明；未能营造有效意境，意象混乱或滥用；情感虚浮空洞，流于“无病呻吟”，无法引起共情。\n"
            f"**打分要求：** 严格打分，可扣可不扣的分必须扣分，扣分至0分为止。注：所有扣分项中'每处'指一个独立错误的最小单位（如：单个字平仄错误、单个词词性错误）。\n"
            f"**输出要求：** 你**必须且只能**输出JSON，**严禁**任何额外文本、思考过程或格式标记。\n"
            f"\n"
            f"**JSON输出格式（严格遵循）：**\n"
            f'{{\n'
            f'  "体裁判断": "体裁名称",\n'
            f'  "详细评审": {{\n'
            f'    "格律规范性评语": "语言务必精炼。平仄：仅列出错误，格式为【句号,字号,字,错误原因】，多处错误用分号隔开；若无错误，则仅输出“平仄合律”。押韵：仅指出韵部和韵脚，若有出韵，格式为【出韵字,所在句号】，若无错误，则仅输出“押韵合规，[韵部]”，例如“押韵合规，【一东】”。",\n'
            f'    "对仗与结构评语": "【律诗】：仅报告问题或亮点。格式：【颔联/颈联：问题/亮点】，例如“颔联：词性不对(天/落)；颈联：流水对，佳。”；若无问题，输出“对仗工整”。【绝句】：用不超过15字的短语评价起承转合的质量，例如“起承转合流畅，转折有力。”。",\n'
            f'    "语言与锤炼评语": "不超过25字的核心评语，直接点出最主要的优缺点，并以此作为打分依据。例如：“炼字精准，佳句在颈联。”或“部分词语略显陈旧。”。",\n'
            f'    "意境与立意评语": "不超过25字的核心评语，概括意境或主旨的水平，并以此作为打分依据。例如：“意境开阔，情景交融。”或“立意较浅，感染力不足。”。"\n'
            f'  }},\n'
            f'  "得分": {{\n'
            f'    "格律规范性": 格律规范分(0-40),\n'
            f'    "对仗与结构": 对仗结构分(0-20),\n'
            f'    "语言与锤炼": 语言锤炼分(0-20),\n'
            f'    "意境与立意": 意境立意分(0-20),\n'
            f'    "总分": 总分(0-100)\n'
            f'  }},\n'  
            f'}}\n'
            f"\n"
            f"**诗作全文：**\n{content}"
        )
        return prompt

    @staticmethod
    def _sanitize_json_text(text: str) -> str:
        """去除 JSON 字符串中的非法控制字符"""
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)

    def _parse_scoring_response(self, response_text: str) -> Optional[ScoringResponse]:
        """解析 API 返回的评分 JSON，失败返回 None"""
        sanitized = self._sanitize_json_text(response_text.strip())
        try:
            return ScoringResponse(**json.loads(sanitized))
        except (json.JSONDecodeError, ValidationError):
            return None

    def _build_failed_result(self, reason: str) -> Dict:
        return {
            "体裁判断": reason,
            "详细评审": self.DEFAULT_DETAILED_REVIEW,
            "得分": self.DEFAULT_SCORING_DETAILS,
        }

    async def _call_scoring_api(self, score_prompt: str, temperature: float) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": score_prompt}],
            stream=False,
            temperature=temperature,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
        )
        return response.choices[0].message.content.strip()
    
    async def _evaluate_single_poetry_async(self, poetry_content: str) -> Dict:
        """异步评价单首诗词"""
        poetry_text = poetry_content
        if not poetry_text:
            logger.warning("诗作内容为空")
            return self._build_failed_result("诗作内容为空")
        
        score_prompt = self._get_score_prompt(poetry_text)
        temperatures = [0.4, 0.1, 0.1]
        
        async with self.semaphore:
            for attempt, temperature in enumerate(temperatures):
                try:
                    response_text = await self._call_scoring_api(score_prompt, temperature)
                    scoring_response = self._parse_scoring_response(response_text)
                    if scoring_response is None:
                        if attempt < len(temperatures) - 1:
                            logger.warning(
                                f"JSON解析失败，重试 {attempt + 1}/{len(temperatures) - 1}: "
                                f"{response_text[:200]!r}"
                            )
                            continue
                        logger.error(f"JSON解析失败: {response_text[:200]!r}")
                        return self._build_failed_result("JSON解析失败")

                    if not (0 <= scoring_response.得分.总分 <= 100):
                        logger.warning(f"总分超出范围: {scoring_response.得分.总分}")
                        scoring_response.得分.总分 = max(0, min(100, scoring_response.得分.总分))

                    return {
                        "体裁判断": scoring_response.体裁判断,
                        "详细评审": scoring_response.详细评审,
                        "得分": scoring_response.得分,
                    }

                except Exception as e:
                    logger.error(f"API调用异常: {e}")
                    return self._build_failed_result("API调用异常")
    
    async def _evaluate_poetry_batch_async(self, poetry_contents: List[str]) -> List[Dict]:
        """异步批量评价诗词"""
        if not poetry_contents:
            return []
        
        # 创建所有评分任务
        tasks = [
            self._evaluate_single_poetry_async(content)
            for content in poetry_contents
        ]
        
        # 并发执行所有任务
        results = await asyncio.gather(*tasks)
        
        return results
    
    def evaluate_poetry_batch(self, poetry_contents: List[str]) -> List[Dict]:
        """批量评价诗词（同步接口）"""
        if not poetry_contents:
            return []
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # 运行异步批量评分
        return loop.run_until_complete(self._evaluate_poetry_batch_async(poetry_contents))

    def _check_poetry_format(self, response: str) -> bool:
        """检查诗词格式是否符合格律诗要求"""
        # 使用逗号和句号分割
        sentences = re.split(r'[，。]', response)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        sentence_count = len(sentences)
        total_chars = len(''.join(sentences))
        
        # 检查是否符合格律诗要求
        if sentence_count == 4:  # 绝句
            return total_chars in [20, 28]  # 五绝20字，七绝28字
        elif sentence_count == 8:  # 律诗
            return total_chars in [40, 56]  # 五律40字，七律56字
        else:
            return False  # 句数不符合要求

    def create_reward_functions(self, grpo_config):
        """创建奖励函数

        Args:
            grpo_config: GRPO 配置对象，需包含 PER_DEVICE_TRAIN_BATCH_SIZE 与 NUM_GENERATIONS
        """
        from poetry_core.poetry_logger import get_global_logger
        
        def content_quality_reward(completions, **kwargs):
            """内容质量奖励函数"""
            scores = []
            
            # 批次处理：为每个样本分别处理
            sample_indices = kwargs['sample_index']
            queries = kwargs['query']
            references = kwargs['reference']

            batch_size = int(grpo_config.PER_DEVICE_TRAIN_BATCH_SIZE / grpo_config.NUM_GENERATIONS)
            completions_per_sample = grpo_config.NUM_GENERATIONS
            
            batch_log_data = []
            for batch_idx in range(batch_size):
                sample_index = sample_indices[batch_idx*completions_per_sample]
                query = queries[batch_idx*completions_per_sample]
                reference = references[batch_idx*completions_per_sample]
                
                # 获取当前样本的completions
                start_idx = batch_idx * completions_per_sample
                end_idx = start_idx + completions_per_sample
                sample_completions = completions[start_idx:end_idx]
                
                # 提取当前样本的completion内容
                completion_contents = []
                for completion in sample_completions:
                    response = completion[0]["content"]
                    completion_contents.append(response)
                
                # 为当前样本评分
                try:
                    evaluation_results = self.evaluate_poetry_batch(completion_contents)
                    
                    sample_completions_data = []
                    sample_scores = []
                    
                    for i, evaluation_result in enumerate(evaluation_results):
                        # 将评分转换为奖励分数（0-10分）
                        score = evaluation_result['得分'].总分 / 10.0
                        sample_scores.append(score)
                        
                        # 收集completion数据用于日志记录
                        # 将Pydantic对象转换为dict
                        evaluation_dict = {
                            "体裁判断": evaluation_result['体裁判断'],
                            "详细评审": evaluation_result['详细评审'].model_dump(),
                            "得分": evaluation_result['得分'].model_dump(),
                        }
                        
                        sample_completions_data.append({
                            "content": completion_contents[i],
                            "evaluation_result": evaluation_dict
                        })
                        
                except Exception as e:
                    logger.error(f"样本 {sample_index} 评分失败: {e}")
                    # 如果评分失败，为当前样本的所有completion设置默认分数
                    default_evaluation_dict = {
                        "体裁判断": "评分失败",
                        "详细评审": self.DEFAULT_DETAILED_REVIEW.model_dump(),
                        "得分": self.DEFAULT_SCORING_DETAILS.model_dump(),
                    }
                    sample_completions_data = []
                    sample_scores = []
                    for completion in sample_completions:
                        sample_scores.append(0.0)
                        sample_completions_data.append({
                            "content": completion[0]["content"],
                            "evaluation_result": default_evaluation_dict
                        })
                
                # 收集当前样本的分数
                scores.extend(sample_scores)
                
                # 收集当前样本的日志数据
                batch_log_data.append({
                    "sample_index": sample_index,
                    "query": query,
                    "reference": reference,
                    "completions_data": sample_completions_data
                })
            
            # 批量保存日志
            training_logger = get_global_logger()
            if training_logger:
                training_logger.log_batch_samples(batch_log_data)
            
            return scores
        
        def format_reward(completions, **kwargs):
            """格式检查奖励函数"""
            scores = []
            
            for completion in completions:
                response = completion[0]["content"]
                
                # 检查格式
                if self._check_poetry_format(response):
                    score = 0  # 格式符合要求
                else:
                    score = -10  # 格式不符合要求
                
                scores.append(score)
            
            return scores
            
        return [content_quality_reward, format_reward]
