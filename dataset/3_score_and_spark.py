#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诗作评分和Spark生成工具
基于chinese_poetry_curated2.csv，调用deepseek-chat大模型对诗作内容进行格律诗评分，
同时模拟诗人创作时的所见所闻所感，生成朴素平实的spark内容

使用示例:
    python 3_score_and_spark.py
    python 3_score_and_spark.py --input_csv custom_input.csv --max_concurrent 30 --temperature 0.2
    python 3_score_and_spark.py --batch_size 5000 --max_concurrent 20
"""

import os
import pandas as pd
import json
import time
import random
import numpy as np
import argparse
from typing import List, Dict, Optional
import logging
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field, ValidationError

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 屏蔽OpenAI SDK的HTTP请求日志
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

class ScoringDetails(BaseModel):
    """评分详细数据模型"""
    总分: int = Field(..., description="总分，范围0-100")
    格律规范性: int = Field(..., description="格律规范性得分")
    对仗与结构: int = Field(..., description="对仗与结构得分")
    意境与立意: int = Field(..., description="意境与立意得分")
    语言与锤炼: int = Field(..., description="语言与锤炼得分")

class Response(BaseModel):
    """评分和spark响应数据模型"""
    score: ScoringDetails = Field(..., description="详细得分信息")
    spark: str = Field(..., description="模拟诗人创作时的所见所闻所感，朴素平实版本")

class PoetryScorer:
    """诗作评分器"""
    
    def __init__(self, temperature: float = 0.1):
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("请在环境变量中设置 DEEPSEEK_API_KEY")
        
        # 使用OpenAI SDK初始化客户端，调用DeepSeek API
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = 'deepseek-chat'  # 使用deepseek-chat模型
        self.temperature = temperature
        
    def read_poetry_collection(self, csv_path: str) -> pd.DataFrame:
        """读取诗作合集CSV文件"""
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
            logger.info(f"成功读取诗作合集文件，共 {len(df)} 条记录")
            return df
        except Exception as e:
            logger.error(f"读取CSV文件失败: {e}")
            raise
    
    def score_poetry(self, content: str, title: str = "") -> Optional[Response]:
        """对单首诗作进行评分并生成spark"""

        prompt = (
            f"你是一位格律诗专家，需要同时完成两个任务：\n"
            f"\n"
            f"**任务一：诗作评分**\n"
            f"根据以下评分规则对格律诗作进行精准打分：\n"
            f"**评分规则（满分100分）：**\n"
            f"-   **格律规范性（40分）：**\n"
            f"    -   平仄： 20分。严格依照《平水韵》平仄谱。每处平仄错误扣5分（扣完20分为止）。**若某处不合平仄但构成了公认的“拗救”格，则视为正确，不扣分。**\n"
            f"    -   押韵： 20分。严格依照《平水韵》韵部，所有押韵句须用同一韵部字。每处押韵错误扣5分（扣完20分为止）。（注：首句押韵与否根据具体诗体判断，不作为错误。）\n"
            f"-   **对仗与结构（20分）：**\n"
            f"    -   律诗：颔联、颈联必须对仗。每联基础分为6分，**对仗工整、意境佳妙（如工对、流水对等），可酌情奖励1-4分，单联最高10分。每处词性不对应扣2分；若出现“合掌”（上下句意思重复），该联扣5分。（单联扣完基础分为止）\n"
            f"    -   绝句：考察“起承转合”结构。结构完整、转折自然得17-20分；结构基本合理但转折略生硬得13-16分；结构有明显缺陷得9-12分；结构混乱或无效得0-8分。\n"
            f"-   **语言与锤炼（20分）：**\n"
            f"    -   评分标准：\n"
            f"        -   18-20分：语言精炼准确，意蕴丰厚，几乎无一字可易；句式灵活多变，富有表现力与张力；音韵和谐优美，节奏感强，朗朗上口，达到炉火纯青的境地。\n"
            f"        -   15-17分：语言通顺流畅，用词较为准确，能服务于主旨表达，无明显语病；句式有一定变化，不显单调；音韵较为和谐，整体语感良好。\n"
            f"        -   12-14分：语言基本通顺，能清晰表达核心意思；但个别词语的运用或句式结构尚可推敲，偶尔出现轻微拗口或不够精炼之处。整体框架无大碍。\n"
            f"        -   8-11分：语言存在少量明显问题，如用词不当、语意重复、句式单调或存在语病；音韵不甚流畅，在一定程度上影响了阅读的美感和顺畅度。\n"
            f"        -   0-7分：语言存在严重缺陷，如词语贫乏、用词陈腐、表达不清、逻辑混乱；音韵严重失调，文理不通，阅读体验差。\n"
            f"-   **意境与立意（20分）：**\n"
            f"    -   **评分标准：**\n"
            f"        -   18-20分：立意高远，主旨深刻新颖，具有独特的思想价值或启发性；意境开阔或深邃，意象丰富且高度贴切，情景交融，浑然天成；情感真挚饱满，能引发读者强烈共鸣，余味无穷。\n"
            f"        -   15-17分：立意明确，主旨清晰且有一定深度；意境营造较为成功，画面感强，能够有力地烘托主旨；情感表达真切自然，能有效打动读者。\n"
            f"        -   12-14分：立意基本清晰，主旨表达无明显偏差；能够营造出特定意境，但可能不够突出或意象略显单一；情感表达基本真实，但感染力有限。\n"
            f"        -   8-11分：立意略显平庸或浅薄，主旨不够突出或模糊；意境营造较为勉强，或与主旨联系不紧密，意象选择不佳；情感表达较浅，缺乏深度和感染力。\n"
            f"        -   0-7分：立意不清、陈旧或存在谬误，主旨不明；未能营造有效意境，意象混乱或滥用；情感虚浮空洞，流于“无病呻吟”，无法引起共情。\n"
            f"**打分要求：** 严格打分，可扣可不扣的分必须扣分，扣分至0分为止。\n"
            f"**区分度要求：** 在进行“语言与锤炼”和“意境与立意”两项主观评分时，**你必须大胆使用整个0-20分的分数区间。** 对于真正卓越、千古流传级别的作品，应果断给予19-20分的极高分；对于平庸、有明显缺陷的作品，也必须果断给出11分以下的低分。评分的核心目标是**“拉开档次，体现差异”**，避免将分数集中在中间区域。\n"
            f"**任务二：模拟创作心路历程**\n"
            f"以第一人称视角，模仿诗人的口吻，用朴素平实的语言描述诗人创作这首诗时的所见所闻所感。\n"
            f"要求：\n"
            f"- 使用'我'作为第一人称，不要提及具体作者姓名\n"
            f"- 结合诗作内容进行合理推演，描述需包含触发创作的具体情境（如睹物、感时、送别等）和详细的内心活动（如情感波动、人生感慨、志向抒发等），使人物形象丰满。\n"
            f"- 使用简单直白的词汇，避免华丽的修辞。语言要像普通人说话一样自然，不要使用过于文雅的词汇或复杂的句式。内容长度适中，不要过长。\n"
            f"\n"
            f"**输出要求：** 你**必须且只能**输出JSON，**严禁**任何额外文本、思考过程或格式标记。\n"
            f"\n"
            f"**JSON输出格式（严格遵循）：**\n"
            f'{{\n'
            f'  "score": {{\n'
            f'    "总分": 总分,\n'
            f'    "格律规范性": 格律规范分,\n'
            f'    "对仗与结构": 对仗结构分,\n'
            f'    "语言与锤炼": 语言锤炼分,\n'
            f'    "意境与立意": 意境立意分\n'
            f'  }},\n'
            f'  "spark": "朴素平实的创作心路历程描述"\n'
            f'}}\n'
            f"\n"
            f"**诗作全文：**\n{content}"
        )
        try:
            # 调用DeepSeek API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'user', 'content': prompt}
                ],
                stream=False,
                temperature=self.temperature  # 使用可配置的温度参数
            )
            
            response_content = response.choices[0].message.content.strip()
            
            # 解析JSON响应
            try:
                response_data = json.loads(response_content)
                response_obj = Response(**response_data)
                
                # 验证分数范围
                if not (0 <= response_obj.score.总分 <= 100):
                    logger.warning(f"《{title}》总分超出范围: {response_obj.score.总分}")
                    response_obj.score.总分 = max(0, min(100, response_obj.score.总分))
                
                # logger.info(f"成功评分《{title}》: {response_obj.score.总分}分")
                return response_obj
                
            except json.JSONDecodeError as json_err:
                logger.error(f"《{title}》JSON解析失败: {json_err}")
                logger.debug(f"原始响应内容: {response_content}")
                return None
            except ValidationError as validation_err:
                logger.error(f"《{title}》数据验证失败: {validation_err}")
                logger.debug(f"解析的数据: {response_data}")
                return None
            
        except Exception as e:
            logger.error(f"评分《{title}》时发生错误: {e}")
            return None
    
    def score_all_poetry(self, df: pd.DataFrame, max_concurrent: int = 5) -> List[Dict]:
        """并发评分所有诗作"""
        results = []
        total_count = len(df)
        
        def score_single_poetry(row):
            """评分单首诗作的包装函数"""
            response_obj = self.score_poetry(
                row['Content'], row.get('Title', '')
            )
            
            # 构建包含原始数据和评分结果的完整记录
            result = {
                'Title': row.get('Title', ''),
                'Author': row.get('Author', ''),
                'Dynasty': row.get('Dynasty', ''),
                'Content': row.get('Content', ''),
                'Genre': row.get('Genre', '')
            }
            
            if response_obj:
                result.update({
                    'Score': response_obj.score.总分,
                    'Rhythm_Score': response_obj.score.格律规范性,
                    'Structure_Score': response_obj.score.对仗与结构,
                    'Meaning_Score': response_obj.score.意境与立意,
                    'Language_Score': response_obj.score.语言与锤炼,
                    'Spark': response_obj.spark
                })
            else:
                result.update({
                    'Score': 0,  # 评分失败标记为0
                    'Rhythm_Score': 0,
                    'Structure_Score': 0,
                    'Meaning_Score': 0,
                    'Language_Score': 0,
                    'Spark': '评分失败'
                })
            
            return result
        
        logger.info(f"开始并发评分 {total_count} 首诗作...")
        start_time = time.time()
        
        # 使用线程池实现并发
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # 提交所有任务
            future_to_row = {
                executor.submit(score_single_poetry, row): (index, row) 
                for index, row in df.iterrows()
            }
            
            # 收集结果
            completed_count = 0
            for future in as_completed(future_to_row):
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1
                    
                    # 每处理100条记录显示一次进度
                    if completed_count % 100 == 0:
                        elapsed_time = time.time() - start_time
                        avg_time_per_item = elapsed_time / completed_count
                        remaining_items = total_count - completed_count
                        estimated_remaining_time = remaining_items * avg_time_per_item
                        
                        logger.info(f"进度: {completed_count}/{total_count} "
                                  f"({completed_count/total_count*100:.1f}%) "
                                  f"预计剩余时间: {estimated_remaining_time/60:.1f}分钟")
                        
                except Exception as e:
                    index, row = future_to_row[future]
                    logger.error(f"评分《{row['Title']}》失败: {e}")
                    results.append({
                        'Title': row.get('Title', ''),
                        'Author': row.get('Author', ''),
                        'Dynasty': row.get('Dynasty', ''),
                        'Content': row.get('Content', ''),
                        'Genre': row.get('Genre', ''),
                        'Score': 0,
                        'Rhythm_Score': 0,
                        'Structure_Score': 0,
                        'Meaning_Score': 0,
                        'Language_Score': 0,
                        'Spark': '评分失败'
                    })
        
        end_time = time.time()
        logger.info(f"完成评分，耗时: {end_time - start_time:.2f} 秒")
        
        return results
    
    def save_to_csv(self, results: List[Dict], output_path: str):
        """保存结果为CSV格式"""
        try:
            df_result = pd.DataFrame(results)
            df_result.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"评分结果已保存到CSV文件: {output_path}")
        except Exception as e:
            logger.error(f"保存CSV文件失败: {e}")
            raise
    
    def get_statistics(self, results: List[Dict]) -> Dict:
        """获取评分统计信息"""
        scores = [r['Score'] for r in results if r['Score'] > 0]
        failed_count = len([r for r in results if r['Score'] == 0])
        
        if not scores:
            return {
                'total_count': len(results),
                'failed_count': failed_count,
                'success_count': 0,
                'avg_score': 0,
                'min_score': 0,
                'max_score': 0,
                'median_score': 0,
                'std_score': 0
            }
        
        return {
            'total_count': len(results),
            'failed_count': failed_count,
            'success_count': len(scores),
            'avg_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'median_score': np.median(scores),
            'std_score': np.std(scores)
        }
    
    def load_progress(self, progress_file: str) -> Dict:
        """加载处理进度"""
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                logger.info(f"加载进度文件: {progress}")
                return progress
            except Exception as e:
                logger.warning(f"加载进度文件失败: {e}")
        return {
            'completed_batches': [],
            'total_batches': 0,
            'total_poems': 0,
            'start_time': None
        }
    
    def save_progress(self, progress: Dict, progress_file: str):
        """保存处理进度"""
        try:
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
            logger.info(f"进度已保存到: {progress_file}")
        except Exception as e:
            logger.error(f"保存进度文件失败: {e}")
    
    def process_batch(self, df_batch: pd.DataFrame, batch_num: int, output_dir: str = '.', max_concurrent: int = 50) -> bool:
        """处理单个批次"""
        batch_output_file = os.path.join(output_dir, f'chinese_poetry_curated3_batch_{batch_num}.csv')
        
        try:
            logger.info(f"开始处理第 {batch_num} 批，共 {len(df_batch)} 首诗")
            start_time = time.time()
            
            # 评分诗作
            results = self.score_all_poetry(df_batch, max_concurrent=max_concurrent)
            
            # 保存批次结果
            self.save_to_csv(results, batch_output_file)
            
            end_time = time.time()
            logger.info(f"第 {batch_num} 批处理完成，耗时: {end_time - start_time:.2f} 秒")
            
            # 统计信息
            stats = self.get_statistics(results)
            spark_success_count = sum(1 for r in results if r.get('Spark', '') and r.get('Spark', '') != '评分失败')
            logger.info(f"第 {batch_num} 批统计: 成功评分 {stats['success_count']}, 失败 {stats['failed_count']}, 平均分 {stats['avg_score']:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"处理第 {batch_num} 批失败: {e}")
            return False
    
    def merge_batches(self, batch_files: List[str], final_output_file: str) -> bool:
        """合并所有批次结果"""
        try:
            logger.info(f"开始合并 {len(batch_files)} 个批次文件...")
            
            all_results = []
            for batch_file in batch_files:
                if os.path.exists(batch_file):
                    df_batch = pd.read_csv(batch_file, encoding='utf-8')
                    all_results.append(df_batch)
                    logger.info(f"已加载批次文件: {batch_file} ({len(df_batch)} 条记录)")
                else:
                    logger.warning(f"批次文件不存在: {batch_file}")
            
            if not all_results:
                logger.error("没有找到任何批次文件")
                return False
            
            # 合并所有结果
            df_final = pd.concat(all_results, ignore_index=True)
            
            # 保存最终结果
            df_final.to_csv(final_output_file, index=False, encoding='utf-8')
            
            logger.info(f"合并完成，最终结果已保存到: {final_output_file}")
            logger.info(f"总计 {len(df_final)} 首诗")
            
            return True
            
        except Exception as e:
            logger.error(f"合并批次文件失败: {e}")
            return False
    
    def cleanup_batch_files(self, batch_files: List[str]):
        """清理批次文件"""
        try:
            for batch_file in batch_files:
                if os.path.exists(batch_file):
                    os.remove(batch_file)
                    logger.info(f"已删除批次文件: {batch_file}")
        except Exception as e:
            logger.warning(f"清理批次文件时出错: {e}")
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='诗作评分和Spark生成工具')
    parser.add_argument('--input_csv', type=str, default='chinese_poetry_curated2.csv',
                       help='输入CSV文件路径 (默认: chinese_poetry_curated2.csv)')
    # parser.add_argument('--input_csv', type=str, default='人教版_律诗合集.csv',
    #                    help='输入CSV文件路径 (默认: chinese_poetry_curated2.csv)')
    parser.add_argument('--max_concurrent', type=int, default=100,
                       help='最大并发数 (默认: 100)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='大模型温度参数 (默认: 1.0)')
    parser.add_argument('--batch_size', type=int, default=10000,
                       help='批次大小 (默认: 10000)')
    return parser.parse_args()

def main():
    """主函数 - 分批处理版本"""
    try:
        # 解析命令行参数
        args = parse_args()
        
        # 设置随机数种子，确保复现性
        random_seed = 42
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 初始化评分器
        scorer = PoetryScorer(temperature=args.temperature)
        
        # 显示配置参数
        logger.info(f"使用配置参数:")
        logger.info(f"  输入文件: {args.input_csv}")
        logger.info(f"  最大并发数: {args.max_concurrent}")
        logger.info(f"  温度参数: {args.temperature}")
        logger.info(f"  批次大小: {args.batch_size}")
        
        # 读取诗作合集
        input_csv = args.input_csv
        df = scorer.read_poetry_collection(input_csv)
           
        # 询问用户要评分多少首诗
        total_poems = len(df)
        print(f"诗作合集中共有 {total_poems} 首诗")
        
        while True:
            try:
                n = input("请输入要评分的前n首诗数量（输入0表示全部评分）: ").strip()
                if n == '0':
                    n = total_poems
                    break
                n = int(n)
                if n > 0 and n <= total_poems:
                    break
                else:
                    print(f"请输入1到{total_poems}之间的数字")
            except ValueError:
                print("请输入有效的数字")
        
        # 选取前n首诗
        if n < total_poems:
            df = df.head(n)
        
        # 分批处理参数
        batch_size = args.batch_size
        total_poems_to_process = len(df)
        total_batches = (total_poems_to_process + batch_size - 1) // batch_size
        
        # 进度文件路径
        progress_file = f'chinese_poetry_curated3_progress.json'
        final_output_file = f'chinese_poetry_curated3_all.csv'
        
        # 加载进度
        progress = scorer.load_progress(progress_file)
        
        # 检查是否需要重新开始
        if progress['total_batches'] != total_batches or progress['total_poems'] != total_poems_to_process:
            logger.info("检测到新的处理任务，重新开始...")
            progress = {
                'completed_batches': [],
                'total_batches': total_batches,
                'total_poems': total_poems_to_process,
                'start_time': time.time()
            }
            scorer.save_progress(progress, progress_file)
        
        logger.info(f"开始分批处理，共 {total_batches} 批，每批最多 {batch_size} 首诗")
        logger.info(f"已完成批次: {len(progress['completed_batches'])}/{total_batches}")
        
        # 处理每个批次
        for batch_num in range(1, total_batches + 1):
            if batch_num in progress['completed_batches']:
                logger.info(f"跳过已完成的第 {batch_num} 批")
                continue
            
            # 计算当前批次的起始和结束位置
            start_idx = (batch_num - 1) * batch_size
            end_idx = min(start_idx + batch_size, total_poems_to_process)
            
            # 获取当前批次的数据
            df_batch = df.iloc[start_idx:end_idx].copy()
            
            logger.info(f"准备处理第 {batch_num}/{total_batches} 批 (第 {start_idx+1}-{end_idx} 首诗)")
            
            # 处理当前批次
            success = scorer.process_batch(df_batch, batch_num, max_concurrent=args.max_concurrent)
            
            if success:
                # 更新进度
                progress['completed_batches'].append(batch_num)
                scorer.save_progress(progress, progress_file)
                
                # 显示总体进度
                completed_count = len(progress['completed_batches'])
                elapsed_time = time.time() - progress['start_time']
                avg_time_per_batch = elapsed_time / completed_count
                remaining_batches = total_batches - completed_count
                estimated_remaining_time = remaining_batches * avg_time_per_batch
                
                logger.info(f"总体进度: {completed_count}/{total_batches} "
                          f"({completed_count/total_batches*100:.1f}%) "
                          f"预计剩余时间: {estimated_remaining_time/3600:.1f}小时")
            else:
                logger.error(f"第 {batch_num} 批处理失败，程序停止")
                logger.info("您可以重新运行程序继续处理剩余批次")
                return
        
        # 所有批次处理完成，合并结果
        logger.info("所有批次处理完成，开始合并结果...")
        
        batch_files = [f'chinese_poetry_curated3_batch_{i}.csv' for i in range(1, total_batches + 1)]
        merge_success = scorer.merge_batches(batch_files, final_output_file)
        
        if merge_success:
            # 清理批次文件
            logger.info("清理临时批次文件...")
            scorer.cleanup_batch_files(batch_files)
            
            # 删除进度文件
            if os.path.exists(progress_file):
                os.remove(progress_file)
                logger.info(f"已删除进度文件: {progress_file}")
            
            # 最终统计
            df_final = pd.read_csv(final_output_file, encoding='utf-8')
            results = df_final.to_dict('records')
            stats = scorer.get_statistics(results)
            spark_success_count = sum(1 for r in results if r.get('Spark', '') and r.get('Spark', '') != '评分失败')
            
            logger.info("=== 分批处理完成统计 ===")
            logger.info(f"总诗作数量: {stats['total_count']}")
            logger.info(f"成功评分: {stats['success_count']}")
            logger.info(f"评分失败: {stats['failed_count']}")
            logger.info(f"成功生成Spark: {spark_success_count}")
            logger.info(f"平均分数: {stats['avg_score']:.2f}")
            logger.info(f"中位数分数: {stats['median_score']:.2f}")
            logger.info(f"标准差: {stats['std_score']:.2f}")
            logger.info(f"最高分数: {stats['max_score']}")
            logger.info(f"最低分数: {stats['min_score']}")
            logger.info(f"最终结果已保存到: {final_output_file}")
        else:
            logger.error("合并批次文件失败")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise

if __name__ == "__main__":
    main()
