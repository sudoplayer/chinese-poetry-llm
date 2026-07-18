#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诗歌体裁严格分类工具
基于chinese_poetry_curated1.csv，调用deepseek-chat大模型对诗歌进行严格体裁划分
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

class GenreClassification(BaseModel):
    """体裁分类数据模型"""
    体裁: str = Field(..., description="诗歌体裁分类（五绝、七绝、五律、七律、古风、乐府、歌行、其他等）")
    分类依据: str = Field(..., description="分类依据说明（小于20字）")


class PoetryGenreClassifier:
    """诗词体裁分类器"""
    
    def __init__(self):
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("请在环境变量中设置 DEEPSEEK_API_KEY")
        
        # 使用OpenAI SDK初始化客户端，调用DeepSeek API
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = 'deepseek-chat'  # 使用deepseek-chat模型
        
    def read_poetry_collection(self, csv_path: str) -> pd.DataFrame:
        """读取诗歌合集CSV文件"""
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
            logger.info(f"成功读取诗歌合集文件，共 {len(df)} 条记录")
            return df
        except Exception as e:
            logger.error(f"读取CSV文件失败: {e}")
            raise
    
    def classify_genre(self, title: str, author: str, content: str) -> Optional[Dict]:
        """对单首诗词进行体裁分类"""
        
        prompt = (
            f"请严格按照以下要求分类诗歌体裁：\n"
            f"分类规则：五绝(4句5言)、七绝(4句7言)、五律(8句5言)、七律(8句7言)、古风(自由体)、乐府(民歌)、歌行(叙事)、其他\n"
            f"输出要求：必须返回标准JSON格式，不要包含任何其他文字\n"
            f"JSON格式：{{\"体裁\":\"类别\",\"依据\":\"说明(20字内)\"}}\n"
            f"诗歌内容：{title} {author}\n{content}\n"
            f"请直接输出JSON："
        )
        try:
            # 调用DeepSeek API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'user', 'content': prompt}
                ],
                stream=False,
                temperature=1.0
            )
            
            response_content = response.choices[0].message.content.strip()
            
            # 解析JSON响应
            try:
                response_data = json.loads(response_content)
                return response_data
                
            except json.JSONDecodeError as json_err:
                logger.error(f"《{title}》JSON解析失败: {json_err}")
                logger.debug(f"原始响应内容: {response_content}")
                return None
            
        except Exception as e:
            logger.error(f"分类《{title}》时发生错误: {e}")
            return None
    
    def classify_all_poetry(self, df: pd.DataFrame, max_concurrent: int = 5) -> List[Dict]:
        """并发分类所有诗歌"""
        results = []
        total_count = len(df)
        
        def classify_single_poetry(row):
            """分类单首诗词的包装函数"""
            genre_response = self.classify_genre(
                row.get('Title', ''),
                row.get('Author', ''),
                row.get('Content', '')
            )
            
            # 构建包含原始数据和分类结果的完整记录
            result = {
                'Title': row.get('Title', ''),
                'Author': row.get('Author', ''),
                'Dynasty': row.get('Dynasty', ''),
                'Content': row.get('Content', ''),
            }
            
            if genre_response:
                result.update({
                    'Genre': genre_response.get('体裁', '未知')
                })
            else:
                result.update({
                    'Genre': '分类失败'
                })
            
            return result
        
        logger.info(f"开始并发分类 {total_count} 首诗歌...")
        start_time = time.time()
        
        # 使用线程池实现并发
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # 提交所有任务
            future_to_row = {
                executor.submit(classify_single_poetry, row): (index, row) 
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
                    logger.error(f"分类《{row['Title']}》失败: {e}")
                    results.append({
                        'Title': row.get('Title', ''),
                        'Author': row.get('Author', ''),
                        'Dynasty': row.get('Dynasty', ''),
                        'Content': row.get('Content', ''),
                        'Genre': '分类失败'
                    })
        
        end_time = time.time()
        logger.info(f"完成分类，耗时: {end_time - start_time:.2f} 秒")
        
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
        """获取分类统计信息"""
        successful_classifications = [r for r in results if r['Genre'] != '分类失败']
        failed_count = len([r for r in results if r['Genre'] == '分类失败'])
        
        # 统计各体裁数量
        genre_counts = {}
        
        for result in successful_classifications:
            genre = result['Genre']
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        return {
            'total_count': len(results),
            'failed_count': failed_count,
            'success_count': len(successful_classifications),
            'genre_distribution': genre_counts
        }
    
    def extract_rhythmic_poems(self, results: List[Dict]) -> List[Dict]:
        """提取格律诗（五绝、七绝、五律、七律）"""
        rhythmic_genres = ['五绝', '七绝', '五律', '七律']
        rhythmic_poems = [r for r in results if r['Genre'] in rhythmic_genres]
        logger.info(f"从 {len(results)} 首诗中提取出 {len(rhythmic_poems)} 首格律诗")
        return rhythmic_poems

def main():
    """主函数"""
    try:
        # 解析命令行参数
        parser = argparse.ArgumentParser(description='诗词体裁分类工具')
        parser.add_argument('-n', '--number', type=int, default=0, 
                          help='要分类的诗歌数量（默认100首，0表示分类所有诗歌）')
        parser.add_argument('-i', '--input', type=str, default='chinese_poetry_curated1.csv',
                          help='输入CSV文件路径')
        parser.add_argument('-c', '--concurrent', type=int, default=50,
                          help='并发数量（默认50）')
        
        args = parser.parse_args()
        
        # 设置随机数种子，确保复现性
        random_seed = 42
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 初始化分类器
        classifier = PoetryGenreClassifier()
        
        # 读取诗词合集
        df = classifier.read_poetry_collection(args.input)
        
        # 如果指定了N，则只处理前N首诗
        if args.number > 0:
            df = df.head(args.number)
            logger.info(f"只处理前 {args.number} 首诗，共 {len(df)} 首")
        else:
            logger.info(f"处理所有诗歌，共 {len(df)} 首")
        
        logger.info(f"开始对 {len(df)} 首诗进行体裁分类...")
        
        # 分类诗词
        results = classifier.classify_all_poetry(df, max_concurrent=args.concurrent)
        
        # 保存所有分类结果
        all_output_csv = 'chinese_poetry_curated2_all.csv'
        classifier.save_to_csv(results, all_output_csv)
        
        # 提取格律诗
        rhythmic_poems = classifier.extract_rhythmic_poems(results)
        
        # 保存格律诗结果
        rhythmic_output_csv = 'chinese_poetry_curated2.csv'
        classifier.save_to_csv(rhythmic_poems, rhythmic_output_csv)
        
        # 统计信息
        stats = classifier.get_statistics(results)
        logger.info("=== 体裁分类完成统计 ===")
        logger.info(f"总诗歌数量: {stats['total_count']}")
        logger.info(f"成功分类: {stats['success_count']}")
        logger.info(f"分类失败: {stats['failed_count']}")
        logger.info(f"格律诗数量: {len(rhythmic_poems)}")
        logger.info(f"格律诗比例: {len(rhythmic_poems)/stats['success_count']*100:.1f}%")
        
        logger.info("=== 体裁分布 ===")
        for genre, count in sorted(stats['genre_distribution'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"{genre}: {count} 首")
        
        logger.info(f"所有分类结果已保存: {all_output_csv}")
        logger.info(f"格律诗结果已保存: {rhythmic_output_csv}")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise

if __name__ == "__main__":
    main()
