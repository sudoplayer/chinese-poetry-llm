#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诗歌体裁粗分类脚本
读取chinese_poetry_simplified.csv，分析每首诗的句数和字数，粗筛体裁
"""

import pandas as pd
import re
from typing import List, Tuple

def analyze_poetry_content(content: str) -> Tuple[int, str, str]:
    """
    分析诗歌内容，返回句数、每句字数和体裁粗分类
    
    Args:
        content: 诗歌内容字符串
        
    Returns:
        tuple: (句数, 每句字数字符串, 体裁粗分类)
    """
    if pd.isna(content) or not content.strip():
        return 0, "", "其它"
    
    # 按逗号和句号分割句子
    sentences = re.split(r'[，。]', content.strip())
    # 过滤空字符串
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return 0, "", "其它"
    
    sentence_count = len(sentences)
    
    # 统计每句的字数（去除标点符号）
    char_counts = []
    for sentence in sentences:
        # 去除所有标点符号，只保留汉字
        clean_sentence = re.sub(r'[^\u4e00-\u9fff]', '', sentence)
        char_counts.append(len(clean_sentence))
    
    char_counts_str = ','.join(map(str, char_counts))
    
    # 体裁分类
    genre = classify_genre(sentence_count, char_counts)
    
    return sentence_count, char_counts_str, genre

def classify_genre(sentence_count: int, char_counts: List[int]) -> str:
    """
    根据句数和字数粗类诗歌体裁
    
    Args:
        sentence_count: 句数
        char_counts: 每句字数列表
        
    Returns:
        str: 体裁分类（绝句/律诗/其它）
    """
    # 检查是否所有句子字数相同
    if not char_counts:
        return "其它"
    
    unique_char_counts = set(char_counts)
    
    # 绝句：4句，每句5字或7字
    if sentence_count == 4 and len(unique_char_counts) == 1 and char_counts[0] in [5, 7]:
        return "绝句"
    
    # 律诗：8句，每句5字或7字
    elif sentence_count == 8 and len(unique_char_counts) == 1 and char_counts[0] in [5, 7]:
        return "律诗"
    
    # 其它情况
    else:
        return "其它"

def main():
    """主函数"""
    print("开始处理诗歌数据...")
    
    # 读取CSV文件
    try:
        df = pd.read_csv('chinese_poetry_simplified.csv', encoding='utf-8')
        print(f"成功读取数据，共 {len(df)} 行")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    # 分析每首诗的Content
    print("正在分析诗歌内容...")
    
    results = []
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"已处理 {idx} 行...")
        
        sentence_count, char_counts_str, genre = analyze_poetry_content(row['Content'])
        results.append({
            'Sentence_Count': sentence_count,
            'Characters_Per_Sentence': char_counts_str,
            'Genre': genre
        })
    
    # 添加新列到DataFrame
    df['Genre'] = [r['Genre'] for r in results]
    
    # 只保留绝句和律诗
    print("过滤数据，只保留绝句和律诗...")
    df_filtered = df[df['Genre'].isin(['绝句', '律诗'])]
    print(f"过滤后剩余 {len(df_filtered)} 首诗歌")
    
    # 删除Genre列
    df_filtered = df_filtered.drop('Genre', axis=1)
    
    # 保存结果
    output_file = 'chinese_poetry_curated1.csv'
    try:
        df_filtered.to_csv(output_file, index=False, encoding='utf-8')
        print(f"结果已保存到 {output_file}")
    except Exception as e:
        print(f"保存文件失败: {e}")
        return
    
    print(f"\n处理完成！共处理 {len(df)} 首诗歌，保留 {len(df_filtered)} 首绝句和律诗")

if __name__ == "__main__":
    main()
