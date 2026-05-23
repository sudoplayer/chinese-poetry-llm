#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诗作数据分析和选择工具
读取chinese_poetry_curated3_all.csv，按评分降序排列并分析Score分布

使用示例:
    python 4_analyze_and_select.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Tuple

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """加载CSV数据文件"""
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {len(df)} records")
        logger.info(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def sort_data(df: pd.DataFrame) -> pd.DataFrame:
    """按评分字段降序排列数据"""
    logger.info("Sorting data by score fields in descending order")
    
    # 定义排序字段
    sort_columns = ['Score', 'Rhythm_Score', 'Structure_Score', 'Language_Score', 'Meaning_Score']
    
    # 检查字段是否存在
    missing_columns = [col for col in sort_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing columns: {missing_columns}")
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # 按多列降序排序
    df_sorted = df.sort_values(by=sort_columns, ascending=False)
    logger.info("Data sorting completed")
    
    return df_sorted

def save_sorted_data(df: pd.DataFrame, output_path: str) -> None:
    """保存排序后的数据"""
    logger.info(f"Saving sorted data to {output_path}")
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved {len(df)} records to {output_path}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

def extract_top_poems(df: pd.DataFrame, max_count: int = 51000) -> pd.DataFrame:
    """提取前N首诗作"""
    logger.info(f"Extracting top {max_count} poems")
    
    # 取前N条
    if len(df) > max_count:
        top_df = df.head(max_count)
        logger.info(f"Selected top {max_count} poems from {len(df)} total poems")
    else:
        top_df = df
        logger.info(f"All {len(df)} poems selected (less than {max_count})")
    
    return top_df

def save_top_poems(df: pd.DataFrame, output_path: str) -> None:
    """保存高分诗作"""
    logger.info(f"Saving top poems to {output_path}")
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved {len(df)} top poems to {output_path}")
        
        # 输出统计信息
        if len(df) > 0:
            logger.info(f"Top poems Score range: {df['Score'].min():.2f} - {df['Score'].max():.2f}")
            logger.info(f"Top poems average Score: {df['Score'].mean():.2f}")
    except Exception as e:
        logger.error(f"Error saving top poems: {e}")
        raise

def analyze_score_distribution(df: pd.DataFrame) -> Tuple[np.ndarray, dict]:
    """分析Score字段的分布情况"""
    logger.info("Analyzing Score distribution")
    
    score_data = df['Score'].dropna()
    
    # 计算基本统计信息
    stats = {
        'count': len(score_data),
        'mean': score_data.mean(),
        'std': score_data.std(),
        'min': score_data.min(),
        'max': score_data.max(),
        'median': score_data.median()
    }
    
    # 计算分位数
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    for q in quantiles:
        stats[f'q{int(q*100)}'] = score_data.quantile(q)
    
    logger.info("Score distribution analysis completed")
    return score_data.values, stats

def plot_score_histogram(score_data: np.ndarray, output_path: str) -> None:
    """绘制Score分布直方图"""
    logger.info("Creating Score distribution histogram")
    
    # 确保输出目录存在
    import os
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
    
    plt.figure(figsize=(12, 8))
    
    # 绘制直方图
    plt.hist(score_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    
    # 添加统计线
    plt.axvline(np.mean(score_data), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(score_data):.2f}')
    plt.axvline(np.median(score_data), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(score_data):.2f}')
    
    # 设置图表属性
    plt.xlabel('Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Poetry Scores', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Histogram saved to {output_path}")

def print_statistics(stats: dict) -> None:
    """打印统计信息"""
    logger.info("Score Distribution Statistics:")
    print("\n" + "="*50)
    print("SCORE DISTRIBUTION STATISTICS")
    print("="*50)
    
    print(f"Total Records: {stats['count']:,}")
    print(f"Mean Score: {stats['mean']:.2f}")
    print(f"Standard Deviation: {stats['std']:.2f}")
    print(f"Min Score: {stats['min']:.2f}")
    print(f"Max Score: {stats['max']:.2f}")
    print(f"Median Score: {stats['median']:.2f}")
    
    print("\nQuantiles:")
    print("-" * 30)
    print(f"10th percentile: {stats['q10']:.2f}")
    print(f"25th percentile: {stats['q25']:.2f}")
    print(f"50th percentile: {stats['q50']:.2f}")
    print(f"75th percentile: {stats['q75']:.2f}")
    print(f"90th percentile: {stats['q90']:.2f}")
    print(f"95th percentile: {stats['q95']:.2f}")
    print(f"99th percentile: {stats['q99']:.2f}")
    
    print("\nScore Ranges:")
    print("-" * 30)
    print(f"Top 25% (>= {stats['q75']:.2f}): {stats['count'] * 0.25:.0f} records")
    print(f"Top 50% (>= {stats['q50']:.2f}): {stats['count'] * 0.5:.0f} records")
    print(f"Top 75% (>= {stats['q25']:.2f}): {stats['count'] * 0.75:.0f} records")
    print("="*50)

def plot_score_distribution_chart(data: pd.DataFrame, title: str, output_path: str, 
                                 color: str = 'skyblue', bins: int = 30) -> None:
    """绘制分数分布图的通用函数"""
    # 确保输出目录存在
    import os
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
    
    score_columns = ['Score', 'Rhythm_Score', 'Structure_Score', 'Language_Score', 'Meaning_Score']
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(score_columns):
        if i < len(axes):
            score_data = data[col].dropna()
            
            if len(score_data) == 0:
                axes[i].text(0.5, 0.5, f'No data for {col}', 
                           transform=axes[i].transAxes, ha='center', va='center')
                axes[i].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
                continue
            
            # 绘制直方图
            axes[i].hist(score_data, bins=bins, alpha=0.7, color=color, edgecolor='black')
            
            # 添加统计线
            axes[i].axvline(score_data.mean(), color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {score_data.mean():.2f}')
            axes[i].axvline(score_data.median(), color='green', linestyle='--', linewidth=2, 
                          label=f'Median: {score_data.median():.2f}')
            
            # 设置图表属性
            axes[i].set_xlabel(col, fontsize=10)
            axes[i].set_ylabel('Frequency', fontsize=10)
            axes[i].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(len(score_columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Score distribution chart saved to {output_path}")

def plot_detailed_score_distributions(df: pd.DataFrame, output_dir: str = "analysis") -> None:
    """绘制高分诗作各项评分的分布直方图，以及不同体裁的分数分布"""
    logger.info("Creating detailed score distribution plots for top poems")
    
    # 绘制总体分数分布
    plot_score_distribution_chart(
        df, 
        "Top Poems Score Distribution", 
        f"{output_dir}/top_poems_detailed_scores.png",
        color='skyblue',
        bins=30
    )
    
    # 绘制不同体裁的分数分布
    logger.info("Creating genre-specific score distribution plots")
    
    # 创建体裁分数分布目录
    genre_dir = f"{output_dir}/genre_score_distributions"
    import os
    os.makedirs(genre_dir, exist_ok=True)
    
    # 体裁中文到拼音的映射
    genre_pinyin_map = {
        '七绝': 'Qijue',
        '七律': 'Qilv', 
        '五绝': 'Wujue',
        '五律': 'Wulv'
    }
    
    # 重点关注七绝、七律、五绝、五律
    target_genres = ['七绝', '七律', '五绝', '五律']
    
    for genre in target_genres:
        genre_data = df[df['Genre'] == genre]
        
        if len(genre_data) == 0:
            logger.warning(f"No data found for genre: {genre}")
            continue
        
        logger.info(f"Plotting score distribution for {genre}: {len(genre_data)} poems")
        
        # 获取拼音名称
        pinyin_name = genre_pinyin_map.get(genre, genre)
        
        plot_score_distribution_chart(
            genre_data,
            f'{pinyin_name} Score Distribution ({len(genre_data)} poems)',
            f"{genre_dir}/{pinyin_name}_score_distribution.png",
            color='lightcoral',
            bins=20
        )

def save_genre_distribution(genre_stats: dict, total_count: int, output_path: str) -> None:
    """保存体裁分布分析结果为markdown格式"""
    logger.info(f"Saving genre distribution analysis to {output_path}")
    
    # 确保输出目录存在
    import os
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 体裁分布分析报告\n\n")
            f.write(f"**总诗作数量**: {total_count:,} 首\n\n")
            
            f.write("## 主要体裁分布\n\n")
            f.write("| 体裁 | 数量 | 百分比 |\n")
            f.write("|------|------|--------|\n")
            
            # 按数量排序
            sorted_genres = sorted(genre_stats.items(), key=lambda x: x[1]['count'], reverse=True)
            
            for genre, stats in sorted_genres:
                f.write(f"| {genre} | {stats['count']:,} | {stats['percentage']:.2f}% |\n")
            
            f.write("\n## 详细统计\n\n")
            for genre, stats in sorted_genres:
                f.write(f"### {genre}\n")
                f.write(f"- **数量**: {stats['count']:,} 首\n")
                f.write(f"- **占比**: {stats['percentage']:.2f}%\n\n")
            
            f.write("---\n")
            f.write(f"*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        logger.info(f"Genre distribution analysis saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving genre distribution analysis: {e}")
        raise

def analyze_genre_distribution(df: pd.DataFrame, output_path: str = "analysis/genre_distribution.md") -> dict:
    """分析Genre分布并保存结果"""
    logger.info("Analyzing Genre distribution")
    
    genre_counts = df['Genre'].value_counts()
    genre_percentages = df['Genre'].value_counts(normalize=True) * 100
    
    # 重点关注七绝、七律、五绝、五律
    target_genres = ['七绝', '七律', '五绝', '五律']
    genre_stats = {}
    
    for genre in target_genres:
        if genre in genre_counts.index:
            genre_stats[genre] = {
                'count': genre_counts[genre],
                'percentage': genre_percentages[genre]
            }
        else:
            genre_stats[genre] = {'count': 0, 'percentage': 0.0}
    
    # 其他类型
    other_genres = genre_counts[~genre_counts.index.isin(target_genres)]
    if len(other_genres) > 0:
        genre_stats['其他'] = {
            'count': other_genres.sum(),
            'percentage': other_genres.sum() / len(df) * 100
        }
    
    # 保存分析结果为markdown格式
    save_genre_distribution(genre_stats, len(df), output_path)
    
    return genre_stats

def analyze_author_distribution(df: pd.DataFrame, top_n: int = 50) -> dict:
    """分析Author分布"""
    logger.info("Analyzing Author distribution")
    
    author_counts = df['Author'].value_counts()
    author_percentages = df['Author'].value_counts(normalize=True) * 100
    
    # 前N名作者
    top_authors = author_counts.head(top_n)
    top_author_percentages = author_percentages.head(top_n)
    
    author_stats = {
        'total_authors': len(author_counts),
        'top_authors': {}
    }
    
    for author in top_authors.index:
        author_stats['top_authors'][author] = {
            'count': top_authors[author],
            'percentage': top_author_percentages[author]
        }
    
    # 其他作者统计
    other_count = author_counts.iloc[top_n:].sum() if len(author_counts) > top_n else 0
    other_percentage = other_count / len(df) * 100
    
    author_stats['others'] = {
        'count': other_count,
        'percentage': other_percentage,
        'count_of_authors': len(author_counts) - top_n if len(author_counts) > top_n else 0
    }
    
    return author_stats

def print_top_poems_analysis(df_top: pd.DataFrame, genre_stats: dict, author_stats: dict) -> None:
    """打印高分诗作的详细分析"""
    print("\n" + "="*60)
    print("TOP POEMS DETAILED ANALYSIS")
    print("="*60)
    
    print(f"Total Top Poems: {len(df_top):,}")
    print(f"Score Range: {df_top['Score'].min():.2f} - {df_top['Score'].max():.2f}")
    print(f"Average Score: {df_top['Score'].mean():.2f}")
    
    print("\nGenre Distribution:")
    print("-" * 40)
    for genre, stats in genre_stats.items():
        print(f"{genre:8}: {stats['count']:6,} poems ({stats['percentage']:5.1f}%)")
    
    print("\nTop Authors (Top 20):")
    print("-" * 40)
    for author, stats in author_stats['top_authors'].items():
        print(f"{author:15}: {stats['count']:4} poems ({stats['percentage']:5.1f}%)")
    
    if author_stats['others']['count'] > 0:
        print(f"{'Others':15}: {author_stats['others']['count']:4} poems ({author_stats['others']['percentage']:5.1f}%)")
        print(f"  ({author_stats['others']['count_of_authors']} additional authors)")
    
    print(f"\nTotal Authors: {author_stats['total_authors']:,}")
    print("="*60)

def extract_sft_dataset(df_top: pd.DataFrame, n: int) -> pd.DataFrame:
    """提取前N首作为SFT训练集"""
    logger.info(f"Extracting top {n} poems for SFT training dataset")
    
    if len(df_top) >= n:
        df_sft = df_top.head(n)
        logger.info(f"Selected top {n} poems from {len(df_top)} total poems for SFT")
    else:
        df_sft = df_top
        logger.info(f"All {len(df_top)} poems selected for SFT (less than {n})")
    
    logger.info(f"SFT dataset Score range: {df_sft['Score'].min():.2f} - {df_sft['Score'].max():.2f}")
    logger.info(f"SFT dataset average Score: {df_sft['Score'].mean():.2f}")
    
    return df_sft

def extract_test_and_grpo_datasets(df_top: pd.DataFrame, m: int, grpo_size: int, test_size: int, random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """从df_top的后m首中随机抽取grpo_size和test_size首"""
    logger.info(f"Extracting GRPO and test datasets from last {m} poems")
    logger.info(f"Target: {grpo_size} for GRPO training, {test_size} for testing")
    logger.info(f"Using random seed: {random_seed}")
    
    # 获取后m首诗作
    if len(df_top) <= m:
        logger.warning(f"Not enough data: {len(df_top)} <= {m}")
        logger.info("Using all available data for GRPO and test extraction")
        df_candidates = df_top
    else:
        df_candidates = df_top.tail(m)
        logger.info(f"Selected last {m} poems from {len(df_top)} total poems")
    
    total_needed = grpo_size + test_size
    
    if len(df_candidates) < total_needed:
        logger.warning(f"Not enough candidates: {len(df_candidates)} < {total_needed}")

    # 随机打乱候选数据
    df_shuffled = df_candidates.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    logger.info("Candidate data shuffled randomly")
    
    # 按指定大小进行不放回抽样分割
    df_grpo = df_shuffled.iloc[:grpo_size]
    df_test = df_shuffled.iloc[grpo_size:grpo_size + test_size]
    
    logger.info(f"GRPO and test extraction completed:")
    logger.info(f"  GRPO training set: {len(df_grpo)} poems")
    logger.info(f"  Test set: {len(df_test)} poems")
    
    if len(df_grpo) > 0:
        logger.info(f"GRPO set Score range: {df_grpo['Score'].min():.2f} - {df_grpo['Score'].max():.2f}")
    if len(df_test) > 0:
        logger.info(f"Test set Score range: {df_test['Score'].min():.2f} - {df_test['Score'].max():.2f}")
    
    return df_grpo, df_test


def save_dataset_split(df_sft: pd.DataFrame, df_grpo: pd.DataFrame, df_test: pd.DataFrame, sft_path: str, grpo_path: str, test_path: str) -> None:
    """保存SFT训练集、GRPO训练集和测试集"""
    logger.info("Saving SFT training, GRPO training and test datasets")
    
    try:
        # 保存SFT训练集
        df_sft.to_csv(sft_path, index=False)
        logger.info(f"SFT training set saved to {sft_path}: {len(df_sft)} poems")
        
        # 保存GRPO训练集
        df_grpo.to_csv(grpo_path, index=False)
        logger.info(f"GRPO training set saved to {grpo_path}: {len(df_grpo)} poems")
        
        # 保存测试集
        df_test.to_csv(test_path, index=False)
        logger.info(f"Test set saved to {test_path}: {len(df_test)} poems")
        
        # 输出统计信息
        logger.info(f"SFT training set Score range: {df_sft['Score'].min():.2f} - {df_sft['Score'].max():.2f}")
        logger.info(f"GRPO training set Score range: {df_grpo['Score'].min():.2f} - {df_grpo['Score'].max():.2f}")
        logger.info(f"Test set Score range: {df_test['Score'].min():.2f} - {df_test['Score'].max():.2f}")
        
    except Exception as e:
        logger.error(f"Error saving datasets: {e}")
        raise

def main():
    """主函数"""
    logger.info("Starting poetry data analysis and selection")
    
    # 文件路径
    input_file = "chinese_poetry_curated3_all.csv"
    sorted_output_file = "chinese_poetry_curated3_sorted.csv"
    top_output_file = "chinese_poetry_curated3_top.csv"
    sft_output_file = "dataset_sft.csv"
    grpo_output_file = "dataset_grpo.csv"
    test_output_file = "dataset_test.csv"
    histogram_output_file = "analysis/score_distribution.png"
    
    try:
        # 1. 加载数据
        df = load_data(input_file)
        
        # 2. 分析整体Score分布
        score_data, stats = analyze_score_distribution(df)
        
        # 3. 绘制整体分布直方图
        plot_score_histogram(score_data, histogram_output_file)
        
        # 4. 打印整体统计信息
        print_statistics(stats)
        
        # 5. 按评分降序排列
        df_sorted = sort_data(df)
        
        # 6. 保存排序后的数据
        save_sorted_data(df_sorted, sorted_output_file)
        
        # 7. 提取前50000首诗作
        df_top = extract_top_poems(df_sorted, max_count=50000)
        
        # 8. 保存高分诗作
        save_top_poems(df_top, top_output_file)
        
        # 9. 绘制高分诗作各项评分分布图
        plot_detailed_score_distributions(df_top, "analysis")
        
        # 10. 分析Genre分布
        genre_stats = analyze_genre_distribution(df_top, "analysis/genre_distribution.md")
        
        # 11. 分析Author分布
        author_stats = analyze_author_distribution(df_top)
        
        # 12. 打印高分诗作详细分析
        print_top_poems_analysis(df_top, genre_stats, author_stats)
        
        # 13. 提取SFT训练集（前30000首）
        df_sft = extract_sft_dataset(df_top, n=2000)
        
        # 14. 从后20000首中随机抽取GRPO集和测试集
        df_grpo, df_test = extract_test_and_grpo_datasets(df_top, m=20000, grpo_size=10000, test_size=10000, random_seed=42)
        
        # 15. 保存所有数据集
        save_dataset_split(df_sft, df_grpo, df_test, sft_output_file, grpo_output_file, test_output_file)
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
