#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练日志分析器
分析GRPO训练过程中的得分趋势和体裁分布
"""

import json
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from datetime import datetime

# 设置字体为默认英文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class LogAnalyzer:
    def __init__(self, logs_dir="training_logs", output_dir="log_analysis", idx_start=None, idx_end=None):
        self.logs_dir = logs_dir
        self.output_dir = output_dir
        self.idx_start = idx_start
        self.idx_end = idx_end
        self.data = []
        self.genres = []
        
        # 定义有效的体裁类型
        self.valid_genres = {'七律', '七绝', '五律', '五绝'}
    
    def filter_genre(self, genre):
        """过滤体裁，只保留四种主要体裁，其余归类为Unknown_Genre"""
        if genre in self.valid_genres:
            return genre
        else:
            return 'Unknown_Genre'
    
    def _get_filename(self, base_filename):
        """生成包含区间信息的文件名"""
        if self.idx_start is None and self.idx_end is None:
            return base_filename
        
        name, ext = os.path.splitext(base_filename)
        start_str = str(self.idx_start) if self.idx_start is not None else "0"
        end_str = str(self.idx_end) if self.idx_end is not None else "inf"
        return f"{name}_{start_str}_{end_str}{ext}"
        
    def load_data(self):
        """加载所有训练日志数据"""
        print("Loading log data...")
        
        # 获取所有JSON文件
        json_files = glob.glob(os.path.join(self.logs_dir, "*.json"))
        json_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        
        print(f"Found {len(json_files)} log files")
        
        # 如果有区间限制，显示区间信息
        if self.idx_start is not None or self.idx_end is not None:
            start_str = str(self.idx_start) if self.idx_start is not None else "0"
            end_str = str(self.idx_end) if self.idx_end is not None else "∞"
            print(f"Filtering samples in range [{start_str}, {end_str})")
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                sample_index = data['sample_index']
                
                # 检查样本索引是否在指定区间内
                if self.idx_start is not None and sample_index < self.idx_start:
                    continue
                if self.idx_end is not None and sample_index >= self.idx_end:
                    continue
                
                completions = data['completions']
                
                # 提取每个completion的得分数据
                scores_data = []
                for completion in completions:
                    eval_result = completion['evaluation_result']
                    scores = eval_result['得分']
                    genre = eval_result['体裁判断']
                    
                    # 过滤体裁
                    filtered_genre = self.filter_genre(genre)
                    
                    scores_data.append({
                        '总分': scores['总分'],
                        '格律规范性': scores['格律规范性'],
                        '对仗与结构': scores['对仗与结构'],
                        '语言与锤炼': scores['语言与锤炼'],
                        '意境与立意': scores['意境与立意'],
                        '体裁': filtered_genre
                    })
                
                # 计算该样本的平均得分
                if scores_data:
                    avg_scores = {}
                    for key in ['总分', '格律规范性', '对仗与结构', '语言与锤炼', '意境与立意']:
                        avg_scores[key] = np.mean([item[key] for item in scores_data])
                    
                    # 记录体裁信息（已经是过滤后的）
                    genres = [item['体裁'] for item in scores_data]
                    self.genres.extend(genres)
                    
                    # 添加样本数据
                    sample_data = {
                        'sample_index': sample_index,
                        'timestamp': data.get('timestamp', ''),
                        **avg_scores
                    }
                    self.data.append(sample_data)
                    
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue
        
        print(f"Successfully loaded data for {len(self.data)} samples")
        return self.data
    
    def create_output_dir(self):
        """创建输出目录"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
    
    def plot_score_trends(self):
        """绘制得分趋势图"""
        if not self.data:
            print("No data available for plotting")
            return
        
        df = pd.DataFrame(self.data)
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Score Trends Analysis', fontsize=16, fontweight='bold')
        
        # 定义得分维度（中文列名）
        score_dimensions = ['总分', '格律规范性', '对仗与结构', '语言与锤炼', '意境与立意']
        
        # 定义列名到英文标签的映射
        dimension_labels = {
            '总分': 'Total Score',
            '格律规范性': 'Rhythm Norm',
            '对仗与结构': 'Structure',
            '语言与锤炼': 'Language',
            '意境与立意': 'Meaning'
        }
        
        # 绘制各维度趋势图
        for i, dimension in enumerate(score_dimensions):
            row = i // 3
            col = i % 3
            
            ax = axes[row, col]
            
            # 绘制原始数据点
            ax.scatter(df['sample_index'], df[dimension], alpha=0.6, s=20, color='lightblue')
            
            # 绘制平滑曲线
            from scipy.interpolate import make_interp_spline
            try:
                x_smooth = np.linspace(df['sample_index'].min(), df['sample_index'].max(), 300)
                y_smooth = make_interp_spline(df['sample_index'], df[dimension], k=3)(x_smooth)
                ax.plot(x_smooth, y_smooth, color='red', linewidth=2, label='Smooth Curve')
            except:
                # 如果平滑失败，使用简单移动平均
                window_size = max(10, len(df) // 20)
                df_smooth = df[dimension].rolling(window=window_size, center=True).mean()
                ax.plot(df['sample_index'], df_smooth, color='red', linewidth=2, label='Moving Average')
            
            # 绘制趋势线
            z = np.polyfit(df['sample_index'], df[dimension], 1)
            p = np.poly1d(z)
            ax.plot(df['sample_index'], p(df['sample_index']), "g--", alpha=0.8, linewidth=2, label='Trend Line')
            
            ax.set_title(f'{dimension_labels[dimension]} Trend', fontsize=12, fontweight='bold')
            ax.set_xlabel('Steps')
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 添加统计信息
            mean_score = df[dimension].mean()
            std_score = df[dimension].std()
            ax.text(0.02, 0.98, f'Mean: {mean_score:.1f}\nStd: {std_score:.1f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 隐藏最后一个子图
        axes[1, 2].set_visible(False)
        
        plt.tight_layout()
        # 生成文件名（包含区间信息）
        filename = self._get_filename('score_trends.png')
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Score trends chart saved")
    
    def plot_total_score_trend(self):
        """绘制总分趋势图"""
        if not self.data:
            return
        
        df = pd.DataFrame(self.data)
        
        plt.figure(figsize=(12, 8))
        
        # 绘制原始数据点
        plt.scatter(df['sample_index'], df['总分'], alpha=0.6, s=30, color='lightblue', label='Raw Data')
        
        # 绘制平滑曲线
        try:
            from scipy.interpolate import make_interp_spline
            x_smooth = np.linspace(df['sample_index'].min(), df['sample_index'].max(), 300)
            y_smooth = make_interp_spline(df['sample_index'], df['总分'], k=3)(x_smooth)
            plt.plot(x_smooth, y_smooth, color='red', linewidth=3, label='Smooth Curve')
        except:
            window_size = max(10, len(df) // 20)
            df_smooth = df['总分'].rolling(window=window_size, center=True).mean()
            plt.plot(df['sample_index'], df_smooth, color='red', linewidth=3, label='Moving Average')
        
        # 绘制趋势线
        z = np.polyfit(df['sample_index'], df['总分'], 1)
        p = np.poly1d(z)
        plt.plot(df['sample_index'], p(df['sample_index']), "g--", alpha=0.8, linewidth=3, label='Trend Line')
        
        plt.title('Total Score Trend', fontsize=16, fontweight='bold')
        plt.xlabel('Steps', fontsize=12)
        plt.ylabel('Total Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # 添加统计信息
        mean_score = df['总分'].mean()
        std_score = df['总分'].std()
        min_score = df['总分'].min()
        max_score = df['总分'].max()
        
        plt.text(0.02, 0.98, f'Statistics:\nMean: {mean_score:.1f}\nStd: {std_score:.1f}\nMin: {min_score:.1f}\nMax: {max_score:.1f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
        
        plt.tight_layout()
        # 生成文件名（包含区间信息）
        filename = self._get_filename('total_score_trend.png')
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Total score trend chart saved")
    
    def plot_genre_distribution(self):
        """绘制体裁分布图"""
        if not self.genres:
            print("No genre data available")
            return
        
        # 统计体裁分布
        genre_counts = Counter(self.genres)
        
        # 定义体裁中文到英文的映射（只包含有效体裁）
        genre_labels = {
            '七律': 'Qilu',
            '五律': 'Wulu', 
            '七绝': 'Qijue',
            '五绝': 'Wujue',
            'Unknown_Genre': 'Unknown_Genre'
        }
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 饼图 - 使用英文标签
        chinese_labels = list(genre_counts.keys())
        english_labels = [genre_labels.get(label, label) for label in chinese_labels]
        sizes = list(genre_counts.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(chinese_labels)))
        
        wedges, texts, autotexts = ax1.pie(sizes, labels=english_labels, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax1.set_title('Genre Distribution Pie Chart', fontsize=14, fontweight='bold')
        
        # 美化饼图文字
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # 柱状图 - 使用英文标签
        bars = ax2.bar(english_labels, sizes, color=colors)
        ax2.set_title('Genre Distribution Bar Chart', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Genre')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        
        # 在柱状图上添加数值标签
        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{size}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        # 生成文件名（包含区间信息）
        filename = self._get_filename('genre_distribution.png')
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Genre distribution chart saved")
        
        return genre_counts
    
    def generate_analysis_report(self, genre_counts):
        """生成分析报告"""
        if not self.data:
            return
        
        df = pd.DataFrame(self.data)
        
        # 生成文件名（包含区间信息）
        filename = self._get_filename('analysis_report.txt')
        report_path = os.path.join(self.output_dir, filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Log Analysis Report\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 基本统计信息
            f.write("1. Basic Statistics\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Samples: {len(self.data)}\n")
            f.write(f"Steps Range: {df['sample_index'].min()} - {df['sample_index'].max()}\n")
            f.write(f"Total Generated Works: {len(self.genres)}\n")
            
            # 添加区间信息
            if self.idx_start is not None or self.idx_end is not None:
                start_str = str(self.idx_start) if self.idx_start is not None else "0"
                end_str = str(self.idx_end) if self.idx_end is not None else "∞"
                f.write(f"Analysis Range: [{start_str}, {end_str})\n")
            f.write("\n")
            
            # 得分统计
            f.write("2. Score Statistics\n")
            f.write("-" * 30 + "\n")
            score_dimensions = ['总分', '格律规范性', '对仗与结构', '语言与锤炼', '意境与立意']
            
            # 定义列名到英文标签的映射
            dimension_labels = {
                '总分': 'Total Score',
                '格律规范性': 'Rhythm Norm',
                '对仗与结构': 'Structure',
                '语言与锤炼': 'Language',
                '意境与立意': 'Meaning'
            }
            
            for dimension in score_dimensions:
                mean_score = df[dimension].mean()
                std_score = df[dimension].std()
                min_score = df[dimension].min()
                max_score = df[dimension].max()
                
                f.write(f"{dimension_labels[dimension]}:\n")
                f.write(f"  Mean: {mean_score:.2f}\n")
                f.write(f"  Std Dev: {std_score:.2f}\n")
                f.write(f"  Min: {min_score:.2f}\n")
                f.write(f"  Max: {max_score:.2f}\n\n")
            
            # 体裁分布
            f.write("3. Genre Distribution Statistics\n")
            f.write("-" * 30 + "\n")
            total_genres = len(self.genres)
            for genre, count in genre_counts.most_common():
                percentage = (count / total_genres) * 100
                f.write(f"{genre}: {count} poems ({percentage:.1f}%)\n")
            f.write("\n")
            
            # 趋势分析
            f.write("4. Trend Analysis\n")
            f.write("-" * 30 + "\n")
            
            # 计算趋势斜率
            for dimension in score_dimensions:
                z = np.polyfit(df['sample_index'], df[dimension], 1)
                slope = z[0]
                if slope > 0:
                    trend = "Rising"
                elif slope < 0:
                    trend = "Falling"
                else:
                    trend = "Stable"
                
                f.write(f"{dimension_labels[dimension]}: {trend} trend (slope: {slope:.4f})\n")
            
            f.write("\n")
            
            # 关键发现
            f.write("5. Key Findings\n")
            f.write("-" * 30 + "\n")
            
            # 找出得分最高的样本
            best_sample = df.loc[df['总分'].idxmax()]
            f.write(f"Best Sample: Step {best_sample['sample_index']}, Total Score {best_sample['总分']:.1f}\n")
            
            # 找出得分最低的样本
            worst_sample = df.loc[df['总分'].idxmin()]
            f.write(f"Worst Sample: Step {worst_sample['sample_index']}, Total Score {worst_sample['总分']:.1f}\n")
            
            # 分析各维度的表现
            dimension_means = {dim: df[dim].mean() for dim in score_dimensions[1:]}
            best_dimension = max(dimension_means, key=dimension_means.get)
            worst_dimension = min(dimension_means, key=dimension_means.get)
            
            f.write(f"Best Performing Dimension: {dimension_labels[best_dimension]} (Mean: {dimension_means[best_dimension]:.1f})\n")
            f.write(f"Dimension Needing Improvement: {dimension_labels[worst_dimension]} (Mean: {dimension_means[worst_dimension]:.1f})\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("Report End\n")
        
        print("Analysis report generated")
    
    def run_analysis(self):
        """运行完整分析"""
        print("Starting log analysis...")
        
        # 创建输出目录
        self.create_output_dir()
        
        # 加载数据
        self.load_data()
        
        if not self.data:
            print("No valid data found, analysis terminated")
            return
        
        # 绘制图表
        self.plot_total_score_trend()
        self.plot_score_trends()
        genre_counts = self.plot_genre_distribution()
        
        # 生成报告
        self.generate_analysis_report(genre_counts)
        
        print(f"\nAnalysis completed! Results saved to {self.output_dir} directory")
        print("Generated files:")
        print("- total_score_trend.png: Total score trend chart")
        print("- score_trends.png: Multi-dimensional score trends chart")
        print("- genre_distribution.png: Genre distribution chart")
        print("- analysis_report.txt: Detailed analysis report")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='分析训练或评估日志')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='eval',
                       help='分析模式: train(训练日志) 或 eval(评估日志)')
    parser.add_argument('--idx-start', type=int, default=0,
                       help='起始样本索引 (包含)')
    parser.add_argument('--idx-end', type=int, default=1000,
                       help='结束样本索引 (不包含)')
    args = parser.parse_args()
    
    # 参数验证
    if args.idx_start is not None and args.idx_end is not None:
        if args.idx_start >= args.idx_end:
            print("错误: idx-start 必须小于 idx-end")
            exit(1)
    if args.idx_start is not None and args.idx_start < 0:
        print("错误: idx-start 不能为负数")
        exit(1)
    if args.idx_end is not None and args.idx_end < 0:
        print("错误: idx-end 不能为负数")
        exit(1)
    
    # 根据模式设置
    if args.mode == 'eval':
        args.logs_dir = 'eval_logs'
        args.output_dir = 'eval_analysis'       
    elif args.mode == 'train':
        args.logs_dir = 'grpo_logs'
        args.output_dir = 'grpo_analysis'

    print(f"分析模式: {args.mode}")
    print(f"日志目录: {args.logs_dir}")
    print(f"输出目录: {args.output_dir}")
    if args.idx_start is not None or args.idx_end is not None:
        start_str = str(args.idx_start) if args.idx_start is not None else "0"
        end_str = str(args.idx_end) if args.idx_end is not None else "∞"
        print(f"样本区间: [{start_str}, {end_str})")
    
    analyzer = LogAnalyzer(logs_dir=args.logs_dir, output_dir=args.output_dir, 
                          idx_start=args.idx_start, idx_end=args.idx_end)
    analyzer.run_analysis()
