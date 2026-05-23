#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文诗词数据整理工具
将全唐诗文件夹中的数据整理，生成繁体版和简体版CSV格式的整理数据
"""

import json
import os
import csv
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
import opencc


class PoetryDataCurator:
    """诗词数据整理器"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.poetry_path = self.base_path / "全唐诗"
        self.traditional_output_file = self.base_path / "chinese_poetry_traditional.csv"
        self.simplified_output_file = self.base_path / "chinese_poetry_simplified.csv"
        
    def get_dynasty_from_filename(self, filename: str) -> str:
        """根据文件名确定朝代"""
        if "tang" in filename:
            return "唐"
        elif "song" in filename:
            return "宋"
        else:
            return ""
    
    def load_poetry_data(self, file_path: Path) -> List[Dict[str, Any]]:
        """加载诗歌数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"加载诗歌文件失败 {file_path}: {e}")
            return []
    
    
    def merge_paragraphs(self, paragraphs: List[str]) -> str:
        """将诗歌段落合并为完整内容"""
        if not paragraphs:
            return ""
        return "".join(paragraphs)
    
    
    def process_single_file(self, poetry_file: Path) -> List[Dict[str, Any]]:
        """处理单个诗歌文件"""
        print(f"处理文件: {poetry_file.name}")
        
        # 确定朝代
        dynasty = self.get_dynasty_from_filename(poetry_file.name)
        
        # 加载诗歌数据
        poetry_data = self.load_poetry_data(poetry_file)
        if not poetry_data:
            return []
        
        # 合并数据
        merged_data = []
        for poem in poetry_data:
            # 提取基本信息
            title = poem.get('title', '')
            author = poem.get('author', '')
            paragraphs = poem.get('paragraphs', [])
            content = self.merge_paragraphs(paragraphs)
            
            # 构建输出数据
            row_data = {
                'Title': title,
                'Author': author,
                'Dynasty': dynasty,
                'Content': content
            }
            
            merged_data.append(row_data)
        
        return merged_data
    
    def process_all_files(self) -> List[Dict[str, Any]]:
        """处理所有文件"""
        all_data = []
        
        # 获取所有诗歌文件
        poetry_files = []
        if self.poetry_path.exists():
            for file_path in self.poetry_path.glob("poet.*.json"):
                # 排除不需要的文件
                if file_path.name in ["README.md", "表面结构字.json", "error", "authors.song.json", "authors.tang.json"]:
                    continue
                poetry_files.append(file_path)
        
        print(f"找到 {len(poetry_files)} 个诗歌文件")
        
        # 处理每个文件
        for poetry_file in sorted(poetry_files):
            file_data = self.process_single_file(poetry_file)
            all_data.extend(file_data)
            print(f"已处理 {poetry_file.name}, 获得 {len(file_data)} 条记录")
        
        return all_data
    
    def save_to_csv(self, data: List[Dict[str, Any]]) -> None:
        """保存数据到CSV文件"""
        if not data:
            print("没有数据需要保存")
            return
        
        # 使用pandas保存CSV
        df = pd.DataFrame(data)
        
        # 保存原始繁体数据
        df.to_csv(self.traditional_output_file, index=False, encoding='utf-8-sig')
        print(f"繁体数据已保存到: {self.traditional_output_file}")
        print(f"总共保存了 {len(data)} 条记录")
        
        # 将繁体字转换为简体字
        print("正在将繁体字转换为简体字...")
        converter = opencc.OpenCC('t2s')  # 繁体转简体
        df_simplified = df.copy()
        
        # 只对指定的文本列进行转换，避免产生空列
        text_columns = ['Title', 'Author', 'Dynasty', 'Content']
        for column in text_columns:
            if column in df_simplified.columns:
                df_simplified[column] = df_simplified[column].apply(lambda x: converter.convert(x) if isinstance(x, str) else x)
        
        # 确保只保留需要的列
        df_simplified = df_simplified[text_columns]
        print("繁体字转换完成")
        
        # 保存简体数据
        df_simplified.to_csv(self.simplified_output_file, index=False, encoding='utf-8-sig')
        print(f"简体数据已保存到: {self.simplified_output_file}")
        print(f"简体数据保存了 {len(df_simplified)} 条记录")
        
        # 显示数据统计
        print("\n数据统计:")
        print(f"唐诗数量: {len(df[df['Dynasty'] == '唐'])}")
        print(f"宋诗数量: {len(df[df['Dynasty'] == '宋'])}")
        print(f"总记录数: {len(df)}")
    
    def run(self) -> None:
        """运行数据整理流程"""
        print("开始整理诗词数据...")
        
        # 处理所有文件
        all_data = self.process_all_files()
        
        if all_data:
            # 保存到CSV
            self.save_to_csv(all_data)
            print("数据整理完成!")
        else:
            print("没有找到任何数据")


def main():
    """主函数"""
    curator = PoetryDataCurator()
    curator.run()


if __name__ == "__main__":
    main()
