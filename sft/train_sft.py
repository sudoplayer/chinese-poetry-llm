# -*- coding: utf-8 -*-
"""
律诗创作模型LoRA微调训练脚本
基于Qwen3-4B-Instruct模型，使用 dataset_sft.csv数据进行SFT训练

GPU适配兼容性说明：
- 当 gpu_flag = 'RTX2080Ti' 时，启用本地适配
- 当 gpu_flag = 'RTX4090' 时，启用AutoDL适配
"""

import os
import sys
import torch
import warnings
from pathlib import Path
from datasets import Dataset
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only
from trl import SFTTrainer, SFTConfig

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 屏蔽PyTorch相关的FutureWarning警告
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", message=".*torch.backends.cuda.sdp_kernel.*")
warnings.filterwarnings("ignore", message=".*torch.nn.attention.sdpa_kernel.*")

try:
    from sft.sft_config import config
except ImportError:
    from sft_config import config
from poetry_core.poetry_data_loader import PoetryDataLoader
from poetry_core.poetry_generator import PoetryGenerator

# 使用config.py中的配置
print(f"使用GPU配置: {config.GPU_FLAG}")
print(f"训练轮数: {config.TRAINING_EPOCHS}")
print(f"LoRA秩: {config.LORA_RANK}")

# 设置CUDA设备
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)

def load_and_prepare_data(csv_path):
    """加载并预处理律诗数据"""
    print("正在加载律诗数据...")
    
    # 使用PoetryDataLoader加载数据
    data_loader = PoetryDataLoader(csv_path)
    poetry_data_list = data_loader.load_data(idx_start=config.IDX_START, idx_end=config.IDX_END)
    
    # 使用PoetryGenerator生成提示词
    poetry_generator = PoetryGenerator()
    
    # 转换为对话格式
    conversations = []
    for poetry_data in poetry_data_list:
        prompt = poetry_generator._get_create_prompt(poetry_data.query)
        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": poetry_data.reference}
        ]
        conversations.append({"conversations": conversation})
    
    # 创建Dataset
    dataset = Dataset.from_list(conversations)
    
    # 标准化数据格式
    dataset = standardize_data_formats(dataset)
    
    print(f"数据预处理完成，共 {len(dataset)} 条对话")
    return dataset

def setup_model_and_tokenizer():
    """设置模型和分词器"""
    print("正在加载Qwen3-4B-Instruct模型...")
     
    model, tokenizer = FastModel.from_pretrained(
        model_name=config.MODEL_NAME,
        max_seq_length=config.MAX_SEQ_LENGTH,
        load_in_4bit=config.LOAD_IN_4BIT,
        load_in_8bit=False,
        full_finetuning=False,
        device_map={"": local_rank},
        token=config.HUGGINGFACE_TOKEN,
    )
    
    # 设置聊天模板
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=config.CHAT_TEMPLATE,
    )
    
    print("模型和分词器加载完成")
    return model, tokenizer

def setup_lora(model):
    """设置LoRA适配器"""
    print("正在配置LoRA适配器...")
    
    model = FastModel.get_peft_model(
        model,
        r=config.LORA_RANK, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config.LORA_RANK*2,
        lora_dropout=0,
        weight_decay=0.01,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.RANDOM_SEED,
        use_rslora=False,
        loftq_config=None,
    )
    
    print("LoRA适配器配置完成")
    return model

def format_dataset(dataset, tokenizer):
    """格式化数据集"""
    print("正在格式化数据集...")
    
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) 
                for convo in convos]
        return {"text": texts}
    
    dataset = dataset.map(formatting_prompts_func, batched=True)
    print("数据集格式化完成")
    return dataset

def setup_trainer(model, tokenizer, dataset):
    """设置训练器"""
    print("正在配置训练器...")
     
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
            warmup_ratio=0.1,
            num_train_epochs=config.TRAINING_EPOCHS,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=config.RANDOM_SEED,
            report_to="swanlab",
            run_name="qwen3-4B-Instruct-SFT",
            output_dir="./sft_lora_output",
            save_steps=config.SAVE_STEPS,
            save_total_limit=1,
        ),
    )
    
    # 只对assistant回复进行训练
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    
    print("训练器配置完成")
    return trainer

def train_model(trainer):
    """执行模型训练"""
    print("开始训练模型...")
    
    # 显示训练前内存状态
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    
    # 开始训练
    trainer_stats = trainer.train()
    
    # 显示训练后内存状态
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
    
    return trainer_stats

def save_model(model, tokenizer, save_path="./sft_merged_model"):
    """保存合并后的模型"""
    print("正在保存模型...")
    
    # 保存LoRA适配器
    model.save_pretrained("./sft_lora_adapters")
    tokenizer.save_pretrained("./sft_lora_adapters")
    print("LoRA适配器已保存到 ./sft_lora_adapters")
    
    # 合并并保存完整模型
    try:
        print("正在合并LoRA适配器到基础模型...")
        model.save_pretrained_merged(save_path, tokenizer, save_method="merged_16bit")
        print(f"合并后的模型已保存到 {save_path}")
        
        # 验证保存是否成功
        import os
        if os.path.exists(save_path):
            safetensors_files = [f for f in os.listdir(save_path) if f.endswith('.safetensors')]
            if safetensors_files:
                print(f"✓ 成功保存 {len(safetensors_files)} 个模型权重文件")
                for f in safetensors_files:
                    file_size = os.path.getsize(os.path.join(save_path, f)) / (1024*1024*1024)  # GB
                    print(f"  - {f}: {file_size:.2f} GB")
            else:
                print("⚠ 警告：未找到.safetensors文件，merged_model可能未正确保存")
        else:
            print("⚠ 警告：merged_model目录不存在")
            
    except Exception as e:
        print(f"保存merged_model时出现错误: {str(e)}")
        print("尝试使用备用保存方法...")
        try:
            # 备用方法：使用lora_merged保存
            model.save_pretrained_merged(save_path, tokenizer, save_method="lora_merged")
            print(f"使用备用方法保存成功: {save_path}")
        except Exception as e2:
            print(f"备用保存方法也失败: {str(e2)}")
            print("请检查模型状态和磁盘空间")
    
    return save_path

def main():
    """主函数"""
    print("=" * 50)
    print("律诗创作模型LoRA微调训练")
    print("=" * 50)
    
    try:
        # 1. 加载和预处理数据
        dataset = load_and_prepare_data(
            os.getenv("SFT_DATA_PATH", "data/dataset_sft.csv")
        )
        
        # 2. 设置模型和分词器
        model, tokenizer = setup_model_and_tokenizer()
        
        # 3. 设置LoRA适配器
        model = setup_lora(model)
        
        # 4. 格式化数据集
        dataset = format_dataset(dataset, tokenizer)
        
        # 5. 设置训练器
        trainer = setup_trainer(model, tokenizer, dataset)
        
        # 6. 训练模型
        trainer_stats = train_model(trainer)
        print("训练统计信息：")
        print(trainer_stats)
        # 7. 保存模型
        save_path = save_model(model, tokenizer)
              
        print("=" * 50)
        print("训练完成！")
        print(f"模型已保存到: {save_path}")
        print("=" * 50)
        
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    """
    使用说明：
    1. 修改 gpu_flag 变量来控制GPU适配：
       - 设置为 'RTX2080Ti' 启用本地适配
       - 设置为 'RTX4090' 启用AutoDL适配
    2. 确保数据文件 'dataset_sft.csv' 存在
    3. 运行脚本开始训练
    """
    main()