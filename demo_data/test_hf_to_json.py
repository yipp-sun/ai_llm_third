# 把Huggingface下载的arrow数据集转化为json格式
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("data")

# 指定保存路径
output_dir = "ds-json"

# 将数据集转换为 JSON 格式
for split in dataset.keys():  # 处理所有划分 (train, validation 等)
    dataset[split].to_json(f"{output_dir}/{split}.json", orient="records", lines=True, force_ascii=False)
