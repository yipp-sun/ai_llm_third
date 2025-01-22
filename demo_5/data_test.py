from datasets import load_dataset, load_from_disk

# 在线加载数据
# dataset = load_dataset(path="NousResearch/hermes-function-calling-v1",split="train")
# print(dataset)
# 转存为CSV格式
# dataset.to_csv(path_or_buf=r"D:\Workspace\AIProject\demo_5\data\hermes-function-calling-v1.csv")
# 加载csv格式数据
# dataset = load_dataset(path="csv", data_files=r"D:\Workspace\AIProject\demo_5\data\hermes-function-calling-v1.csv")
# print(dataset)
# 加载缓存数据
dataset = load_from_disk(r"D:\Workspace\AIProject\demo_5\data\ChnSentiCorp")
print(dataset)

test_data = dataset["train"]
for data in test_data:
    print(data)
