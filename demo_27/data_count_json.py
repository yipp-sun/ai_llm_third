import json

# 读取JSON文件
with open('data/train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 初始化各键的最大长度记录
max_lengths = {
    'instruction': 0,
    'input': 0,
    'output': 0
}

# 遍历每个条目并更新最大长度
for item in data:
    for key in max_lengths:
        current_len = len(item[key])
        if current_len > max_lengths[key]:
            max_lengths[key] = current_len

# 计算全局最大长度
global_max = max(max_lengths.values())

# 输出结果
print("各键的最大长度：")
for key, value in max_lengths.items():
    print(f"{key}: {value}")

print(f"\n全局最大长度：{global_max}")
