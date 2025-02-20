import json
import random

# 指定输入的JSON文件路径
input_json_filepath = 'data/train.json'
# 指定输出的JSON文件路径
output_json_filepath = 'data/test.json'
# 指定选择数据的比例因子（例如，0.1表示选择10%的数据）
selection_ratio = 0.3

# 从JSON文件中读取数据
def read_json_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

# 将随机选择的数据写入新的JSON文件
def write_random_data_to_json(random_data, filepath):
    with open(filepath, 'w', encoding='utf-8') as file:
        json.dump(random_data, file, ensure_ascii=False, indent=4)

# 主函数
def main():
    # 读取JSON文件中的所有数据
    all_data = read_json_file(input_json_filepath)
    # 计算要选择的数据数量
    number_of_items_to_select = int(len(all_data) * selection_ratio)
    # 随机选择指定数量的数据
    random_data = random.sample(all_data, number_of_items_to_select)
    # 将随机选择的数据写入新的JSON文件
    write_random_data_to_json(random_data, output_json_filepath)

# 调用主函数
main()
