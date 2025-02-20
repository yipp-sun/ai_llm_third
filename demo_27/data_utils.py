import csv
import json
import os

# 指定包含CSV文件的目录
csv_directory = 'data'
# 输出的JSON文件路径
json_filepath = 'data/train.json'

# 读取CSV文件并转换为JSON格式
def csv_to_json(csv_directory, json_filepath):
    data = []  # 初始化一个空列表来存储转换后的数据

    # 遍历目录中的所有文件
    for filename in os.listdir(csv_directory):
        if filename.endswith('.csv'):
            csv_filepath = os.path.join(csv_directory, filename)
            try:
                # 尝试使用UTF-8编码打开CSV文件
                with open(csv_filepath, mode='r', encoding='utf-8') as csvfile:
                    csvreader = csv.DictReader(csvfile)
                    # 遍历CSV文件中的每一行
                    for row in csvreader:
                        # 将CSV行转换为字典，并添加到数据列表中
                        data.append({
                            "instruction": row['题目（含完整选项）'],
                            "input": "",
                            "output": row['答案']
                        })
            except UnicodeDecodeError:
                # 如果UTF-8编码失败，尝试使用GBK编码
                with open(csv_filepath, mode='r', encoding='gbk') as csvfile:
                    csvreader = csv.DictReader(csvfile)
                    # 遍历CSV文件中的每一行
                    for row in csvreader:
                        # 将CSV行转换为字典，并添加到数据列表中
                        data.append({
                            "instruction": row['题目（含完整选项）'],
                            "input": "",
                            "output": row['答案']
                        })

    # 将数据列表转换为JSON格式，并写入文件
    with open(json_filepath, mode='w', encoding='utf-8') as jsonfile:
        jsonfile.write(json.dumps(data, ensure_ascii=False, indent=4))

# 调用函数
csv_to_json(csv_directory, json_filepath)
