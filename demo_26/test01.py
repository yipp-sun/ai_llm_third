import json


# 读取JSONL文件
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


# 转换为Llama Factory格式
def convert_to_llama_factory_format(data):
    llama_data = []
    for item in data:
        instruction = item['question']
        input_text = ""
        for key in ['A', 'B', 'C', 'D', 'E', 'F']:
            if item[key]:
                input_text += f"{key}: {item[key]}\n"
        output_text = item['answer']
        llama_data.append({
            "instruction": instruction,
            "input": input_text.strip(),
            "output": output_text
        })
    return llama_data


# 保存为新的JSONL文件
def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


# 主函数
def main():
    input_file = 'data/CAP_civics_test.jsonl'  # 替换为你的输入文件路径
    output_file = 'data/llama_factory_format.jsonl'  # 替换为你的输出文件路径

    # 读取原始数据
    data = read_jsonl(input_file)

    # 转换为Llama Factory格式
    llama_data = convert_to_llama_factory_format(data)

    # 保存转换后的数据
    save_jsonl(llama_data, output_file)


if __name__ == "__main__":
    main()
