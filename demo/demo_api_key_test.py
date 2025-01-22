import os
from openai import OpenAI

# 设置 openai相关的参数
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)


def chat():
    response = client.chat.completions.create(
        model="gpt-4o",  # 接入 gpt-4o的基座
        messages=[
            {"role": "system", "content": "你是一个专业的中英文翻译器，请将用户输入的内容进行翻译."},
            {"role": "user", "content": "我今天有很多会议，太忙了！"}
            # {"role": "system", "content": "你是一个AI 助手."},
            # {"role": "user", "content": "我想学习OpenAI，请给出详细的 Python 调用代码"}
        ],
        temperature=0
    )
    msg = response.choices[0].message.content
    print(msg)


chat()
