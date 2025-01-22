# 使用openai的代码风格调用ollama
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")
# 单人对话模式
responce = client.chat.completions.create(
    # ollama自带系统信息，所以这里省略
    messages=[{"role": "user", "content": "你好，你叫什么名字？你是由谁创造的？"}], model="llama3.2:1b"
)

# print(responce)
print(responce.choices[0])
