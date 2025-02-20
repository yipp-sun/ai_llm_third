# 取得数据最大max_length
import pandas as pd

df = pd.read_csv("data/data_origin.csv")
list=[]
for item in df["题目（含完整选项）"].values:
    list.append(len(item))
for item in df["答案"].values:
    list.append(len(item))
print(f"max_length:{max(list)}")
# max_length:695
