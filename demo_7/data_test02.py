import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# 读取CSV文件
# csv_file_path = "D:/Workspace/llm/data/Weibo/train.csv"
# csv_file_path = "D:/Workspace/llm/data/Weibo/test.csv"
csv_file_path = "D:/Workspace/llm/data/Weibo/validation.csv"
df = pd.read_csv(csv_file_path)

# 定义重采样策略
# 如果想要过采样，使用RandomOverSampler
# 如果想要欠采样，使用RandomUnderSampler
# 我们在这里使用RandomUnderSampler进行欠采样
# random_state控制随机数生成器的种子
rus = RandomUnderSampler(sampling_strategy="auto", random_state=42)

# 将特征和标签分开
X = df[["text"]]
Y = df[["label"]]
print(Y)

# 应用重采样
X_resampled, Y_resampled = rus.fit_resample(X, Y)
print(Y_resampled)
# 合并特征和标签，创建系的DataFrame
df_resampled = pd.concat([X_resampled, Y_resampled], axis=1)

print(df_resampled)

# 保存均衡数据到新的csv文件
# df_resampled.to_csv("new_train.csv", index=False)
# df_resampled.to_csv("new_test.csv", index=False)
df_resampled.to_csv("validation.csv", index=False)
