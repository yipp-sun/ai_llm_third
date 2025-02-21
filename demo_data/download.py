# 数据集下载
from modelscope.msdatasets import MsDataset

ds = MsDataset.load('liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT', subset_name='default', split='train',
                    cache_dir='data')
# 您可按需配置 subset_name、split，参照“快速使用”示例代码

# 加载本地的arrow文件：load_from_disk
# from datasets import load_from_disk
#
# path = 'data'  # train：表示上述训练集在本地的路径
# dataset = load_from_disk(path)

# 从 git官网 下载windows版本的git，在你要放克隆项目的工作空间中启动cmd命令，执行下属克隆命令
# git clone https://www.modelscope.cn/datasets/liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT.git
