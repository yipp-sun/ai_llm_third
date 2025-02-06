import torch

# x = torch.rand(5, 3)
# print(x)

print(torch.version.cuda)
print(torch.cuda.is_available())

# !/usr/bin/python

list1 = ['physics', 'chemistry', 1997, 2000]
list2 = [0, 1, 2, 3, 4, 5, 6, 7]
# 存储对象[start : end : step]
# 列表索引从0开始
print("list1[0]: ", list1[0])
# 返回最后一个数据
print("list1[-1]: ", list1[-1])
# 返回从1到0的数据,故返回第二个到最后一个的数据（不包含结束索引位置0）
print("list1[1:]: ", list1[1:])
# 返回从-1到0的数据，故返回最后一个数据
print("list1[-1:]: ", list1[-1:])
# 返回从0到-1的数据，故返回第一个到倒数第二个的数据（不包含结束索引位置-1）
print("list1[:-1]: ", list1[:-1])
# 表示步长为1，步长大于0时，返回序列为原顺序；。
print("list1[::1]: ", list1[::1])
# list[::-1]: 表示从右往左以步长为1进行切片。步长小于0时，返回序列为倒序
print("list1[::-1]: ", list1[::-1])
# list[::2]: 表示从左往右步长为2进行切片==>取奇数
print("list1[::2]: ", list1[::2])
# ==>取偶数
print("list1[1::2]: ", list1[1::2])
print("list2[0:5]: ", list2[0:4])
print("list2[1:5]: ", list2[1:5])

'''切片健壮性的体现'''
# 使用切片操作就不会产生该问题，会自动截断或者返回空列表。
ll = [5, 17, 13, 14, 8, 19, 3, 7, 9, 12]
print(ll[0:20:3])
# 就是说，不会产生下标越界问题
print(ll[21:])
# 下标越界
# print(ll[21])

list = []
# list.append(obj)在列表末尾添加新的对象
list.append('Google')
list.append('Runoob')
print(list)
