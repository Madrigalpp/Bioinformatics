import torch

y = torch.tensor([10, 5, 8])

# 创建一个所有元素为1的矩阵
mask = torch.ones(len(y), len(y))

# 将生存时间不一致的样本对应的mask元素设置为0
mask[(y.T - y) > 0] = 0

print(mask)
