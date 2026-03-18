import torch
import torch.nn as nn

# t1 = torch.arange(6.0)
# drop_out_layer = nn.Dropout(p=0.5)
# t2 = drop_out_layer(t1)
# print(t2)

# layer = nn.Linear(3,5,bias=True)
# t1 = torch.randint(0, 10, (2,3)).float()
# print(t1)
# output = layer(t1)
# print(output)

# 线性变换，输入维度是3，输出维度是5，输入的张量是2行3列的矩阵，输出的张量是2行5列的矩阵

# t = torch.tensor([[1,2,3],[4,5,6]])
# t.view(3,2)
# print(t.view(3,2))

# t = torch.tensor([[1,2,3],[4,5,6]])
# print(t.shape)
# print(t.transpose(0,1),t.shape)

t = torch.arange(1,13).view(3,4)
print(t)
t1= torch.triu(t,diagonal=1)
print(t1)