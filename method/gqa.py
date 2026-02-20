import torch
import torch.nn as nn

t1 = torch.tensor([1,2,3,4,5])
drop_out_layer = nn.Dropout(p=0.5)
t2 = drop_out_layer(t1)
print(t2)