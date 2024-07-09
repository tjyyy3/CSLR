import torch
import numpy as np


def transform():
    a = np.identity(84)
    x = 42
    for i in range(1, 84):
        if i % 2 == 1:
            a = np.insert(a, i, values=a[x], axis=0)
            x = x + 2
    b = torch.tensor(a[0:84], dtype=torch.float32)

    return b

# print(b.shape)
# t_one = torch.ones((2, 6, 6))
# t_two = 2 * t_one
# c = torch.cat((t_one, t_two), 1)
# print(b.shape)
# print(c.shape)
# merge = torch.matmul(b, c)
# print(merge.shape)
