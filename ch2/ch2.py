import torch



x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])

X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

Z = torch.zeros_like(Y)
print(Z)
print('id(Z): ', id(Z))
Z[:] = X + Y
print(Z)
print('id(Z): ', id(Z))
