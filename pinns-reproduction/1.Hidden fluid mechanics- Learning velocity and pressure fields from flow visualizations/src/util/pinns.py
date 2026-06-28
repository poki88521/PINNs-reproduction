import torch

def derivative(x, y):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]

def ns_2d(c, u, v, p, t, x, y):
