import torch
import numpy as np

def derivative(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]

def ns_2d(c, u, v, p, txy, Pe, Re):
    #batch_size * 1 -> batch_size * 3
    c_txy = derivative(c, txy)
    u_txy = derivative(u, txy)
    v_txy = derivative(v, txy)
    p_txy = derivative(p, txy)

    #batch_size * 3 -> batch_size * 1
    c_t = c_txy[:, 0:1]
    c_x = c_txy[:, 1:2]
    c_y = c_txy[:, 2:3]
    u_t = u_txy[:, 0:1]
    u_x = u_txy[:, 1:2]
    u_y = u_txy[:, 2:3]
    v_t = v_txy[:, 0:1]
    v_x = v_txy[:, 1:2]
    v_y = v_txy[:, 2:3]
    p_t = p_txy[:, 0:1]
    p_x = p_txy[:, 1:2]
    p_y = p_txy[:, 2:3]

    c_xx = derivative(c_x, txy)[:, 1:2]
    c_yy = derivative(c_y, txy)[:, 2:3]
    u_xx = derivative(u_x, txy)[:, 1:2]
    u_yy = derivative(u_y, txy)[:, 2:3]
    v_xx = derivative(v_x, txy)[:, 1:2]
    v_yy = derivative(v_y, txy)[:, 2:3]

    #loss
    e1 = c_t + u * c_x + v * c_y - (1 / Pe) * (c_xx + c_yy)
    e2 = u_t + u * u_x + v * u_y + p_x - (1 / Re) * (u_xx + u_yy)
    e3 = v_t + u * v_x + v * v_y + p_y - (1 / Re) * (v_xx + v_yy)
    e4 = u_x + v_y

    return e1, e2, e3, e4

