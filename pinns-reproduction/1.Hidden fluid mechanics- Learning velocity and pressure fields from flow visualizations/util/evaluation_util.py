import numpy as np
import torch

def test_data(c_star, u_star, v_star, p_star, t_star, x_star, y_star):
    snap = np.random.randint(0, 200)
    c_snap = c_star[:, snap: snap + 1] #n * 1
    u_snap = u_star[:, snap: snap + 1]
    v_snap = v_star[:, snap: snap + 1]
    p_snap = p_star[:, snap: snap + 1]
    t_snap = t_star[:, snap: snap + 1]
    x_snap = x_star[:, snap: snap + 1]
    y_snap = y_star[:, snap: snap + 1]

    variable_star = torch.FloatTensor(np.concatenate([t_snap, x_snap, y_snap], axis=1))
    target_star = torch.FloatTensor(np.concatenate([c_snap, u_snap, v_snap, p_snap], axis=1))
    return variable_star, target_star

def relative_l2_error(pred, target):
    #要进行中心化操作
    pred_mean = torch.mean(pred)
    target_mean = torch.mean(target)
    return torch.sqrt(torch.mean((pred_mean - target_mean)**2)
                      / torch.mean(target_mean**2)).cpu().numpy()


