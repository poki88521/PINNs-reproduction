import scipy.io
import numpy as np
import torch


def get_index(t, n, sample_num):
    # 在保留全部时间步以及固定头尾（为了边界条件）的情况下，打乱时间步
    index_t = np.concatenate([np.array([0]),
                              np.random.choice(t - 2, t - 2, replace=False) + 1,
                              np.array([t - 1])])
    # 抽取sample_num个空间采样点
    index_x = np.random.choice(n, sample_num, replace=False)
    return index_x, index_t


def load_data(path, sample_num, is_evaluate=False):
    data = scipy.io.loadmat(path) #path should include file name
    # star: c, u, v, p, t, x, y

    c_star = data['C_star'] #n * t
    u_star = data['U_star'] #n * t
    v_star = data['V_star'] #n * t
    p_star = data['P_star'] #n * t

    t_star = data['t_star'] #t * 1
    x_star = data['x_star'] #n * 1
    y_star = data['y_star'] #n * 1

    t = t_star.shape[0]  # 201 time sample points in total
    n = x_star.shape[0]  # 30189 sample points per dimension(x, y)

    #extend t, x and y to n * t
    t_star = np.tile(t_star, (1, n)).T #t * n -> n * t
    x_star = np.tile(x_star, (1, t)) # n * t
    y_star = np.tile(y_star, (1, t)) #n * t

    #测试调用此函数时直接返回
    if is_evaluate:
        return c_star, u_star, v_star, p_star, t_star, x_star, y_star
    #获取抽取时空点的索引index
    index_x, index_t = get_index(t, n, sample_num)
    #loss_c参考点获取
    t_data_c = t_star[index_x, :][:, index_t].flatten().reshape(-1, 1) # (s * t) * 1
    x_data_c = x_star[index_x, :][:, index_t].flatten().reshape(-1, 1)
    y_data_c = y_star[index_x, :][:, index_t].flatten().reshape(-1, 1)
    c_ref = c_star[index_x, :][:, index_t].flatten().reshape(-1, 1)
    data_c = torch.FloatTensor(np.concatenate([t_data_c, x_data_c, y_data_c], axis=1)) #(s * t) * 3
    c_ref = torch.FloatTensor(c_ref) #(s * t) * 1

    #loss_e参考点获取
    index_x, index_t = get_index(t, n, sample_num) #重新抽取确保两个loss的采样点几乎不重合
    t_data_e = t_star[index_x, :][:, index_t].flatten().reshape(-1, 1) # (s * t) * 1
    x_data_e = x_star[index_x, :][:, index_t].flatten().reshape(-1, 1)
    y_data_e = y_star[index_x, :][:, index_t].flatten().reshape(-1, 1)
    data_e = torch.FloatTensor(np.concatenate([t_data_e, x_data_e, y_data_e], axis=1)) # (s * t) * 3

    return data_c, c_ref, data_e, c_star, u_star, v_star, p_star, t_star, x_star, y_star




