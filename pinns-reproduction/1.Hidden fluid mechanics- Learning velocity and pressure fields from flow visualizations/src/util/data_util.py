
import scipy.io
import numpy as np


def load_data(path):
    data = scipy.io.loadmat(path) #path should include file name
    # star: c, u, v, p, t, x, y

    c_star = data['c_star'] #n * t
    u_star = data['u_star'] #n * t
    v_star = data['v_star'] #n * t
    p_star = data['p_star'] #n * t

    t_star = data['t_star'] #t * 1
    x_star = data['x_star'] #n * 1
    y_star = data['y_star'] #n * 1

    t = t_star.shape[0]  # 201 time sample points in total
    n = x_star.shape[0]  # 30189 sample points per dimension(x, y)

    #extend t, x and y to n * t
    t_star = np.tile(t_star, (1, n)).T #t * n -> n * t
    x_star = np.tile(x_star, (1, t)) # n * t
    y_star = np.tile(y_star, (1, t)) #n * t



