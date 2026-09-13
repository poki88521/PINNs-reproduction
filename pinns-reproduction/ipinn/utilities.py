import numpy as np

class Sampler:
    def __init__(self, coords, sample_num, function, dims=2, name=None):
        self.coords = coords
        self.sample_num = sample_num
        self.function = function
        self.dims = dims
        self.name = name

    def sample(self):
        sample_coords = (self.coords[0:1, :] +
                         (self.coords[0:1, :] - self.coords[1:2, :]) * np.random.rand(self.sample_num, self.dims))
        x = sample_coords[:, 0]
        y = sample_coords[:, 1]
        z = self.function(x, y)
        return x, y, z

#解析解u(x, y)
def u(x, y, a1=1, a2=4):
    return np.sin(a1 * np.pi * x) * np.sin(a2 * np.pi * y)

def f(x, y, k=1.0, a1=1, a2=4):
    u_xx = - (a1 * np.pi) ** 2 * u(x, y, a1, a2)
    u_yy = - (a2 * np.pi) ** 2 * u(x, y, a1, a2)
    return u_xx + u_yy + k ** 2 * u(x, y, a1, a2)

