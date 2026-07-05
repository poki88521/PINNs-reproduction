import os.path

import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from matplotlib import cm

#各个输出的相对L2误差，曲线图，散点暂未加入
def error_graph(graph_dir, version_name, iterations, error_c, error_u, error_v, error_p):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    fig.suptitle('relative L2 error')
    c_graph, u_graph, v_graph, p_graph = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    c_graph.plot(iterations, error_c, label="concentration", color="red")
    c_graph.set_xlabel("iterations")
    c_graph.set_ylabel("rel_L2_error")
    c_graph.legend()

    u_graph.plot(iterations, error_u, label="velocity of x", color="blue")
    u_graph.set_xlabel("iterations")
    u_graph.set_ylabel("rel_L2_error")
    u_graph.legend()

    v_graph.plot(iterations, error_v, label="velocity of y", color="green")
    v_graph.set_xlabel("iterations")
    v_graph.set_ylabel("rel_L2_error")
    v_graph.legend()

    p_graph.plot(iterations, error_p, label="pressure", color="yellow")
    p_graph.set_xlabel("iterations")
    p_graph.set_ylabel("rel_L2_error")
    p_graph.legend()

    fig.savefig(os.path.join(graph_dir, f'{version_name}_error_graph_.png'))
    pass

#随机抽取一个时间样本点（t在135到145之间）绘制c, u, v, p的参考与预测值的色彩图
def colormap_graph(graph_dir, version_name, x_star, y_star,
                    c_ref, c_pred, u_ref, u_pred, v_ref, v_pred, p_ref, p_pred):
    p_ref = p_ref - p_ref.mean()
    p_pred = p_pred - p_pred.mean()
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
    axs[0, 0].scatter(x_star, y_star, c=c_ref, cmap='jet', vmin=c_ref.min(), vmax=c_ref.max())
    axs[0, 0].set_xlim(0, 6)
    axs[0, 0].set_ylim(-3, 3)
    axs[0, 0].set_title("c_reference")

    axs[0, 1].scatter(x_star, y_star, c=c_pred, cmap='jet', vmin=c_ref.min(), vmax=c_ref.max())
    axs[0, 1].set_xlim(0, 6)
    axs[0, 1].set_ylim(-3, 3)
    axs[0, 1].set_title("c_prediction")

    axs[0, 2].scatter(x_star, y_star, c=u_ref, cmap='jet', vmin=u_ref.min(), vmax=u_ref.max())
    axs[0, 2].set_xlim(0, 6)
    axs[0, 2].set_ylim(-3, 3)
    axs[0, 2].set_title("u_reference")

    axs[0, 3].scatter(x_star, y_star, c=u_pred, cmap='jet', vmin=u_ref.min(), vmax=u_ref.max())
    axs[0, 3].set_xlim(0, 6)
    axs[0, 3].set_ylim(-3, 3)
    axs[0, 3].set_title("u_prediction")

    axs[1, 0].scatter(x_star, y_star, c=v_ref, cmap='jet', vmin=v_ref.min(), vmax=v_ref.max())
    axs[1, 0].set_xlim(0, 6)
    axs[1, 0].set_ylim(-3, 3)
    axs[1, 0].set_title("v_reference")

    axs[1, 1].scatter(x_star, y_star, c=v_pred, cmap='jet', vmin=v_ref.min(), vmax=v_ref.max())
    axs[1, 1].set_xlim(0, 6)
    axs[1, 1].set_ylim(-3, 3)
    axs[1, 1].set_title("v_prediction")

    axs[1, 2].scatter(x_star, y_star, c=p_ref, cmap='jet', vmin=p_ref.min(), vmax=p_ref.max())
    axs[1, 2].set_xlim(0, 6)
    axs[1, 2].set_ylim(-3, 3)
    axs[1, 2].set_title("p_reference")

    axs[1, 3].scatter(x_star, y_star, c=p_pred, cmap='jet', vmin=p_ref.min(), vmax=p_ref.max())
    axs[1, 3].set_xlim(0, 6)
    axs[1, 3].set_ylim(-3, 3)
    axs[1, 3].set_title("p_prediction")

    fig.savefig(os.path.join(graph_dir, f'{version_name}_colormap_graph_.png'))
