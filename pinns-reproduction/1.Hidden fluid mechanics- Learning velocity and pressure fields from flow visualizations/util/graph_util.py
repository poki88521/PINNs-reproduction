import matplotlib.pyplot as plt

#各个输出的相对L2误差，曲线图，散点暂未加入
def error_graph(graph_path, version_name, iterations, error_c, error_u, error_v, error_p):
    fig, axs = plt.subplots(nrows=2, ncols=2)
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

    fig.savefig(graph_path, fname=version_name + '_error_graph_.png')
    pass

def colormesh_graph(graph_path, version_name,
                    c_ref, c_pred, u_ref, u_pred, v_ref, v_pred, p_ref, p_pred):


