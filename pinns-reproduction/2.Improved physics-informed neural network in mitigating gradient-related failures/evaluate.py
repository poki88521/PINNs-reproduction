import argparse
import os
import torch
import numpy as np
import pandas as pd

from util import fcmodel, data_util, evaluation_util, graph_util

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    GRAPH_DIR = os.path.join(BASE_DIR, "graphs")
    parser = argparse.ArgumentParser(description='HFM evaluation')
    parser.add_argument('--version', default='V1.0', type=str, help='version name')
    parser.add_argument('--datapath', default=os.path.join(DATA_DIR, 'Cylinder2D_flower.mat'),
                        type=str, help='data path')
    parser.add_argument('--model_path', default=os.path.join(CKPT_DIR, "V1.0_best_model.pth"),
                        type=str, help='model path')
    parser.add_argument('--log_path', default=os.path.join(LOG_DIR, "V1.0_log.csv"),
                        type=str, help='log path')
    args = parser.parse_args()
    #参数设置
    print(f"version name:{args.version}")
    model_path = args.model_path
    data_path = args.datapath
    log_path = args.log_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer_list = [3] + 11 * [200] + [4]
    model = fcmodel.FcModel(layer_list).to(device)
    snap = np.random.randint(135, 145)
    #获取数据集数据
    print("loading data...")
    (c_star, u_star, v_star, p_star,
     t_star, x_star, y_star) = data_util.load_data(data_path, 30189, is_evaluate=True)
    #根据模型获取单个时间采样点的预测数据
    print("loading model...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    with torch.no_grad():
        variable_star, target_star = evaluation_util.test_data(c_star, u_star, v_star, p_star,
                                                               t_star, x_star, y_star, snap=snap)
        data_pred = model(variable_star.to(device))
        c_pred = data_pred[:, 0:1]
        u_pred = data_pred[:, 1:2]
        v_pred = data_pred[:, 2:3]
        p_pred = data_pred[:, 3:4]
        print("making colormap graph...")
        graph_util.colormap_graph(GRAPH_DIR, args.version,
                                  x_star[:, snap:snap+1], y_star[:, snap:snap+1],
                                  c_star[:, snap:snap+1], c_pred,
                                  u_star[:, snap:snap+1], u_pred,
                                  v_star[:, snap:snap+1], v_pred,
                                  p_star[:, snap:snap+1], p_pred)

    print("making error graph...")
    log = pd.read_csv(log_path)
    log = log.dropna(how='all')
    graph_util.error_graph(GRAPH_DIR, args.version, log['iterations'],
                           log['error_c'], log['error_u'], log['error_v'], log['error_p'])
