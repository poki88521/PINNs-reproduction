import torch
import argparse
import sys
import os
from util import fcmodel


if __name__ == '__main__':
    #获取绝对路径
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # 定义常用子目录
    DATA_DIR = os.path.join(BASE_DIR, "data")
    CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    # 自动创建目录（如果不存在）
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    # 存档路径设置
    last_model_path = os.path.join(CKPT_DIR, "last_model.pth")
    best_model_path = os.path.join(CKPT_DIR, "best_model.pth")
    checkpoint_path = os.path.join(CKPT_DIR, "checkpoint.pth")
    #args设置
    parser = argparse.ArgumentParser(description='HFM reproduction')
    parser.add_argument('--num_samples', default=157879,
                        type=int, help='number of samples: N out of 157879')
    parser.add_argument('--batch_size', default=10000,
                        type=int, help='batch size')
    parser.add_argument('--lr', default=1e-4,
                        type=float, help='learning rate')
    parser.add_argument('total_iterations', default=2e5,)
    args = parser.parse_args()
    #训练参数设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer_list = [3] + 11 * [200] + [4]
    model = fcmodel.FcModel(layer_list)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    iteration = 0
    batch_size = args.batch_size
    min_loss = 100
    running_time = 0

    #正式训练
