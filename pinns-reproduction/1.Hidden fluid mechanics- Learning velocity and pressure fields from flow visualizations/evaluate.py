import argparse
import os
from util import fcmodel, data_util, pinns, evaluation_util

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    parser = argparse.ArgumentParser(description='HFM evaluation')
    parser.add_argument('--version', default='V1.0', type=str, help='version name')
    parser.add_argument('--datapath', default=os.path.join(DATA_DIR, 'Cylinder2D_flower.mat'),
                        type=str, help='data path')
    #获取数据集数据

    #根据模型获取预测数据

    #计算误差