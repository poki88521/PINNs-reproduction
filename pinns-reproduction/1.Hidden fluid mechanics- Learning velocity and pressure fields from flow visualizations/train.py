import time
import torch
import numpy as np
import argparse
import csv
import os
from util import fcmodel, data_util, pinns, evaluation_util


def update_checkpoint(checkpoint, iteration, min_loss, running_time, model, optimizer):
    checkpoint["iteration"] = iteration
    checkpoint["min_loss"] = min_loss
    checkpoint["running_time"] = running_time
    checkpoint["model_state_dict"] = model.state_dict()
    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    return checkpoint

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
    best_model_path = os.path.join(CKPT_DIR, "best_model.pth")
    checkpoint_path = os.path.join(CKPT_DIR, "checkpoint.pth")
    log_path = os.path.join(LOG_DIR, "log.csv")
    # args设置
    parser = argparse.ArgumentParser(description='HFM reproduction')
    parser.add_argument('--version', default='V1.0', type=str, help='version name')
    parser.add_argument('--datapath', default=os.path.join(DATA_DIR, 'Cylinder2D_flower.mat'),
                        type=str, help='data path')
    parser.add_argument('--Re', default=100,
                        type=int, help='Reynoids number')
    parser.add_argument('--Pe', default=100,
                        type=int, help='Peclet number')
    parser.add_argument('--num_samples', default=157879,
                        type=int, help='number of samples: N out of 157879')
    parser.add_argument('--batch_size', default=5000,
                        type=int, help='batch size')
    parser.add_argument('--lr', default=1e-3,
                        type=float, help='learning rate')
    parser.add_argument('--total_iterations', default=200000,
                        type=int, help='total number of iterations')
    args = parser.parse_args()
    #训练参数设置
    data_path = args.datapath
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer_list = [3] + 11 * [200] + [4]
    model = fcmodel.FcModel(layer_list).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    iteration = 0
    batch_size = args.batch_size
    min_loss = 100
    start_time = time.time()
    running_time = 0
    checkpoint = {"iteration": iteration, "min_loss": min_loss, "running_time": running_time,
                  "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}
    #建立日志文件（如果没有）
    if not os.path.exists(log_path):
        with open(log_path, mode='w') as f:
            writer = csv.writer(f).writerow(["iterations", "loss_c", "loss_e", "total_loss",
                                             "error_c", "error_u", "error_v", "error_p"])
    #加载存档（如果有）
    print("loading checkpoint...")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        iteration = checkpoint["iteration"]
        min_loss = checkpoint["min_loss"]
        running_time = checkpoint["running_time"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("checkpoint loaded successfully")
    else:
        print("checkpoint not exist, start a new training")
    #加载数据
    print("loading data...")
    (data_c, c_ref, data_e,
     c_star, u_star, v_star, p_star,
     t_star, x_star, y_star) = data_util.load_data(data_path, batch_size)
    #正式训练
    print("start training the model...")
    while iteration < args.total_iterations:
        optimizer.zero_grad()
        #抽取单批量数据
        index_data_c_batch = np.random.choice(args.num_samples, batch_size)
        index_data_e_batch = np.random.choice(args.num_samples, batch_size)
        data_c_batch = data_c[index_data_c_batch, :].to(device)
        c_ref_batch = c_ref[index_data_c_batch, :].to(device)
        data_e_batch = data_e[index_data_e_batch, :].to(device)
        data_c_batch.requires_grad = True
        data_e_batch.requires_grad = True
        #产生预测结果
        c_c_pred = model(data_c_batch)[:, 0:1]

        e_pred = model(data_e_batch)
        c_pred = e_pred[:, 0:1]
        u_pred = e_pred[:, 1:2]
        v_pred = e_pred[:, 2:3]
        p_pred = e_pred[:, 3:4]

        e1, e2, e3, e4 = pinns.ns_2d(c_pred, u_pred, v_pred, p_pred,
                                     data_e_batch,
                                     args.Pe, args.Re)
        loss_c = torch.mean((c_ref_batch - c_c_pred) ** 2)
        loss_e = torch.mean(e1 ** 2) + torch.mean(e2 ** 2) + torch.mean(e3 ** 2) + torch.mean(e4 ** 2)
        loss = loss_c + loss_e
        loss.backward()
        optimizer.step()

        #best model
        if loss < min_loss:
            min_loss = loss.item()
            torch.save(model.state_dict(), best_model_path)

        #print data and save checkpoint per 100 iterations
        if iteration % 100 == 0:
            #print data
            elapsed = time.time() - start_time
            running_time += elapsed / 3600.0
            print('Iteration: %d, Loss: %.4f, Time: %.2fs, Running Time: %.2fh'
                  % (iteration, loss.item(), elapsed, running_time))
            #checkpoint
            checkpoint = update_checkpoint(checkpoint, iteration, min_loss, running_time, model, optimizer)
            torch.save(checkpoint, checkpoint_path)
            start_time = time.time()

        #calculate relative error and save result in log per 1000 iterations
        if iteration % 1000 == 0 and iteration != 0:
            with torch.no_grad():
                variable_star, target_star = evaluation_util.test_data(c_star, u_star, v_star, p_star,
                                                       t_star, x_star, y_star)
                data_pred = model(variable_star.to(device))
                c_pred = data_pred[:, 0:1]
                u_pred = data_pred[:, 1:2]
                v_pred = data_pred[:, 2:3]
                p_pred = data_pred[:, 3:4]

                c_error = evaluation_util.relative_l2_error(c_pred, target_star[:, 0:1].to(device))
                u_error = evaluation_util.relative_l2_error(u_pred, target_star[:, 1:2].to(device))
                v_error = evaluation_util.relative_l2_error(v_pred, target_star[:, 2:3].to(device))
                p_error = evaluation_util.relative_l2_error(p_pred, target_star[:, 3:4].to(device))
                print('Error: c: %.3f, u: %.3f, v: %.3f, p: %.3f' % (c_error, u_error, v_error, p_error))
                #save result in log
                with open(log_path, 'a') as f:
                    csv.writer(f).writerow([iteration, loss_c.item(), loss_e.item(), loss.item(),
                                            c_error, u_error, v_error, p_error])
        iteration += 1



