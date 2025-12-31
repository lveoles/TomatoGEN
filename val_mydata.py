import logging
import os
import sys
import importlib
import argparse
import munch
import yaml
import torch
import numpy as np
from utils.train_utils import *
from dataset import mydataloader  # 根据实际情况可能需要修改


def load_txt_to_tensor(txt_path):
    """从txt文件加载点云数据并转换为PyTorch张量"""
    # 读取txt文件，每行包含3个坐标值
    points = np.loadtxt(txt_path, dtype=np.float32)
    # 转换为PyTorch张量并添加批次和通道维度 [1, 3, N]
    tensor = torch.from_numpy(points.T).unsqueeze(0)  # 转置为[3, N]后添加批次维度
    return tensor


def save_txt(data_tensor, save_path):
    """将模型输出张量保存为txt文件"""
    # 转换为numpy数组并转置为[N, 3]
    points = data_tensor.squeeze(0).cpu().numpy().T
    # 保留4位小数
    np.savetxt(save_path, points, fmt='%.4f')


def inference(args):
    """执行推理流程"""
    # 加载模型
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = model_module.Model(args)  # 移除DataParallel

    # 加载权重（处理多GPU训练保存的模型）
    state_dict = torch.load(args.load_model)['net_state_dict']
    # 如果保存的是DataParallel模型，需要去除模块前缀
    if next(iter(state_dict)).startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    net.load_state_dict(state_dict)
    net.cuda()
    net.eval()
    logging.info("Model loaded successfully from: %s", args.load_model)

    # 加载输入数据
    input_tensor = load_txt_to_tensor(args.input_txt).cuda()

    with torch.no_grad():
        # 执行推理（假设模型接受单个输入）
        output = net(input_tensor, is_training=False)['out2']  # 根据实际输出结构调整

    # 保存结果
    save_txt(output, args.output_txt)
    logging.info("Result saved to: %s", args.output_txt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Completion Inference')
    # 必需参数
    parser.add_argument('-m', '--load_model', required=True, help='Path to .pth model file')
    parser.add_argument('-i', '--input_txt', required=True, help='Path to input .txt file')
    parser.add_argument('-o', '--output_txt', required=True, help='Path to output .txt file')
    # 可选参数
    parser.add_argument('-c', '--config', help='Path to config file', default='config.yaml')
    parser.add_argument('--device', default='0', help='CUDA device ID')
    args = parser.parse_args()
    # 加载配置文件
    if args.config:
        config_path = os.path.join('./cfgs', args.config)
        args = munch.munchify(yaml.safe_load(open(config_path)))
    # 设置设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.splitext(args.output_txt)[0] + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    # 验证文件路径
    if not os.path.exists(args.input_txt):
        raise FileNotFoundError(f"Input file not found: {args.input_txt}")
    os.makedirs(os.path.dirname(args.output_txt), exist_ok=True)
    # 执行推理
    inference(args)