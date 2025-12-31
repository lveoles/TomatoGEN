import logging
import os
import sys
import importlib
import argparse
import torch
import numpy as np
import datetime
import munch
import yaml
from tqdm import tqdm  # 添加进度条支持
from dataset import *

def pc_normalize(partial_path):
    pc = np.loadtxt(partial_path, delimiter=',', usecols=(0, 1, 2)).astype(np.float32)
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    x_normalized = pc / m
    return x_normalized, centroid, m

def pc_denormal(x_normalized,centroid,m):
        """
        逆归一化：
        输入：x_normalized (B, N, 3), centroid (B, 3, 1), m (B, 1)
        输出：恢复的点云数据 (x_restored)
        """
        # 由于输入是 (B, N, 3)，我们需要将 centroid 和 m 扩展到相应的形状 (B, N, 3)
        # 确保 x_normalized 在 CPU 上
        x_normalized = x_normalized.cpu()  # 将 tensor 移到 CPU 上
        # 确保 centroid 为 tensor 类型并且在正确的设备上
        if isinstance(centroid, np.ndarray):
            centroid = torch.tensor(centroid, dtype=torch.float32).to(
                x_normalized.device)  # 转换为 tensor 并移动到与 x_normalized 相同的设备上
        # 如果 m 是一个 float32 类型的标量，确保它是一个 tensor 类型并且在正确的设备上
        if isinstance(m, float):
            m = torch.tensor(m, dtype=torch.float32).to(x_normalized.device)  # 将 m 转换为 tensor 并移动到与 x_normalized 相同的设备上
        # 恢复点云（逆标准化）
        x_restored = (x_normalized * m) + centroid
        return x_restored

def save_txt(tensor, save_path):
    """增强鲁棒性的保存函数"""
    parent_dir = os.path.dirname(save_path)
    os.makedirs(parent_dir, exist_ok=True)
    points = tensor.data.cpu().numpy()
    if points.size == 0:
        raise ValueError("输出点云数据为空")
    normalized_path = os.path.normpath(save_path).replace("“", "")
    file_name = f"{os.path.splitext(normalized_path)[0]}.txt"
    np.savetxt(file_name, points, fmt='%.6f')


def batch_processor(model, args, input_folder, output_root, device):
    """批量处理核心函数"""
    # 获取所有输入文件路径
    input_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('.txt'):
                input_files.append(os.path.join(root, file))

    # 初始化进度条
    pbar = tqdm(input_files, desc="Processing", unit="file")

    for input_path in pbar:
        try:
            # 更新进度条描述
            pbar.set_postfix(file=os.path.basename(input_path))

            # 生成输出路径
            relative_path = os.path.relpath(input_path, input_folder)
            output_filename = f"{os.path.splitext(relative_path)[0]}_result.txt"
            output_path = os.path.join(output_root, output_filename)

            # 处理单个文件
            process_single_file(model, args, input_path, output_path, device)

        except Exception as e:
            logging.error(f"处理失败 {input_path}: {str(e)}")
            continue


def process_single_file(model, args, input_path, output_path, device):
    """处理单个文件"""
    # 数据预处理
    pc, centroid, m = pc_normalize(input_path)

    # 准备数据加载器
    dataset = mydataloader(input_path, args.num_points_input, prefix="def")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False, num_workers=2)

    # 模型推理
    with torch.no_grad():
        for data in dataloader:
            label, inputs_cpu, gt_cpu = data
            inputs = inputs_cpu.float().to(device)
            gt = gt_cpu.float().to(device)
            inputs = inputs.transpose(2, 1).contiguous()
            output = model(inputs, gt, is_training=False)
            results = pc_denormal(output['out2'][0], centroid, m)

    # 保存结果
    save_txt(results, output_path)
    logging.info(f"成功保存: {output_path}")


def main():
    # 初始化日志
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # 加载配置文件
    config_path = './cfgs/PointGAN-mydata.yaml'
    args = munch.munchify(yaml.safe_load(open(config_path)))

    # 硬件配置
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # 路径配置（修改为文件夹路径）
    ##############################################
    MODEL_PATH = ("/media/zhutianyu/6D6A209A6663F595/ubuntushare/zty/PointAttN/"
                  "log/Point_Mymodel_PointGAN_cd_debug_mydata/2025-04-17T15-26-37/network.pth")
    INPUT_FOLDER = "/media/zhutianyu/6D6A209A6663F595/ubuntushare/zty/dataset-all/果实数据集预处理/半监督数据-预处理"  # 输入文件夹路径
    OUTPUT_ROOT = "/media/zhutianyu/6D6A209A6663F595/ubuntushare/zty/dataset-all/果实数据集预处理/补全结果"  # 输出根目录
    ##############################################

    # 验证路径有效性
    if not os.path.exists(INPUT_FOLDER):
        raise FileNotFoundError(f"输入路径不存在: {INPUT_FOLDER}")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    # 加载模型（只加载一次）
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = torch.nn.DataParallel(model_module.Model(args)).cuda()
    net.module.load_state_dict(torch.load(MODEL_PATH)['net_state_dict'])
    net.eval()
    logging.info(f"模型加载完成: {MODEL_PATH}")

    # 执行批量处理
    batch_processor(net, args, INPUT_FOLDER, OUTPUT_ROOT, device)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"程序异常终止: {str(e)}")
        sys.exit(1)