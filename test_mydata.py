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
from dataset import *  # 导入自定义数据集类


def save_txt(tensor, save_path):
    """增强鲁棒性的保存函数"""
    # 创建父目录
    parent_dir = os.path.dirname(save_path)
    os.makedirs(parent_dir, exist_ok=True)  # 关键修复点1：自动创建目录
    # 处理张量数据
    points = tensor.data.cpu().numpy()  # [3, N] -> [N, 3]
    # 验证数据有效性
    if points.size == 0:
        raise ValueError("输出点云数据为空，请检查模型输出")
    # 标准化文件路径（解决中文符号问题）
    normalized_path = os.path.normpath(save_path).replace("“", "")  # 关键修复点2：去除非法字符
    file_name = f"{os.path.splitext(normalized_path)[0]}.txt"
    # 保存前打印验证路径
    print(f"保存路径验证：{file_name}")  # 调试用
    np.savetxt(file_name, points, fmt='%.6f')

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

def main():
    print('0')
    config_path = os.path.join('./cfgs/TomatoGAN-mydata.yaml')
    args = munch.munchify(yaml.safe_load(open(config_path)))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    print('0')
    # 硬编码路径配置（可根据需要修改这些变量）
    ##############################################
    MODEL_PATH = ("/media/zhutianyu/6D6A209A6663F595/ubuntushare/zty/PointAttN/log/"
                  "PointAttN_cd_debug_c3d/2025-03-21T16-34-27/network.pth")  # 预训练模型路径
    INPUT_TXT = ("/media/zhutianyu/6D6A209A6663F595/ubuntushare/zty/dataset-all/果实数据增强/train/partial/揉谷/"
                    "1-label_label_1-task1-ratio60-part-perturb-4-0.6000.txt")  # 输入点云文件
    OUTPUT_PATH = ("/media/zhutianyu/6D6A209A6663F595/ubuntushare/zty/PointAttN/log/"
                   "PointAttN_cd_debug_c3d/2025-03-21T16-34-27/")
    #输出结果路径
    GPU_ID = args.device # 使用的GPU编号
    ##############################################
    # 正确做法
    base_name = os.path.basename(INPUT_TXT)  # 获取纯文件名
    file_name = os.path.splitext(base_name)[0]  # 正确分割
    OUTPUT_TXT = file_name+ '_test1225.txt'
    OUTPUT_dir = os.path.join(OUTPUT_PATH, OUTPUT_TXT)
    print('1')
    # try:
        # 加载模型
        # load model
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = torch.nn.DataParallel(model_module.Model(args))
    net.cuda()
    print('2')
    net.module.load_state_dict(torch.load(MODEL_PATH)['net_state_dict'])
    net.eval()
    print(f"成功加载模型: {MODEL_PATH}")
    pc, centroid, m = pc_normalize(INPUT_TXT)
    dataset_test = mydataloader(INPUT_TXT, args.num_points_input,prefix="def")
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                  shuffle=False, num_workers=1)
    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            label, inputs_cpu, gt_cpu = data
            print(f"输入点云形状: {inputs_cpu.shape}")
            inputs = inputs_cpu.float().cuda()
            gt = gt_cpu.float().cuda()
            inputs = inputs.transpose(2, 1).contiguous()
            output = net(inputs, gt, is_training=False)
            resluts=pc_denormal(output['out2'][0],centroid,m)
    # 保存结果
    save_txt(resluts, OUTPUT_dir)
    print(f"结果已保存至: {OUTPUT_dir}")
    # except Exception as e:
    #     # print(f"运行出错: {str(e)}")
    #     # sys.exit(1)

if __name__ == "__main__":
    main()