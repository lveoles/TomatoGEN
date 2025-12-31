import os
import open3d as o3d
import torch
import numpy as np
import torch.utils.data as data
import h5py
import math
import transforms3d
import random
from utils.mm3d_pn2 import furthest_point_sample
from tensorpack import dataflow

class mydataloader(data.Dataset):
    # 自定义数据集类，继承自torch的Dataset
    def __init__(self, path, samplepoints,prefix="train"):
        # 初始化方法
        # path: 数据集根目录
        # prefix: 数据集类型标识（train/val/test）
        # 设置数据子集路径
        if prefix == "train":
            self.file_path = os.path.join(path, 'train')
        elif prefix == "test":
            self.file_path = os.path.join(path, 'test')
        elif prefix == "def":
            self.file_path = path
        else:
            raise ValueError("ValueError prefix should be [train/test] ")
        self.prefix = prefix  # 保存数据集类型标识
        self.label_map = {'揉谷': '0', '延安': '1', 'all': '2'}  # 中文标签到数字的映射
        # 加载数据
        if prefix == "train":  # 修正：应使用 != 代替 is not
            # 训练/验证集加载partial和gt数据
            self.input_data, self.labels = self.get_data(os.path.join(self.file_path, 'partial'))
            self.gt_data, _ = self.get_data(os.path.join(self.file_path, 'gt'))
            print(len(self.gt_data), len(self.labels))  # 验证数据一致性
        elif prefix == "test":
            # 测试集只加载partial数据
            self.input_data, self.labels = self.get_data(os.path.join(self.file_path, 'partial'))
        elif prefix == "def":
            self.input_data, self.labels = self.get_datadef(self.file_path)
        # print(len(self.input_data))  # 打印加载的数据量
        self.len = len(self.input_data)  # 数据集长度
        # 数据增强参数初始化
        self.sample = 1  # 是否进行采样
        self.samplepoints=samplepoints

    def __len__(self):
        # 返回数据集总长度
        return self.len

    def pc_normalize(self,pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def get_data(self, path):
        # 数据加载核心方法
        # path: 要加载的数据路径
        cls = os.listdir(path)  # 获取类别目录列表
        data = []  # 存储文件路径
        labels = []  # 存储对应标签
        for c in cls:  # 遍历每个类别目录
            category_path = os.path.join(path, c)
            if not os.path.isdir(category_path):
                continue  # 跳过非目录项
            # 递归遍历所有子目录
            for root, _, files in os.walk(category_path):
                for file in files:# 遍历每个文件
                    file_path = os.path.join(root, file)# # 构建完整文件路径
                    data.append(file_path)
                    # 标签处理：测试集使用文件名，其他用映射表
                    if self.prefix == "test":
                        labels.append(file)  # 测试集标签直接使用文件名
                    else:
                        labels.append(self.label_map[c])  # 通过映射表转换标签
        return data, labels

    def get_datadef(self, path):
        data = []  # 存储文件路径
        labels = []  # 存储对应标签
        data.append(path)
        file = os.path.basename(path)
        split_name = os.path.splitext(file)
        # 标签处理：测试集使用文件名，其他用映射表
        labels.append(split_name[0])  # 测试集标签直接使用文件名
        return data, labels

    def __getitem__(self, index):
        # 获取单个数据样本的核心方法
        partial_path = self.input_data[index]
        # 获取partial数据路径
        # 例子：'D:\\实验数据\\2024-11-14\\果实数据增强\\train\\partial\\延安\\10\\2416118 - label - Cloud_label_1.txt'
        # 加载partial数据
        partial = np.loadtxt(partial_path, delimiter=',',usecols=(0, 1, 2)).astype(np.float32)  # 读取数据并转换为float32
        # 通过np.round将数据四舍五入至小数点后3位
        partial = np.round(partial, decimals=4)
        partial = self.pc_normalize(partial)
        # 训练时的下采样处理
        if self.prefix == 'train' and self.sample:
            # 随机选择args.num_points个点
            choice = np.random.permutation((partial.shape[0]))
            partial = partial[choice[:self.samplepoints]]
            # 不足点数时补零（可能存在问题，见注释）
            if partial.shape[0] < self.samplepoints:
                zeros = np.zeros((self.samplepoints - partial.shape[0], 3))
                partial = np.concatenate([partial, zeros])
        # 非测试集处理流程（只对val和train作处理
        if self.prefix not in ["test","def"]:
            complete_path = partial_path.replace('partial', 'gt')  # 构造gt路径
            complete = np.loadtxt(complete_path, delimiter=',',usecols=(0, 1, 2)).astype(np.float32)  # 读取数据并转换为float32
            complete = self.pc_normalize(complete)
            # 转换为Tensor
            complete = torch.from_numpy(complete).float()
            label = int(self.labels[index])  # 转换标签为整数
            partial = torch.from_numpy(partial).float()
            return label, partial, complete
        # 测试集处理流程
        else:
            # 随机选择1024个点
            # partial=farthest_point_sampling_with_density(partial,self.samplepoints)
            choice = np.random.permutation((partial.shape[0]))
            partial = partial[choice[:self.samplepoints]]
            partial = torch.from_numpy(partial).float()
            label = self.labels[index]  # 测试集标签保持原始值
            return label, partial, partial  # 返回两次partial作为占位符

class mydataloaderrgb(data.Dataset):
    # 自定义数据集类，继承自torch的Dataset
    def __init__(self, path, samplepoints,prefix="train"):
        # 初始化方法
        # path: 数据集根目录
        # prefix: 数据集类型标识（train/val/test）
        # 设置数据子集路径
        if prefix == "train":
            self.file_path = os.path.join(path, 'train')
        elif prefix == "test":
            self.file_path = os.path.join(path, 'test')
        elif prefix == "def":
            self.file_path = path
        else:
            raise ValueError("ValueError prefix should be [train/test] ")
        self.prefix = prefix  # 保存数据集类型标识
        self.label_map = {'揉谷': '0', '延安': '1', 'all': '2'}  # 中文标签到数字的映射
        # 加载数据
        if prefix == "train":  # 修正：应使用 != 代替 is not
            # 训练/验证集加载partial和gt数据
            self.input_data, self.labels = self.get_data(os.path.join(self.file_path, 'partial'))
            self.gt_data, _ = self.get_data(os.path.join(self.file_path, 'gt'))
            print(len(self.gt_data), len(self.labels))  # 验证数据一致性
        elif prefix == "test":
            # 测试集只加载partial数据
            self.input_data, self.labels = self.get_data(os.path.join(self.file_path, 'partial'))
        elif prefix == "def":
            self.input_data, self.labels = self.get_datadef(self.file_path)
        # print(len(self.input_data))  # 打印加载的数据量
        self.len = len(self.input_data)  # 数据集长度
        # 数据增强参数初始化
        self.sample = 1  # 是否进行采样
        self.samplepoints=samplepoints

    def __len__(self):
        # 返回数据集总长度
        return self.len

    def pc_normalize(self,pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def get_data(self, path):
        # 数据加载核心方法
        # path: 要加载的数据路径
        cls = os.listdir(path)  # 获取类别目录列表
        data = []  # 存储文件路径
        labels = []  # 存储对应标签
        for c in cls:  # 遍历每个类别目录
            category_path = os.path.join(path, c)
            if not os.path.isdir(category_path):
                continue  # 跳过非目录项
            # 递归遍历所有子目录
            for root, _, files in os.walk(category_path):
                for file in files:# 遍历每个文件
                    file_path = os.path.join(root, file)# # 构建完整文件路径
                    data.append(file_path)
                    # 标签处理：测试集使用文件名，其他用映射表
                    if self.prefix == "test":
                        labels.append(file)  # 测试集标签直接使用文件名
                    else:
                        labels.append(self.label_map[c])  # 通过映射表转换标签
        return data, labels

    def get_datadef(self, path):
        data = []  # 存储文件路径
        labels = []  # 存储对应标签
        data.append(path)
        file = os.path.basename(path)
        split_name = os.path.splitext(file)
        # 标签处理：测试集使用文件名，其他用映射表
        labels.append(split_name[0])  # 测试集标签直接使用文件名
        return data, labels

    def __getitem__(self, index):
        # 获取单个数据样本的核心方法
        partial_path = self.input_data[index]
        # 获取partial数据路径
        # 例子：'D:\\实验数据\\2024-11-14\\果实数据增强\\train\\partial\\延安\\10\\2416118 - label - Cloud_label_1.txt'
        # 加载partial数据
        partial = np.loadtxt(partial_path, delimiter=',').astype(np.float32)  # 读取数据并转换为float32
        # 通过np.round将数据四舍五入至小数点后3位
        partial = np.round(partial, decimals=4)
        # 训练时的下采样处理
        if self.prefix == 'train' and self.sample:
            # 随机选择args.num_points个点
            choice = np.random.permutation((partial.shape[0]))
            partial = partial[choice[:self.samplepoints]]
            # 不足点数时补零（可能存在问题，见注释）
            if partial.shape[0] < self.samplepoints:
                zeros = np.zeros((self.samplepoints - partial.shape[0], 3))
                partial = np.concatenate([partial, zeros])
        # partial_xyz = partial[:,0:3]
        partial_rgb = partial[:, 3:6]
        partial_xyz = self.pc_normalize(partial[:,0:3])
        # 非测试集处理流程（只对val和train作处理
        if self.prefix not in ["test","def"]:
            complete_path = partial_path.replace('partial', 'gt')  # 构造gt路径
            complete = np.loadtxt(complete_path, delimiter=',').astype(np.float32)  # 读取数据并转换为float32
            # complete_xyz = complete[:, 0:3]
            complete_rgb = complete[:, 3:6]
            complete_xyz = self.pc_normalize(complete[:, 0:3])
            # new_complete = np.hstack((complete_xyz, complete_rgb))
            # new_partial = np.hstack((partial_xyz,partial_rgb))
            # 转换为Tensor
            complete = torch.from_numpy(np.hstack((complete_xyz, complete_rgb))).float()
            label = int(self.labels[index])  # 转换标签为整数
            partial = torch.from_numpy(np.hstack((partial_xyz,partial_rgb))).float()
            return label, partial, complete
        # 测试集处理流程
        else:
            # 随机选择1024个点
            # partial=farthest_point_sampling_with_density(partial,self.samplepoints)
            choice = np.random.permutation((partial.shape[0]))
            partial = partial[choice[:self.samplepoints]]
            partial_rgb = partial[:, 3:6]
            partial_xyz = self.pc_normalize(partial[:, 0:3])
            partial = torch.from_numpy(np.hstack((partial_xyz,partial_rgb))).float()
            label = self.labels[index]  # 测试集标签保持原始值
            return label, partial, partial  # 返回两次partial作为占位符

# 定义 PointAttN 模型中用于处理 PCN 数据集的 Dataset 类
class PCN_pcd(data.Dataset):
    def __init__(self, path, prefix="train"):
        """
        初始化 PCN_pcd 数据集
        参数:
            path (str): 数据集的根目录路径
            prefix (str): 数据集的子目录前缀，取值为 'train'、'val' 或 'test'
        """
        # 根据前缀选择对应的数据子目录
        if prefix == "train":
            self.file_path = os.path.join(path, 'train')
        elif prefix == "val":
            self.file_path = os.path.join(path, 'val')
        elif prefix == "test":
            self.file_path = os.path.join(path, 'test')
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")
        self.prefix = prefix
        # 定义类别标签映射，从类别名称到数字编码
        self.label_map = {
            '02691156': '0', '02933112': '1', '02958343': '2',
            '03001627': '3', '03636649': '4', '04256520': '5',
            '04379243': '6', '04530566': '7', 'all': '8'
        }
        # 定义逆向类别标签映射，从数字编码到类别名称
        self.label_map_inverse = {
            '0': '02691156', '1': '02933112', '2': '02958343',
            '3': '03001627', '4': '03636649', '5': '04256520',
            '6': '04379243', '7': '04530566', '8': 'all'
        }
        # 获取输入数据和对应标签
        self.input_data, self.labels = self.get_data(os.path.join(self.file_path, 'partial'))
        # 随机打乱数据顺序
        random.shuffle(self.input_data)
        # 数据集长度
        self.len = len(self.input_data)
        # 数据增强参数初始化
        self.scale = 0  # 是否进行缩放
        self.mirror = 1  # 是否进行镜像
        self.rot = 0  # 是否进行旋转
        self.sample = 1  # 是否进行采样

    def __len__(self):
        """
        返回数据集的长度
        """
        return self.len

    def read_pcd(self, path):
        """
        读取 PCD 文件并返回点云坐标

        参数:
            path (str): PCD 文件的路径

        返回:
            points (np.ndarray): 点云坐标数组，形状为 (N, 3)
        """
        pcd = o3d.io.read_point_cloud(path)
        points = np.asarray(pcd.points)
        return points

    def get_data(self, path):
        """
        获取数据集中的所有点云文件路径及其对应的标签

        参数:
            path (str): 点云文件所在的目录路径

        返回:
            data (list): 包含所有点云文件路径的列表，每个元素是一个对象的所有视图路径列表
            labels (list): 每个对象的类别标签列表
        """
        cls = os.listdir(path)  # 获取类别目录列表
        data = []
        labels = []
        for c in cls:
            objs = os.listdir(os.path.join(path, c))  # 获取每个类别下的对象目录
            for obj in objs:
                f_names = os.listdir(os.path.join(path, c, obj))  # 获取每个对象下的文件名
                obj_list = []
                for f_name in f_names:
                    data_path = os.path.join(path, c, obj, f_name)
                    obj_list.append(data_path)  # 添加每个点云文件的路径到对象列表
                data.append(obj_list)  # 将对象的所有点云文件路径添加到数据列表
                labels.append(self.label_map[c])  # 将对象的类别标签添加到标签列表
        return data, labels

    def randomsample(self, ptcloud, n_points):
        """
        随机采样点云中的点，返回指定数量的点

        参数:
            ptcloud (np.ndarray): 输入点云，形状为 (N, 3)
            n_points (int): 需要采样的点数

        返回:
            ptcloud (np.ndarray): 采样后的点云，形状为 (n_points, 3)
        """
        choice = np.random.permutation(ptcloud.shape[0])  # 随机打乱点的顺序
        ptcloud = ptcloud[choice[:n_points]]  # 选择前 n_points 个点
        # 如果点数不足 n_points，则用零填充
        if ptcloud.shape[0] < n_points:
            zeros = np.zeros((n_points - ptcloud.shape[0], 3))
            ptcloud = np.concatenate([ptcloud, zeros])
        return ptcloud

    def upsample(self, ptcloud, n_points):
        """
        对点云进行上采样，返回指定数量的点

        参数:
            ptcloud (np.ndarray): 输入点云，形状为 (N, 3)
            n_points (int): 需要的点数

        返回:
            ptcloud (np.ndarray): 上采样后的点云，形状为 (n_points, 3)
        """
        curr = ptcloud.shape[0]
        need = n_points - curr
        # 如果当前点数大于需要的点数，进行随机采样
        if need < 0:
            return ptcloud[np.random.permutation(n_points)]
        # 不断重复点云，直到满足需要的点数
        while curr <= need:
            ptcloud = np.tile(ptcloud, (2, 1))
            need -= curr
            curr *= 2
        # 随机选择剩余需要的点数
        choice = np.random.permutation(need)
        ptcloud = np.concatenate((ptcloud, ptcloud[choice]))
        return ptcloud

    def get_transform(self, points):
        """
        对点云进行数据增强，包括镜像、缩放和旋转

        参数:
            points (list): 包含两个点云数组的列表，通常为 [partial, complete]

        返回:
            result[0] (np.ndarray): 变换后的第一个点云
            result[1] (np.ndarray): 变换后的第二个点云
        """
        result = []
        rnd_value = np.random.uniform(0, 1)  # 生成一个随机值用于决定是否进行镜像
        # 数据增强仅在训练集上进行
        if self.mirror and self.prefix == 'train':
            # 初始化变换矩阵为单位矩阵
            trfm_mat = transforms3d.zooms.zfdir2mat(1)
            # 定义沿 x 轴和 z 轴的镜像变换矩阵
            trfm_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
            trfm_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)
            # 根据随机值决定是否进行镜像
            if rnd_value <= 0.25:
                trfm_mat = np.dot(trfm_mat_x, trfm_mat)
                trfm_mat = np.dot(trfm_mat_z, trfm_mat)
            elif rnd_value > 0.25 and rnd_value <= 0.5:
                trfm_mat = np.dot(trfm_mat_x, trfm_mat)
            elif rnd_value > 0.5 and rnd_value <= 0.75:
                trfm_mat = np.dot(trfm_mat_z, trfm_mat)
        # 对每个点云应用变换矩阵
        for ptcloud in points:
            ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)
            # 如果设置了缩放参数，则对点云进行缩放
            if self.scale:
                ptcloud = ptcloud * self.scale
            result.append(ptcloud)
        return result[0], result[1]

    def __getitem__(self, index):
        """
        获取指定索引的数据样本

        参数:
            index (int): 数据索引

        返回:
            tuple: (label, partial, complete) 或 (label, partial, complete, obj)
                   其中，obj 仅在测试集时返回
        """
        partial_path = self.input_data[index]  # 获取部分点云的文件路径列表
        n_sample = len(partial_path)
        idx = random.randint(0, n_sample - 1)  # 随机选择一个视图
        partial_path = partial_path[idx]  # 选择具体的部分点云路径
        partial = self.read_pcd(partial_path)  # 读取部分点云
        # 如果是训练集并且设置了采样，则对部分点云进行上采样到2048个点
        partial = self.upsample(partial, 2048)
        # 构建对应的完整点云路径，假设文件名格式一致
        gt_path = partial_path.replace('/' + partial_path.split('/')[-1], '.pcd')
        gt_path = gt_path.replace('partial', 'complete')
        if self.prefix == 'train':
            complete = self.read_pcd(gt_path)  # 读取完整点云
            partial, complete = self.get_transform([partial, complete])  # 对点云进行数据增强
        else:
            complete = self.read_pcd(gt_path)  # 读取完整点云（测试集不进行数据增强）
        # 将 numpy 数组转换为 torch 张量
        complete = torch.from_numpy(complete)
        partial = torch.from_numpy(partial)
        label = partial_path.split('/')[-3]  # 获取类别标签
        label = self.label_map[label]
        obj = partial_path.split('/')[-2]  # 获取对象名称
        if self.prefix == 'test':
            return label, partial, complete, obj  # 测试集返回对象名称
        else:
            return label, partial, complete  # 训练集和验证集不返回对象名称

# 定义 PointAttN 模型中用于处理 C3D 数据集的 Dataset 类
class C3D_h5(data.Dataset):
    def __init__(self, path, prefix="train"):
        """
        初始化 C3D_h5 数据集

        参数:
            path (str): 数据集的根目录路径
            prefix (str): 数据集的子目录前缀，取值为 'train'、'val' 或 'test'
        """
        # 根据前缀选择对应的数据子目录
        if prefix == "train":
            self.file_path = os.path.join(path, 'train')
        elif prefix == "val":
            self.file_path = os.path.join(path, 'val')
        elif prefix == "test":
            self.file_path = os.path.join(path, 'test')
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")
        self.prefix = prefix
        # 定义类别标签映射，从类别名称到数字编码
        self.label_map = {
            '02691156': '0', '02933112': '1', '02958343': '2',
            '03001627': '3', '03636649': '4', '04256520': '5',
            '04379243': '6', '04530566': '7', 'all': '8'
        }
        if prefix != "test":
            # 获取部分点云数据和标签
            self.input_data, self.labels = self.get_data(os.path.join(self.file_path, 'partial'))
            # 获取完整点云数据和标签
            self.gt_data, _ = self.get_data(os.path.join(self.file_path, 'gt'))
            print(len(self.gt_data), len(self.labels))
        else:
            # 仅获取部分点云数据和标签（测试集）
            self.input_data, self.labels = self.get_data(os.path.join(self.file_path, 'partial'))
        print(len(self.input_data))
        # 数据集长度
        self.len = len(self.input_data)
        # 数据增强参数初始化
        self.scale = 1  # 是否进行缩放
        self.mirror = 1  # 是否进行镜像
        self.rot = 0  # 是否进行旋转
        self.sample = 1  # 是否进行采样

    def __len__(self):
        """
        返回数据集的长度
        """
        return self.len

    def pc_normalize(self,pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def get_data(self, path):
        """
        获取数据集中的所有点云文件路径及其对应的标签

        参数:
            path (str): 点云文件所在的目录路径

        返回:
            data (list): 包含所有点云文件路径的列表
            labels (list): 每个点云的类别标签列表
        """
        cls = os.listdir(path)  # 获取类别目录列表
        data = []
        labels = []
        for c in cls:
            objs = os.listdir(os.path.join(path, c))  # 获取每个类别下的对象目录
            for obj in objs:
                data.append(os.path.join(path, c, obj))  # 添加每个点云文件的路径到数据列表
                if self.prefix == "test":
                    labels.append(obj)  # 测试集标签为对象名称
                else:
                    labels.append(self.label_map[c])  # 训练集和验证集标签为类别编码
        return data, labels

    def get_transform(self, points):
        """
        对点云进行数据增强，包括镜像、缩放和旋转

        参数:
            points (list): 包含两个点云数组的列表，通常为 [partial, complete]

        返回:
            result[0] (np.ndarray): 变换后的第一个点云
            result[1] (np.ndarray): 变换后的第二个点云
        """
        result = []
        rnd_value = np.random.uniform(0, 1)  # 生成一个随机值用于决定是否进行镜像
        angle = random.uniform(0, 2 * math.pi)  # 随机生成旋转角度
        scale = np.random.uniform(1 / 1.6, 1)  # 随机生成缩放因子
        # 初始化变换矩阵为单位矩阵
        trfm_mat = transforms3d.zooms.zfdir2mat(1)
        if self.mirror and self.prefix == 'train':
            # 定义沿 x 轴和 z 轴的镜像变换矩阵
            trfm_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
            trfm_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)
            # 根据随机值决定是否进行镜像
            if rnd_value <= 0.25:
                trfm_mat = np.dot(trfm_mat_x, trfm_mat)
                trfm_mat = np.dot(trfm_mat_z, trfm_mat)
            elif rnd_value > 0.25 and rnd_value <= 0.5:
                trfm_mat = np.dot(trfm_mat_x, trfm_mat)
            elif rnd_value > 0.5 and rnd_value <= 0.75:
                trfm_mat = np.dot(trfm_mat_z, trfm_mat)
        # 如果设置了旋转，则应用旋转变换
        if self.rot:
            trfm_mat = np.dot(transforms3d.axangles.axangle2mat([0, 1, 0], angle), trfm_mat)
        # 对每个点云应用变换矩阵和缩放
        for ptcloud in points:
            ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)
            if self.scale:
                ptcloud = ptcloud * scale
            result.append(ptcloud)
        return result[0], result[1]

    def __getitem__(self, index):
        """
        获取指定索引的数据样本

        参数:
            index (int): 数据索引

        返回:
            tuple: (label, partial, complete) 或 (label, partial, partial)
                   其中，后者用于测试集
        """
        partial_path = self.input_data[index]  # 获取部分点云的文件路径
        with h5py.File(partial_path, 'r') as f:
            partial = np.array(f['data'])  # 读取部分点云数据
            partial = self.pc_normalize(partial)
        # 如果是训练集并且设置了采样，则对部分点云进行采样到2048个点
        if self.prefix == 'train' and self.sample:
            choice = np.random.permutation(partial.shape[0])
            partial = partial[choice[:2048]]
            # 如果点数不足2048，则用零填充
            if partial.shape[0] < 2048:
                zeros = np.zeros((2048 - partial.shape[0], 3))
                partial = np.concatenate([partial, zeros])
        if self.prefix not in ["test"]:
            # 获取对应的完整点云路径
            complete_path = partial_path.replace('partial', 'gt')
            with h5py.File(complete_path, 'r') as f:
                complete = np.array(f['data'])  # 读取完整点云数据
                complete = self.pc_normalize(complete)
            # # 对部分点云和完整点云进行数据增强
            # partial, complete = self.get_transform([partial, complete])
            # 将 numpy 数组转换为 torch 张量
            complete = torch.from_numpy(complete)
            partial = torch.from_numpy(partial)
            label = self.labels[index]  # 获取类别标签
            return label, partial, complete
        else:
            # 对测试集，仅返回部分点云及其标签
            partial = torch.from_numpy(partial)
            label = self.labels[index]
            return label, partial, partial  # 测试集完整点云未知，使用部分点云代替
# 主程序，用于测试 C3D_h5 数据集的加载
if __name__ == '__main__':
    # 实例化 C3D_h5 数据集，设置为测试模式
    dataset = C3D_h5(prefix='test')
    # 创建 DataLoader，用于批量加载数据
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=True, num_workers=0)
    for idx, data in enumerate(dataloader, 0):
        print(data.shape)  # 打印数据形状
