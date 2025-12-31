from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import math
import utils.model_utils
import utils.mm3d_pn2

class BoxQueryAndGroup(nn.Module):
    def __init__(self, radius, nsample, use_xyz=True, normalize_xyz=False):
        super().__init__()
        self.radius = radius
        self.nsample = nsample
        self.use_xyz = use_xyz
        self.normalize_xyz = normalize_xyz

    def forward(self, key_xyz, key_features, query_box):
        """
        key_xyz: [B, N, 3]
        key_features: [B, C, N]
        query_box: [B, nq, 6] -> (x, y, z, l, w, h)
        """
        B, N, _ = key_xyz.shape
        query_centers = query_box[:, :, :3]  # [B, nq, 3]
        box_sizes = query_box[:, :, 3:]  # [B, nq, 3]
        key_xyz_expand = key_xyz.unsqueeze(1)  # [B, 1, N, 3]
        query_centers_expand = query_centers.unsqueeze(2)  # [B, nq, 1, 3]
        offsets = torch.abs(key_xyz_expand - query_centers_expand)  # [B, nq, N, 3]
        # 构造 mask：判断是否在 box 范围内
        box_boundaries = box_sizes.unsqueeze(2) / 2  # [B, nq, 1, 3]
        in_box_mask = (offsets <= box_boundaries).all(dim=-1)  # [B, nq, N]
        # 按照掩码选出前 nsample 个点的索引（用 topk 近似）
        idx = in_box_mask.float() * torch.arange(N, device=key_xyz.device).float().view(1, 1, -1)
        idx_sorted = torch.argsort(idx, dim=-1)[:, :, :self.nsample]  # [B, nq, nsample]
        local_group_mask = torch.zeros_like(idx_sorted, dtype=torch.bool)
        local_group_mask[(idx_sorted == 0)] = True
        local_group_mask[:, :, 0] = False  # 确保至少一个不被 mask
        idx_sorted = idx_sorted.contiguous().int()
        xyz_trans = key_xyz.transpose(1, 2).contiguous()
        key_features = key_features.contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx_sorted)  # [B, 3, nq, nsample]
        grouped_xyz -= query_centers.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= (box_sizes.transpose(1, 2).unsqueeze(-1) + 1e-6)
        grouped_features = grouping_operation(key_features, idx_sorted)  # [B, C, nq, nsample]
        if self.use_xyz:
            new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
        else:
            new_features = grouped_features
        return grouped_xyz, new_features, local_group_mask

class LightSpatialTransformerLayer(nn.Module):
    def __init__(self, d_model=256, base_nsample=16, use_xyz=False):
        super().__init__()
        self.base_nsample = base_nsample
        self.use_xyz = use_xyz
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        in_channels = d_model + 3 if use_xyz else d_model
        self.attn_mlp = nn.Sequential(
            nn.Conv2d(in_channels * 2, d_model, 1),
            nn.GELU(),
            nn.Conv2d(d_model, 1, 1)
        )
        self.ffn = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, 1)
        )

    def forward(self, xyz, features):
        """
        xyz: [B, N, 3]
        features: [B, C, N]
        return: [B, C, N]
        """
        B, C, N = features.shape
        # 自动估计每个 query 的 box 尺寸
        with torch.no_grad():
            dist = torch.cdist(xyz, xyz, p=2)  # [B, N, N]
            kth_dist = dist.topk(k=8, dim=-1, largest=False)[0][:, :, -1]  # [B, N]
            box_sizes = kth_dist.unsqueeze(-1).repeat(1, 1, 3) * 2.0  # [B, N, 3]
            query_boxes = torch.cat([xyz, box_sizes], dim=-1)  # [B, N, 6]
        # 使用 BoxQueryAndGroup
        local_grouper = BoxQueryAndGroup(
            radius=1.0,  # ignored
            nsample=self.base_nsample,
            use_xyz=self.use_xyz
        )
        grouped_xyz, grouped_features, _ = local_grouper(xyz, features, query_boxes)
        pos_embed = self.pos_encoder(grouped_xyz.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        query_expand = grouped_features[:, :C] if self.use_xyz else grouped_features
        attn_input = torch.cat([query_expand, pos_embed], dim=1)
        attn_weight = torch.softmax(self.attn_mlp(attn_input), dim=-1)
        out = (query_expand * attn_weight).sum(dim=-1)
        out = out + self.ffn(out)
        return out

# 定义 模型中的 cross_transformer 类
class cross_transformer(nn.Module):
    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=2048, dropout=0.0):
        super().__init__()
        # 多头自注意力机制，输入和输出维度为 d_model_out
        self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        # 前馈网络的第一层线性变换
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        # Dropout层，用于防止过拟合
        self.dropout1 = nn.Dropout(dropout)
        # 前馈网络的第二层线性变换
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)
        # 层归一化，用于稳定训练过程
        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)
        # 额外的Dropout层
        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)
        # GELU激活函数
        self.activation1 = torch.nn.GELU()
        # 输入投影层，将输入特征维度从 d_model 转换为 d_model_out
        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)
    def with_pos_embed(self, tensor, pos):
        """如果有位置嵌入，则将其加到张量上"""
        return tensor if pos is None else tensor + pos
    def forward(self, src1, src2, if_act=False):
        """
        前向传播函数
        src1: 第一个输入特征
        src2: 第二个输入特征
        if_act: 是否激活（未使用）
        """
        # 对输入特征进行投影
        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)
        b, c, _ = src1.shape  # 获取批次大小和通道数
        # 调整输入形状以适应多头注意力层
        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)  # 形状变为 (序列长度, 批次, 通道)
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1)
        # 对输入进行层归一化
        src1 = self.norm13(src1)
        src2 = self.norm13(src2)
        # 通过多头注意力机制计算 src1 和 src2 之间的注意力
        src12 = self.multihead_attn1(query=src1, key=src2, value=src2)[0]
        # 将注意力输出加到 src1 上，并进行 Dropout
        src1 = src1 + self.dropout12(src12)
        # 再次进行层归一化
        src1 = self.norm12(src1)
        # 前馈网络：线性变换 -> 激活 -> Dropout -> 线性变换
        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        # 将前馈网络的输出加到 src1 上
        src1 = src1 + self.dropout13(src12)
        # 调整输出形状，返回 (批次, 通道, 序列长度)
        src1 = src1.permute(1, 2, 0)
        return src1

#--------------#
#定义模型中的 PCT_refine 类，采用深度可分离卷积的细化模块
class PCT_refine(nn.Module):
    def __init__(self, channel=128, ratio=1):
        super(PCT_refine, self).__init__()
        self.ratio = ratio
        # 深度可分离卷积替换 conv_1: 256→channel
        self.conv_1 = nn.Sequential(
            # depthwise
            nn.Conv1d(256, 256, kernel_size=1, groups=256),
            # pointwise
            nn.Conv1d(256, channel, kernel_size=1)
        )
        # 深度可分离卷积替换 conv_11: 512→256
        self.conv_11 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1, groups=512),
            nn.Conv1d(512, 256, kernel_size=1)
        )
        # 深度可分离卷积替换 conv_x: 3→64
        self.conv_x = nn.Sequential(
            nn.Conv1d(3, 3, kernel_size=1, groups=3),
            nn.Conv1d(3, 64, kernel_size=1)
        )
        # LST 和 cross_transformer 保持不变
        self.lst1 = LightSpatialTransformerLayer(d_model=512, base_nsample=16, use_xyz=False)
        self.lst2 = LightSpatialTransformerLayer(d_model=512, base_nsample=16, use_xyz=False)
        self.sa1 = cross_transformer(channel * 2, 512)
        self.sa2 = cross_transformer(512, 512)
        self.sa3 = cross_transformer(512, channel * ratio)
        self.relu = nn.GELU()

        # 深度可分离卷积替换 conv_out: 64→3
        self.conv_out = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1, groups=64),
            nn.Conv1d(64, 3, kernel_size=1)
        )

        self.channel = channel
        # 深度可分离卷积替换 conv_delta: (channel*2)→channel
        self.conv_delta = nn.Sequential(
            nn.Conv1d(channel * 2, channel * 2, kernel_size=1, groups=channel * 2),
            nn.Conv1d(channel * 2, channel, kernel_size=1)
        )
        # 深度可分离卷积替换 conv_ps: (channel*ratio)→(channel*ratio)
        self.conv_ps = nn.Sequential(
            nn.Conv1d(channel * ratio, channel * ratio, kernel_size=1, groups=channel * ratio),
            nn.Conv1d(channel * ratio, channel * ratio, kernel_size=1)
        )
        # 深度可分离卷积替换 conv_x1: 64→channel
        self.conv_x1 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1, groups=64),
            nn.Conv1d(64, channel, kernel_size=1)
        )
        # 深度可分离卷积替换 conv_out1: channel→64
        self.conv_out1 = nn.Sequential(
            nn.Conv1d(channel, channel, kernel_size=1, groups=channel),
            nn.Conv1d(channel, 64, kernel_size=1)
        )

    def forward(self, x, coarse, feat_g):
        batch_size, _, N = coarse.size()
        xyz = coarse.permute(0, 2, 1).contiguous()  # [B, N, 3]
        # 粗糙点云特征变换
        y = self.conv_x1(self.relu(self.conv_x(coarse)))               # B, C, N
        # 全局特征变换
        feat_g = self.conv_1(self.relu(self.conv_11(feat_g)))          # B, C, N
        # 拼接并逐层细化
        y0 = torch.cat([y, feat_g.repeat(1, 1, y.shape[-1])], dim=1)   # B, 2C, N
        y1 = self.sa1(y0, y0)                                          # B, 512, N
        y1 = self.lst1(xyz, y1)                                        # B, 512, N
        y2 = self.sa2(y1, y1)                                          # B, 512, N
        y2 = self.lst2(xyz, y2)                                        # B, 512, N
        y3 = self.sa3(y2, y2)                                          # B, C*ratio, N
        # 特征后处理与上采样
        y3 = self.conv_ps(y3).reshape(batch_size, -1, N * self.ratio) # B, C*ratio, N*ratio
        y_up = y.repeat(1, 1, self.ratio)                              # B, C, N*ratio
        y_cat = torch.cat([y3, y_up], dim=1)                           # B, 2C, N*ratio
        y4 = self.conv_delta(y_cat)                                    # B, C, N*ratio

        # 最终细化输出
        x = self.conv_out(self.relu(self.conv_out1(y4))) + coarse.repeat(1, 1, self.ratio)
        return x, y3

# 定义 模型中的 PCT_encoder 类
class PCT_encoder(nn.Module):
    def __init__(self,channel=64):
        super(PCT_encoder, self).__init__()
        self.channel = channel
        # 初始卷积层，将输入点云的3维坐标映射到64维
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        # 第二层卷积，将64维特征映射到指定的通道数
        self.conv2 = nn.Conv1d(64, channel, kernel_size=1)
        # 多个 cross_transformer 模块，用于不同层次的特征交互
        self.sa1 = cross_transformer(channel, channel)#[64,64]
        self.sa1_1 = cross_transformer(channel * 2, channel * 2)#[128,128]
        self.sa2 = cross_transformer(channel * 2, channel * 2)#[128,128]
        self.sa2_1 = cross_transformer(channel * 4, channel * 4)#[256,256]
        self.sa3 = cross_transformer(channel * 4, channel * 4)#[256,256]
        self.sa3_1 = cross_transformer(channel * 8, channel * 8)#[512,512]
        # GELU激活函数
        self.relu = nn.GELU()
        # 自注意力模块，用于生成种子特征
        self.sa0_d = cross_transformer(channel * 8, channel * 8)#[512,512]
        self.sa1_d = cross_transformer(channel * 8, channel * 8)#[512,512]
        self.sa2_d = cross_transformer(channel * 8, channel * 8)#[512,512]
        # 输出卷积层，将64维特征映射回3维
        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        # 额外的卷积层，用于特征变换
        self.conv_out1 = nn.Conv1d(channel * 4, 64, kernel_size=1)# 256,64
        # 反卷积层，用于上采样
        self.ps = nn.ConvTranspose1d(channel * 8, channel, 128, bias=True)# 512,64
        # 更多卷积层，用于特征融合
        self.ps_refuse = nn.Conv1d(channel, channel * 8, kernel_size=1)
        self.ps_adj = nn.Conv1d(channel * 8, channel * 8, kernel_size=1)

    def forward(self,points_ori):
        """
        前向传播函数
        points: 输入的点云数据，形状为 (B, 3, N)
        N=args.numpoints
        编码器前向传播（包含多级下采样）
        """
        idx = furthest_point_sample(points_ori.transpose(1, 2).contiguous(), 2048)
        # 最远点采样索引与dim_feedforward保持一致
        points = gather_points(points_ori, idx) #(B, 3, N)
        batch_size, _, N = points.size()
        # 初始卷积和激活，提取基础特征
        x = self.relu(self.conv1(points))  # B, 64, N
        x0 = self.conv2(x)  # B, channel, N
        # GDP（全局降采样池化）：使用最远点采样降采样点云
        idx_0 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 4)
        x_g0 = gather_points(x0, idx_0)  # 采样后的特征 (B, channel, N//4)
        points = gather_points(points, idx_0)  # 采样后的点云坐标
        # SFA（自特征聚合）：通过自注意力模块进行特征交互
        x1 = self.sa1(x_g0, x0).contiguous()
        x1 = torch.cat([x_g0, x1], dim=1)  # 拼接特征
        # SFA 进一步聚合特征
        x1 = self.sa1_1(x1, x1).contiguous()
        # 第二层 GDP，进一步降采样
        idx_1 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 8)
        x_g1 = gather_points(x1, idx_1)#(B, channel*2, N//8)
        points = gather_points(points, idx_1)
        x2 = self.sa2(x_g1, x1).contiguous()  # C*2, N/8
        x2 = torch.cat([x_g1, x2], dim=1)  # 拼接特征
        # SFA 进一步聚合特征
        x2 = self.sa2_1(x2, x2).contiguous()
        # 第三层 GDP，再次降采样
        idx_2 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 16)
        x_g2 = gather_points(x2, idx_2)
        # points = gather_points(points, idx_2)  # 此行被注释掉，可能不需要继续降采样
        # 通过自注意力模块进行特征交互
        x3 = self.sa3(x_g2, x2).contiguous()  # C*4, N/16
        x3 = torch.cat([x_g2, x3], dim=1)  # 拼接特征
        # SFA 进一步聚合特征
        x3 = self.sa3_1(x3, x3).contiguous()
        # 生成种子特征，通过最大池化获取全局特征
        x_g = F.adaptive_max_pool1d(x3, 1).view(batch_size, -1).unsqueeze(-1)  # B, C*8, 1
        # 通过反卷积和卷积层进一步处理全局特征 B, C*8, 1---->B, C*8, upsampled_length
        x = self.relu(self.ps_adj(x_g))  # B, C*8, 1
        x = self.relu(self.ps(x))  # B, channel, upsampled_length
        x = self.relu(self.ps_refuse(x))  # B, channel*8, upsampled_length
        # 通过自注意力模块细化特征
        x0_d = self.sa0_d(x, x) #B, C*8, upsampled_length
        x1_d = self.sa1_d(x0_d, x0_d) #B, C*8, upsampled_length
        x2_d = self.sa2_d(x1_d, x1_d).reshape(batch_size, self.channel * 4, N // 8) #B, C*4, N // 8  # 调整形状
        # 生成最终细化特征
        fine = self.conv_out(self.relu(self.conv_out1(x2_d)))  # B, 3, N/8
        return x_g, fine

class Discriminator(nn.Module):
    def __init__(self, input_dim=3, base_channels=32):
        super(Discriminator, self).__init__()
        # 基本卷积层
        self.conv1 = nn.Conv1d(input_dim, base_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(base_channels, base_channels * 2, kernel_size=1)
        self.conv3 = nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=1)
        self.conv4 = nn.Conv1d(base_channels * 4, base_channels * 8, kernel_size=1)
        # 注意力层
        self.attn = nn.MultiheadAttention(embed_dim=base_channels * 8, num_heads=2, dropout=0.1)
        # 最大池化层，将点云的所有点压缩为一个全局表示
        self.global_pool = nn.AdaptiveMaxPool1d(1)  # 用于池化每个点云到一个单一的全局特征
        # 最终分类层
        self.fc = nn.Linear(base_channels * 8, 1)

    def forward(self, x):
        """
        输入 x 形状为 (batch_size, channels, num_points)
        """
        x = F.relu(self.conv1(x), inplace=True)  # B, C, N
        x = F.relu(self.conv2(x), inplace=True)  # B, C*2, N
        x = F.relu(self.conv3(x), inplace=True)  # B, C*4, N
        x = F.relu(self.conv4(x), inplace=True)  # B, C*8, N
        # 使用自注意力机制来增强特征表示
        x = x.permute(2, 0, 1)  # 调整维度为 (N, B, C*8)，适应多头注意力机制
        attn_output, _ = self.attn(x, x, x)  # B, C*8, N
        x = attn_output.permute(1, 2, 0)  # 恢复为 (B, C*8, N)
        # 使用最大池化来提取全局特征
        x = self.global_pool(x)  # B, C*8, 1
        x = x.squeeze(-1)  # 压缩为 B, C*8
        # 分类层，输出一个概率
        x = self.fc(x)  # B, 1
        # 返回 Sigmoid 激活后的概率值，表示输入点云为真实的概率
        return torch.sigmoid(x)

# 定义 PointAttN 模型中的 Model 类
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        # 根据数据集选择上采样比例，决定了从corase到fine的点数变化 也就是放大倍数
        if args.dataset == 'pcn':
            step1 = 4
            step2 = 8
        elif args.dataset == 'c3d':
            step1 = 1
            step2 = 4
        elif args.dataset == 'mydata':
            step1 = 1
            step2 = 2
        else:
            raise ValueError('dataset does not exist')  # 修正为抛出异常
        # self.normal_mod= DataNormalization()
        # 实例化编码器和两个细化模块
        self.encoder = PCT_encoder()
        self.refine = PCT_refine(ratio=step1)#调整corase数据分布
        self.refine1 = PCT_refine(ratio=step2)
        # 实例化判别器
        self.discriminator = Discriminator()

    def forward(self, x_ori, gt_ori=None, is_training=True):
        """
        前向传播函数
        x: 输入点云，形状为 (B, 3, N)
        gt: 真实点云（用于计算损失）(B, 3, N)
        is_training: 是否处于训练模式
        """
        x=x_ori  #(B, 3, N)
        gt=gt_ori #(B, 3, N)
        # #输入批次数据标准化
        # x,centroid,max_distance=self.normal_mod(x_ori)
        # gt, _, _ = self.normal_mod(gt_ori)
        # 使用编码器提取全局特征和粗糙点云
        feat_g, coarse= self.encoder(x) # B, Channel*8, 1;B, 3, N/8
        # 将输入点云和粗糙点云拼接
        new_x = torch.cat([x, coarse], dim=2)  # B, 3+3, N
        # 通过最远点采样和特征聚合生成新的点云
        new_x = gather_points(new_x, furthest_point_sample(new_x.transpose(1, 2).contiguous(), 1024)) # B, 3, N
        # 通过两个细化模块逐步细化点云
        fine, feat_fine = self.refine(None, new_x, feat_g) # B, 3, N;;# B, 512, N
        fine1, feat_fine1 = self.refine1(feat_fine, fine, feat_g)
        # 调整粗糙点云和细化点云的形状
        coarse = coarse.transpose(1, 2).contiguous()  # B, N, C
        fine = fine.transpose(1, 2).contiguous() # B, C, N
        fine1 = fine1.transpose(1, 2).contiguous() # B, N, C
        if is_training:
            gt_fine = gather_points(gt.transpose(1, 2).contiguous(),
                                     furthest_point_sample(gt, fine1.shape[1])).transpose(1, 2).contiguous()#[B,N,3]
            # 计算最终细化点云与真实点云之间的Chamfer距离损失
            loss3, _ = calc_cd(fine1, gt)
            # 采样真实点云用于中间层损失
            gt_fine1 = gather_points(gt.transpose(1, 2).contiguous(),
                                     furthest_point_sample(gt, fine.shape[1])).transpose(1, 2).contiguous()#[B,N,3]
            # 计算中间细化点云与采样真实点云之间的损失
            loss2, _ = calc_cd(fine, gt_fine1)
            # 采样真实点云用于粗糙层损失
            gt_coarse = gather_points(gt_fine1.transpose(1, 2).contiguous(),
                                      furthest_point_sample(gt_fine1, coarse.shape[1])).transpose(1, 2).contiguous()#[B,N,3]
            # 计算粗糙点云与采样真实点云之间的损失
            loss1, _ = calc_cd(coarse, gt_coarse)#l1cd
            ###----------GAN对抗损失-------###
            # 判别器损失
            # real_label = torch.ones_like(coarse)  # 真实标签
            # fake_label = torch.zeros_like(coarse)  # 假标签
            real_label = torch.ones(coarse.size(0), 1).to(coarse.device)  # 真实标签 (1)
            # fake_label = torch.zeros(coarse.size(0), 1).to(coarse.device)  # 假标签 (0)
            # 判别器计算真实点云和生成点云的分类
            # real_output = self.discriminator(gt_fine.transpose(1, 2).contiguous())
            fake_output = self.discriminator(fine1.transpose(1, 2).contiguous())
            # real_output = real_output.view(-1, 1)  # 确保 real_output 是 [batch_size, 1]
            fake_output = fake_output.view(-1, 1)  # 确保 fake_output 是 [batch_size, 1]
            # # 计算对抗损失
            # d_loss_real = F.binary_cross_entropy(real_output, real_label)# binary_cross_entropy（target，input）
            # d_loss_fake = F.binary_cross_entropy(fake_output, fake_label)
            # d_loss = d_loss_real + d_loss_fake
            # 生成器损失
            g_loss = F.binary_cross_entropy(fake_output, real_label)
            total_train_loss = loss1.mean() + loss2.mean() + loss3.mean() + g_loss
            ###----------GAN对抗损失-------###

            ###----------原始损失-------###
            # 总训练损失为各层损失的平均和
            # total_train_loss = loss1.mean() + loss2.mean() + loss3.mean()
            ###----------原始损失-------###
            #最终输出 补全结果+判别器损失+生成器损失
            return fine1,gt_fine, coarse, total_train_loss
        else:
            # 在测试模式下，计算各层Chamfer距离
            cd_p, cd_t = calc_cd(fine1, gt)
            cd_p_coarse, cd_t_coarse = calc_cd(coarse, gt)
            # 返回不同层次的输出和损失
            return {
                'out1': coarse,
                'out2': fine1,
                'cd_t_coarse': cd_t_coarse,#粗糙与真值
                'cd_p_coarse': cd_p_coarse,
                'cd_p': cd_p,#精细与真值 L2cd
                'cd_t': cd_t###L2cd 平方
            }

