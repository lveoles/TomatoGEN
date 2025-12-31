import torch.optim as optim
import torch
import logging
import math
import importlib
import datetime
import random
import munch
import yaml
from tqdm import tqdm
import os
import sys
import argparse
from dataset import *  # 导入自定义数据集类
from torch.cuda.amp import autocast, GradScaler
from thop import profile
import torch.nn.functional as F
from utils.mm3d_pn2 import furthest_point_sample, gather_points,grouping_operation,ball_query
from utils.model_utils import *

class EarlyStopping:
    """
    EarlyStopping 用于实现早停法。
    """
    def __init__(self, patience=10, verbose=True, delta=0, path='checkpoint.pth'):
        """
        参数:
        patience: 等待多少个epoch验证损失没有改善后停止训练
        verbose: 是否打印日志
        delta: 增加的损失阈值，只有损失改善超过该值时才认为是改善
        path: 保存最佳模型的路径
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.best_loss = None
        self.counter = 0  # 用于记录多少个epoch没有改善
        self.early_stop = False

    def __call__(self, val_loss, model):
        """
        根据验证损失判断是否需要早停
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        """保存当前最佳模型"""
        if self.verbose:
            print(f"Validation loss improved, saving model to {self.path}")
        torch.save(model.state_dict(), self.path)

def setup_dataloader(args):
    """
    根据配置加载数据集并返回数据加载器。
    """
    print(args.dataset)
    print(args.is_rgb)
    if args.dataset == 'pcn':
        dataset = PCN_pcd(args.pcnpath, prefix="train")
        dataset_test = PCN_pcd(args.pcnpath, prefix="test")
    elif args.dataset == 'c3d':
        dataset = C3D_h5(args.c3dpath, prefix="train")
        dataset_test = C3D_h5(args.c3dpath, prefix="val")
    elif args.dataset == 'mydata' and args.is_rgb == False:
        dataset = mydataloader(args.mydatapath, args.num_points_input, prefix="train")
        dataset_test = mydataloader(args.mydatapath, args.num_points_input, prefix="test")
    elif args.dataset == 'mydata' and args.is_rgb == True:
        dataset = mydataloaderrgb(args.mydatapath, args.num_points_input, prefix="train")
        dataset_test = mydataloaderrgb(args.mydatapath, args.num_points_input, prefix="test")
    else:
        raise ValueError('Dataset does not exist')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=int(args.workers),pin_memory=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=int(args.batch_size/2),
                                                  shuffle=False, num_workers=int(args.workers),pin_memory=True)
    return dataloader, dataloader_test

def setup_model(args):
    """
    加载模型并返回。
    """
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = torch.nn.DataParallel(model_module.Model(args))
    net.cuda()
    if hasattr(model_module, 'weights_init'):
        net.module.apply(model_module.weights_init)
    return net

def setup_optimizer(args, net):
    """
    配置优化器并返回。
    """
    lr = args.lr
    optimizer = getattr(optim, args.optimizer)
    if args.optimizer == 'Adagrad':
        optimizer = optimizer(net.module.parameters(), lr=lr, initial_accumulator_value=args.initial_accum_val)
    else:
        betas = tuple(map(float, args.betas.split(',')))
        optimizer = optimizer(net.module.parameters(), lr=lr, weight_decay=args.weight_decay, betas=betas)
    return optimizer

def setup_optimizers_GEN(args, net):
    """
    为生成器和判别器分别配置优化器。
    """
    # 生成器参数（排除判别器）
    gen_params = list(net.module.encoder.parameters()) + \
                 list(net.module.refine.parameters()) + \
                 list(net.module.refine1.parameters())
    dis_params = list(net.module.discriminator.parameters())
    # 初始化优化器
    optimizer_G = optim.Adam(gen_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    optimizer_D = optim.Adam(dis_params, lr=args.lr_d, betas=(0.5, 0.999))
    return optimizer_G, optimizer_D

def setup_scheduler(args, optimizer):
    """
    配置学习率调度器。
    """
    if args.lr_scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_interval, gamma=args.lr_decay_rate)
    elif args.lr_scheduler == 'CosineAnnealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nepoch, eta_min=args.lr_clip)
    elif args.lr_scheduler == 'OneCycleLR':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr_max,
                                                  epochs=args.nepoch, steps_per_epoch=args.steps_per_epoch,
                                                  pct_start=0.3, anneal_strategy='linear')
    elif args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         factor=0.5, patience=5,
                                                         verbose=True, min_lr=1e-6)
    elif args.lr_scheduler == 'LinelrCosineAnnealing':
        # 使用 Linelr + CosineAnnealing 的组合策略
        def lr_lambda(epoch):
            # 线性衰减学习率，前半部分线性衰减，后半部分使用余弦退火
            if epoch < 20:
                # 线性衰减
                return 1 - (epoch / (args.nepoch // 2))
            else:
                # 后半部分使用余弦退火
                return 0.5 * (1 + torch.cos(torch.tensor(epoch - args.nepoch // 2) * 3.14159 / (args.nepoch // 2)))
        # 组合的调度器，先使用线性衰减，后使用余弦退火
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif args.lr_scheduler == 'CosineAnnealingWarmRestarts':
        # CosineAnnealingWarmRestarts 是余弦退火与重启的组合调度器
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # 第一次重启的周期
            T_mult=0.7,  # 重启间隔的增长因子
            eta_min=args.lr_clip  # 最小学习率
        )
    else:
        raise ValueError("Unsupported learning rate scheduler: {}".format(args.lr_scheduler))
    return scheduler

def setup_schedulers_GEN(args, optimizer_G, optimizer_D):
    """
    为生成器和判别器分别配置调度器。
    """
    # 生成器采用余弦退火重启
    scheduler_G = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_G, T_0=10, T_mult=2, eta_min=args.lr_clip
    )
    # 判别器采用性能感知调度器
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_D, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
    )
    return scheduler_G, scheduler_D

def count_parameters(model):
    """
    计算并返回模型的参数总数。
    """
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

# def train_one_epoch(epoch, args, net, dataloader, optimizer, train_loss_meter, lr, accumulation_steps=8):
#     """
#     训练一个epoch。
#     """
#     net.module.train()
#     train_loss_meter.reset()
#     for i, data in enumerate(tqdm(dataloader,desc=f"Epoch {epoch}/{args.nepoch}",ncols=100), 0):
#         optimizer.zero_grad()
#         _, inputs, gt = data
#         inputs = inputs.float().cuda()
#         gt = gt.float().cuda()
#         inputs = inputs.transpose(2, 1).contiguous()#数据转秩
#         # 使用梯度累积
#         out2, loss2, net_loss = net(inputs, gt)  # 前向传播，计算损失
#         train_loss_meter.update(net_loss.mean().item())  # 更新损失记录
#         # 累积梯度
#         net_loss.backward()
#         # 梯度累积策略：每 accumulate_steps 步后更新一次参数
#         if (i + 1) % accumulation_steps == 0:
#             optimizer.step()  # 更新参数
#             optimizer.zero_grad()  # 清空梯度
#     torch.cuda.empty_cache()
#     log_msg = (f'{exp_name} train epoch[{epoch}] '
#                f'loss_type: {args.loss}, '
#                f'fine_loss: {loss2.mean().item():f} '
#                f'total_loss: {net_loss.mean().item():f} '
#                f'lr: {lr:f}')
#     logging.info(log_msg)

def train(args):
    """
    主训练函数，负责模型的训练过程。
    """
    print(str(args))
    logging.info(str(args))
    metrics = ['cd_p', 'cd_t', 'cd_t_coarse', 'cd_p_coarse']
    best_epoch_losses = {m: (0, 0) if m == 'f1' else (0, math.inf) for m in metrics}
    train_loss_meter = AverageValueMeter()
    val_loss_meters = {m: AverageValueMeter() for m in metrics}
    dataloader, dataloader_test = setup_dataloader(args)
    set_random_seed(args)
    net = setup_model(args)
    # optimizer = setup_optimizer(args, net)
    optimizer_G, optimizer_D = setup_optimizers_GEN(args, net)
    lr = args.lr
    # **添加学习率调度器**
    # scheduler = setup_scheduler(args, optimizer)
    scheduler_G, scheduler_D = setup_schedulers_GEN(args, optimizer_G, optimizer_D)
    scaler_G = GradScaler()
    scaler_D = GradScaler()
    # 获取模型的总参数量和FLOPs
    total_params = count_parameters(net)
    logging.info(f'Total model parameters: {total_params}')
    # 初始化早停
    early_stopping = EarlyStopping(patience=20, verbose=True, path=os.path.join(log_dir, 'best_model.pth'))
    for epoch in range(args.start_epoch, args.nepoch):
        train_one_epoch(epoch, args, net, dataloader, optimizer_G, optimizer_D, train_loss_meter, lr,scheduler_D,scheduler_G,scaler_G,scaler_D)
        if epoch % args.epoch_interval_to_save == 0:
            save_model(f'{log_dir}/network.pth', net)
            print("Saving net...")
            logging.info("Saving net...")
        if epoch % args.epoch_interval_to_val == 0 or epoch == args.nepoch - 1:
            val_loss=val(net, epoch, val_loss_meters, dataloader_test, best_epoch_losses)
        # if args.lr_decay:
        #     lr = update_lr(epoch, args, lr, optimizer)
        # **更新学习率**
            # 进行早停判断
            early_stopping(val_loss, net)
            # 如果达到早停条件，退出训练
            if early_stopping.early_stop:
                print("Early stopping triggered")
                logging.info("Early stopping triggered")
                break
        #     scheduler.step()

# def train_one_epoch(epoch, args, net, dataloader, optimizer_G, optimizer_D, train_loss_meter, lr, scheduler_D,scheduler_G, accumulation_steps=8):
#     net.module.train()
#     train_loss_meter.reset()
#     for i, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}/{args.nepoch}", ncols=100), 0):
#         optimizer_G.zero_grad()
#         optimizer_D.zero_grad()
#         _, inputs, gt = data
#         inputs = inputs.float().cuda()
#         gt = gt.float().cuda()
#         inputs = inputs.transpose(2, 1).contiguous()
#         # Forward + loss
#         result,gt2, coarse, total_train_loss = net(inputs, gt, is_training=True)
#         # ========== 判别器训练 ==========
#         real_label = torch.ones(coarse.size(0), 1).to(coarse.device)
#         fake_label = torch.zeros(coarse.size(0), 1).to(coarse.device)
#         # before forward:
#         for p in net.module.discriminator.parameters():
#             p.requires_grad = False
#         optimizer_G.zero_grad()
#         total_train_loss.backward()
#         optimizer_G.step()
#         # re-enable D
#         for p in net.module.discriminator.parameters():
#             p.requires_grad = True
#         # now do D update on detached fake samples:
#         fake_detached = result.detach()
#         real_output = net.module.discriminator(gt2.transpose(1, 2).contiguous())
#         fake_output = net.module.discriminator(fake_detached.transpose(1, 2).contiguous())
#         d_loss = F.binary_cross_entropy(real_output, real_label) + \
#                  F.binary_cross_entropy(fake_output, fake_label)
#         optimizer_D.zero_grad()
#         d_loss.backward()
#         optimizer_D.step()
#         scheduler_D.step(d_loss.item())
#     torch.cuda.empty_cache()
#     logging.info(f"Epoch [{epoch}] Gen Loss: {total_train_loss.item():.6f}, Disc Loss: {d_loss.item():.6f}")
#     logging.info(
#         f"Epoch [{epoch}] | Gen Loss: {total_train_loss.item():.6f} | "
#         f"Disc Loss: {d_loss.item():.6f} | "
#         f"LR_G: {optimizer_G.param_groups[0]['lr']:.6e} | "
#         f"LR_D: {optimizer_D.param_groups[0]['lr']:.6e}"
#     )

def train_one_epoch(epoch, args, net, dataloader,
                    optimizer_G, optimizer_D, train_loss_meter, lr,
                    scheduler_D, scheduler_G,
                    scaler_G=GradScaler(), scaler_D=GradScaler(),
                    accumulation_steps=8):
    # 切换模型到训练模式（支持并行训练时访问 module）
    net.module.train()
    # 重置损失计量器，用于记录本 epoch 的平均损失
    train_loss_meter.reset()
    # 遍历数据加载器，i 为 batch 索引，data 为批次数据
    for i, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}/{args.nepoch}", ncols=100), 0):
        _, inputs, gt = data
        # 将输入和真实值转为 float 并搬到 GPU
        inputs = inputs.float().cuda()
        gt = gt.float().cuda()
        # 转置点云维度为 [B, C, N]
        inputs = inputs.transpose(2, 1).contiguous()
        # =================== 混合精度前向 ===================
        with autocast():
            # encoder 不支持半精度运算，手动关闭 autocast
            with autocast(enabled=False):
                # 编码器生成全局特征 feat_g 和粗糙输出 coarse
                feat_g, coarse = net.module.encoder(inputs.float())  # [B, 512, N/8]; [B, 3, N/8]
                # 将原始点云拼接上粗糙输出
                new_x = torch.cat([inputs, coarse], dim=2)
                # 从拼接后的特征中采样 1024 个点
                new_x = gather_points(
                    new_x,
                    furthest_point_sample(new_x.transpose(1, 2).contiguous(), 1024)
                )
            # refine/refine1 网络和点云重建部分启用混合精度
            fine, feat_fine = net.module.refine(None, new_x, feat_g)
            fine1, feat_fine1 = net.module.refine1(feat_fine, fine, feat_g)
            # 转置回 [B, N, C] 或 [B, C, N]，以适配后续操作
            coarse = coarse.transpose(1, 2).contiguous()   # [B, N, 3]
            fine = fine.transpose(1, 2).contiguous()       # [B, N, C]
            fine1 = fine1.transpose(1, 2).contiguous()     # [B, N, C]
            # 针对 fine1 大小，从 GT 点云中采样对应数量的真值点
            gt_fine = gather_points(
                gt.transpose(1, 2).contiguous(),
                furthest_point_sample(gt, fine1.shape[1])
            ).transpose(1, 2).contiguous()
            # 计算 fine1 到原始 GT 的 Chamfer 距离损失
            loss3, _ = calc_cd(fine1, gt)
            # 针对 fine 大小从 GT 中采样点，计算 Chamfer 距离
            gt_fine1 = gather_points(
                gt.transpose(1, 2).contiguous(),
                furthest_point_sample(gt, fine.shape[1])
            ).transpose(1, 2).contiguous()
            loss2, _ = calc_cd(fine, gt_fine1)
            # 针对 coarse 大小再采样，计算 Chamfer 距离
            gt_coarse = gather_points(
                gt_fine1.transpose(1, 2).contiguous(),
                furthest_point_sample(gt_fine1, coarse.shape[1])
            ).transpose(1, 2).contiguous()
            loss1, _ = calc_cd(coarse, gt_coarse)
            # 判别器对生成数据的输出
            fake_output = net.module.discriminator(fine1.transpose(1, 2).contiguous())
            # 构造真实标签 all-one，用于生成器损失
            real_label = torch.ones(fine1.size(0), 1).to(fine1.device)
            # 生成器的对抗损失，使用带 logits 的 BCE
            # g_loss = F.binary_cross_entropy(fake_output.view(-1, 1), real_label)
            g_loss = F.binary_cross_entropy_with_logits(fake_output.view(-1, 1), real_label)
            # 总损失由三层 Chamfer 损失 + 对抗损失 组成
            total_train_loss = 0.1*loss1.mean() + 0.5*loss2.mean() + loss3.mean()+1.2*g_loss
        # =============== Generator 反向传播 ===============
        # 冻结判别器参数，只更新生成器
        for p in net.module.discriminator.parameters():
            p.requires_grad = False
        optimizer_G.zero_grad()
        # 梯度缩放并反向
        scaler_G.scale(total_train_loss).backward()
        scaler_G.step(optimizer_G)
        scaler_G.update()
        scheduler_G.step()
        # =============== Discriminator 反向传播 ===============
        # 解冻判别器参数
        for p in net.module.discriminator.parameters():
            p.requires_grad = True
        # 准备真实与伪造标签
        real_label = torch.ones(fine1.size(0), 1).to(fine1.device)
        fake_label = torch.zeros(fine1.size(0), 1).to(fine1.device)
        with autocast():
            # 判别器对真实点云的输出
            real_output = net.module.discriminator(gt_fine.transpose(1, 2).contiguous())
            # 判别器对生成点云的输出（detach 避免梯度回传至生成器）
            fake_output = net.module.discriminator(fine1.detach().transpose(1, 2).contiguous())
            # 判别器损失为真实/伪造样本的 BCE 之和
            # d_loss = F.binary_cross_entropy(real_output, real_label) + \
            #          F.binary_cross_entropy(fake_output, fake_label)
            d_loss = F.binary_cross_entropy_with_logits(real_output, real_label) + \
                     F.binary_cross_entropy_with_logits(fake_output, fake_label)
        optimizer_D.zero_grad()
        scaler_D.scale(d_loss).backward()
        scaler_D.step(optimizer_D)
        scaler_D.update()
        # 更新判别器学习率（可基于当前 d_loss 调度）
        scheduler_D.step(d_loss.item())
        # =============== 记录与日志 ===============
        # 更新平均损失记录器
        train_loss_meter.update(total_train_loss.item())
    # 在每个 epoch 结束后打印日志：生成器损失、判别器损失及学习率
    logging.info(
        f"Epoch [{epoch}] | Gen Loss: {total_train_loss.item():.6f} | "
        f"Disc Loss: {d_loss.item():.6f} | "
        f"LR_G: {optimizer_G.param_groups[0]['lr']:.6e} | "
        f"LR_D: {optimizer_D.param_groups[0]['lr']:.6e}"
    )
    # 清理无用的 CUDA 缓存
    torch.cuda.empty_cache()

def set_random_seed(args):
    """
    设置随机种子。
    """
    if not args.manual_seed:
        seed = random.randint(1, 10000)
    else:
        seed = int(args.manual_seed)
    print('Random Seed:', seed)
    logging.info(f'Random Seed: {seed}')
    random.seed(seed)
    torch.manual_seed(seed)

def val(net, curr_epoch_num, val_loss_meters, dataloader_test, best_epoch_losses):
    """
    验证函数，评估模型在验证集上的表现。
    """
    print('Testing...')
    logging.info('Testing...')
    for v in val_loss_meters.values():
        v.reset()
    net.module.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader_test, desc=f"Validation Epoch {curr_epoch_num}", ncols=100)):
            label, inputs, gt = data
            inputs = inputs.float().cuda()
            gt = gt.float().cuda()
            inputs = inputs.transpose(2, 1).contiguous()
            result_dict = net(inputs, gt, is_training=False)
            for k, v in val_loss_meters.items():
                v.update(result_dict[k].mean().item())
        save_best_model(val_loss_meters, best_epoch_losses, curr_epoch_num, net)

def save_best_model(val_loss_meters, best_epoch_losses, curr_epoch_num, net):
    """
    保存最佳模型。
    """
    fmt = 'best_%s: %f [epoch %d]; '
    best_log = ''
    for loss_type, (curr_best_epoch, curr_best_loss) in best_epoch_losses.items():
        if (val_loss_meters[loss_type].avg < curr_best_loss and loss_type != 'f1') or \
                (val_loss_meters[loss_type].avg > curr_best_loss and loss_type == 'f1'):
            best_epoch_losses[loss_type] = (curr_epoch_num, val_loss_meters[loss_type].avg)
            save_model(f'{log_dir}/best_{loss_type}_network.pth', net)
            print(f'Best {loss_type} net saved!')
            logging.info(f'Best {loss_type} net saved!')
            best_log += fmt % (loss_type, best_epoch_losses[loss_type][1], best_epoch_losses[loss_type][0])
        else:
            best_log += fmt % (loss_type, curr_best_loss, curr_best_epoch)
    curr_log = ''
    for loss_type, meter in val_loss_meters.items():
        curr_log += f'curr_{loss_type}: {meter.avg:f}; '
    print(curr_log)
    logging.info(curr_log)
    print(best_log)
    logging.info(best_log)

def update_lr(epoch, args, lr, optimizer):
    """
    根据学习率衰减规则更新学习率。
    """
    if args.lr_decay:
        if args.lr_decay_interval:
            if epoch > 0 and epoch % args.lr_decay_interval == 0:
                lr *= args.lr_decay_rate
        elif args.lr_step_decay_epochs:
            decay_epoch_list = [int(ep.strip()) for ep in args.lr_step_decay_epochs.split(',')]
            decay_rate_list = [float(rt.strip()) for rt in args.lr_step_decay_rates.split(',')]
            if epoch in decay_epoch_list:
                lr *= decay_rate_list[decay_epoch_list.index(epoch)]
        if args.lr_clip:
            lr = max(lr, args.lr_clip)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr

if __name__ == "__main__":
    config_path = os.path.join('./cfgs/TomatoGAN-mydata.yaml')
    args = munch.munchify(yaml.safe_load(open(config_path)))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    time = datetime.datetime.now().isoformat()[:19]
    time = time.replace(":", "-")
    if args.load_model:
        exp_name = os.path.basename(os.path.dirname(args.load_model))
        log_dir = os.path.dirname(args.load_model)
    else:
        exp_name = args.model_name + '_' + args.loss + '_' + args.flag + '_' + args.dataset
        log_dir = os.path.join(args.work_dir, exp_name)
        log_dir = os.path.join(log_dir, time)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    print('save_path:', args.work_dir)
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(log_dir, 'train.log')),
                                  logging.StreamHandler(sys.stdout)],
                        format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
                        force=True)
    train(args)
