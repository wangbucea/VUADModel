import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
from tqdm import tqdm

# 分布式训练相关导入
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from vla_dataset_loader import create_vla_dataloader, VLADataset, collate_fn
from VLAModel import RobotUniADModel
from compute import HungarianMatcher, DETRLossComputer

class VLATrainer:
    def __init__(self, config, rank=0, world_size=1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1
        
        # 设置设备
        if self.is_distributed:
            self.device = torch.device(f'cuda:{rank}')
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        
        # 先创建保存目录（只在主进程中创建）
        if rank == 0:
            self.setup_directories()
        
        # 分布式训练时需要同步
        if self.is_distributed:
            dist.barrier()
        
        # 再设置日志
        self.setup_logging()
        
        # 初始化模型
        self.model = self.build_model()
        
        # 初始化数据加载器
        self.train_loader, self.val_loader = self.build_dataloaders()
        
        # 初始化优化器和调度器
        self.optimizer, self.scheduler = self.build_optimizer_and_scheduler()
        
        # 初始化TensorBoard（只在主进程中）
        if rank == 0:
            self.writer = SummaryWriter(log_dir=os.path.join(self.config['save_dir'], 'tensorboard'))
        else:
            self.writer = None
        
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def setup_logging(self):
        """设置日志记录"""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(os.path.join(self.config['save_dir'], 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """创建必要的目录"""
        os.makedirs(self.config['save_dir'], exist_ok=True)
        os.makedirs(os.path.join(self.config['save_dir'], 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.config['save_dir'], 'tensorboard'), exist_ok=True)
        
    def build_model(self):
        """构建模型"""
        model = RobotUniADModel(
            num_seg_classes=self.config['num_seg_classes'],
            num_det_classes=self.config['num_det_classes'],
            num_seg_queries=self.config['num_seg_queries'],
            num_det_queries=self.config['num_det_queries'],
            num_points=self.config['num_points'],
            action_dim=self.config['action_dim'],
            seq_len=self.config['seq_len'],
            feature_dim=self.config['feature_dim'],
            image_size=self.config['image_size'],
            use_dual_view=self.config['use_dual_view']
        )
        
        model = model.to(self.device)
        
        # 使用DistributedDataParallel包装模型
        if self.is_distributed:
            model = DDP(
                model, 
                device_ids=[self.rank], 
                output_device=self.rank, 
                find_unused_parameters=True,
                broadcast_buffers=False,  # 减少通信开销
                bucket_cap_mb=25,  # 减少bucket大小
                gradient_as_bucket_view=True  # 优化内存使用
            )
        
        # 打印模型参数数量（只在主进程中打印）
        if self.rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.logger.info(f"Total parameters: {total_params:,}")
            self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model
        
    def build_dataloaders(self):
        """构建数据加载器"""
        # 定义图像变换
        transform = transforms.Compose([
            # transforms.Resize((self.config['image_size'], self.config['image_size'])),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 创建数据集
        train_dataset = VLADataset(
            data_root=self.config['data_root'],
            transform=transform,
            stride=self.config.get('stride', 1)
        )
        
        # 创建采样器
        if self.is_distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
            shuffle = False  # 使用DistributedSampler时不能shuffle
        else:
            train_sampler = None
            shuffle = True
        
        # 创建训练数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            collate_fn=collate_fn  # 添加自定义collate函数
        )
        
        # 创建验证数据加载器（如果有验证数据）
        val_loader = None
        if 'val_data_root' in self.config and self.config['val_data_root']:
            val_dataset = VLADataset(
                data_root=self.config['val_data_root'],
                transform=transform,
                stride=self.config.get('stride', 1)
            )
            
            if self.is_distributed:
                val_sampler = DistributedSampler(
                    val_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False
                )
            else:
                val_sampler = None
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                sampler=val_sampler,
                num_workers=self.config['num_workers'],
                pin_memory=True,
                collate_fn=collate_fn  # 添加自定义collate函数
            )
        
        if self.rank == 0:
            self.logger.info(f"Train dataset size: {len(train_dataset)}")
            if val_loader:
                self.logger.info(f"Validation dataset size: {len(val_loader.dataset)}")
        
        return train_loader, val_loader
        
    def build_optimizer_and_scheduler(self):
        """构建优化器和学习率调度器，实现差异化学习率"""
        # 获取模型参数（处理DDP包装的情况）
        model_params = self.model.module if self.is_distributed else self.model
        
        # 分离DINOv2骨干网络参数和其他参数
        backbone_params = []
        other_params = []
        
        # 获取DINOv2参数
        if hasattr(model_params.visual_encoder, 'dinov2_view1'):
            # 双视角模型
            backbone_params.extend(list(model_params.visual_encoder.dinov2_view1.parameters()))
            backbone_params.extend(list(model_params.visual_encoder.dinov2_view2.parameters()))
        elif hasattr(model_params.visual_encoder, 'dinov2'):
            # 单视角模型
            backbone_params.extend(list(model_params.visual_encoder.dinov2.parameters()))
        
        # 获取其他所有参数
        backbone_param_ids = set(id(p) for p in backbone_params)
        for name, param in model_params.named_parameters():
            if id(param) not in backbone_param_ids:
                other_params.append(param)
        
        # 创建参数组，DINOv2使用较小学习率
        param_groups = [
            {
                'params': backbone_params,
                'lr': self.config['backbone_lr'],
                'name': 'backbone'
            },
            {
                'params': other_params,
                'lr': self.config['learning_rate'],
                'name': 'other'
            }
        ]
        
        # AdamW优化器
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Warmup + Cosine学习率调度器
        def lr_lambda(current_step):
            warmup_steps = self.config['warmup_epochs'] * len(self.train_loader)
            total_steps = self.config['num_epochs'] * len(self.train_loader)
            
            if current_step < warmup_steps:
                # Warmup阶段：线性增长
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine退火
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return 0.5 * (1.0 + np.cos(np.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        if self.rank == 0:
            self.logger.info(f"Backbone learning rate: {self.config['backbone_lr']}")
            self.logger.info(f"Other modules learning rate: {self.config['learning_rate']}")
            self.logger.info(f"Backbone parameters: {len(backbone_params)}")
            self.logger.info(f"Other parameters: {len(other_params)}")
        
        return optimizer, scheduler
        
    def prepare_targets(self, batch):
        """准备训练目标"""
        targets = {}
        
        # 语义分割目标
        if 'segmentation_mask' in batch:
            targets['seg_masks'] = batch['segmentation_mask'].to(self.device)
        
        # 目标检测目标 - 修改为处理列表格式的检测框
        det_targets = []
        # 获取批次大小
        batch_size = len(batch['detection_boxes'])
        
        for i in range(batch_size):
            # 过滤有效的检测框
            boxes = batch['detection_boxes'][i]
            labels = batch['detection_labels'][i]
            
            # 检查是否为空张量
            if boxes.numel() > 0:
                valid_mask = (boxes.sum(dim=1) > 0)
                target = {
                    'boxes': boxes[valid_mask].to(self.device),
                    'labels': labels[valid_mask].to(self.device)
                }
            else:
                # 处理空张量情况
                target = {
                    'boxes': boxes.to(self.device),
                    'labels': labels.to(self.device)
                }
            
            det_targets.append(target)
        
        targets['det_targets'] = det_targets
        
        # 关键帧目标
        if 'key_frame_labels' in batch:
            targets['point_coords'] = batch['key_frame_labels'].to(self.device)
        
        # 动作序列目标
        if 'master_actions' in batch:
            targets['actions'] = batch['master_actions'].to(self.device)
        
        return targets
        
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        epoch_losses = []
        epoch_total = 0.0
        epoch_seg = 0.0
        epoch_det_bbox = 0.0
        epoch_det_ce = 0.0
        epoch_point = 0.0
        epoch_action = 0.0
        
        # 分布式训练时设置sampler的epoch
        if self.is_distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.current_epoch)
            
        # 同步所有进程，确保epoch开始时所有进程都准备好
        if self.is_distributed:
            dist.barrier()
        
        # 只在主进程显示进度条
        if self.rank == 0:
            progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        else:
            progress_bar = self.train_loader
            
        for batch_idx, batch in enumerate(progress_bar):
            # 准备输入数据
            if self.config['use_dual_view']:
                images = [batch['chest_images'].to(self.device), batch['head_images'].to(self.device)]
            else:
                images = batch['chest_images'].to(self.device)
            
            input_actions = batch['current_state'].to(self.device)
            targets = self.prepare_targets(batch)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            outputs = self.model(
                images=images,
                input_actions=input_actions,
                target_actions=targets.get('slave_actions')
            )
            
            # 计算损失
            losses = self.model.compute_loss(outputs, targets)
            total_loss = losses['total_loss']
            seg_loss = losses['seg_loss']
            det_box_loss = losses['det_bbox_loss']
            det_ce_loss = losses['det_ce_loss']
            point_loss = losses['point_loss']
            trajectory_loss = losses['trajectory_loss']
            epoch_total += total_loss
            epoch_seg += seg_loss
            epoch_det_bbox += det_box_loss
            epoch_det_ce += det_ce_loss
            epoch_point += point_loss
            epoch_action += trajectory_loss
            
            # 反向传播
            total_loss.backward()
            
            # 分布式训练时同步梯度
            if self.is_distributed:
                # 确保所有进程的梯度计算完成
                dist.barrier()
            
            # 梯度裁剪
            if self.config['grad_clip_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip_norm'])
            
            self.optimizer.step()
            self.scheduler.step()
            
            # 分布式训练时在每个batch后同步
            if self.is_distributed and batch_idx % 10 == 0:  # 每10个batch同步一次
                dist.barrier()
            
            # 记录损失
            epoch_losses.append(total_loss.item())
            
            # 更新进度条（只在主进程）
            if self.rank == 0:
                current_lr_backbone = self.optimizer.param_groups[0]['lr']
                current_lr_other = self.optimizer.param_groups[1]['lr']
                progress_bar.set_postfix({
                'Loss': f'{epoch_total.item():.4f}',
                'seg': f'{epoch_seg.item():.4f}',
                'box': f'{epoch_det_bbox.item():.4f}',
                'ce': f'{epoch_det_ce.item():.4f}',
                'point': f'{epoch_point.item():.4f}',
                'actions': f'{epoch_action.item():.4f}',
                'LR_backbone': f'{current_lr_backbone:.2e}',
                'LR_other': f'{current_lr_other:.2e}'
            })
                
                # 记录到TensorBoard
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Total_Loss', total_loss.item(), global_step)
                self.writer.add_scalar('Train/LR_Backbone', current_lr_backbone, global_step)
                self.writer.add_scalar('Train/LR_Other', current_lr_other, global_step)
                
                # 记录各个子损失
                for loss_name, loss_value in losses.items():
                    if loss_name != 'total_loss':
                        self.writer.add_scalar(f'Train/{loss_name}', loss_value.item(), global_step)
        
        # 分布式训练时确保所有进程完成epoch
        if self.is_distributed:
            dist.barrier()
            
        avg_loss = np.mean(epoch_losses)
        self.train_losses.append(avg_loss)
        
        # 分布式训练时同步损失值
        if self.is_distributed:
            # 将损失转换为tensor并同步
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()
        
        return avg_loss
        
    def validate_epoch(self):
        """验证一个epoch"""
        if self.val_loader is None:
            return None
            
        self.model.eval()
        epoch_losses = []
        
        # 分布式训练时设置sampler的epoch
        if self.is_distributed and hasattr(self.val_loader.sampler, 'set_epoch'):
            self.val_loader.sampler.set_epoch(self.current_epoch)
        
        with torch.no_grad():
            # 只在主进程显示进度条
            if self.rank == 0:
                progress_bar = tqdm(self.val_loader, desc='Validation')
            else:
                progress_bar = self.val_loader
                
            for batch in progress_bar:
                # 准备输入数据
                if self.config['use_dual_view']:
                    images = [batch['chest_images'].to(self.device), batch['head_images'].to(self.device)]
                else:
                    images = batch['chest_images'].to(self.device)
                
                input_actions = batch['current_state'].to(self.device)
                targets = self.prepare_targets(batch)
                
                # 前向传播
                outputs = self.model(
                    images=images,
                    input_actions=input_actions,
                    target_actions=targets.get('actions'),
                    det_targets=targets.get('det_targets')
                )
                
                # 计算损失
                losses = self.model.compute_loss(outputs, targets)
                total_loss = losses['total_loss']
                
                epoch_losses.append(total_loss.item())
        
        avg_loss = np.mean(epoch_losses)
        self.val_losses.append(avg_loss)
        
        # 记录到TensorBoard（只在主进程）
        if self.rank == 0 and self.writer is not None:
            self.writer.add_scalar('Val/Total_Loss', avg_loss, self.current_epoch)
        
        return avg_loss
        
    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        # 只在主进程中保存
        if self.rank != 0:
            return
            
        # 获取模型state_dict（处理DDP包装的情况）
        model_state_dict = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.config['save_dir'], 'checkpoints', 'latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳检查点
        # if is_best:
        #     best_path = os.path.join(self.config['save_dir'], 'checkpoints', 'best.pth')
        #     torch.save(checkpoint, best_path)
        #     self.logger.info(f"Saved best model with loss: {self.best_loss:.4f}")
        
        # 定期保存epoch检查点
        # if (self.current_epoch + 1) % self.config['save_freq'] == 0:
        #     epoch_path = os.path.join(self.config['save_dir'], 'checkpoints', f'epoch_{self.current_epoch+1}.pth')
        #     torch.save(checkpoint, epoch_path)
            
    def save_training_info(self):
        """保存训练过程信息"""
        training_info = {
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_loss': self.best_loss,
            'total_epochs': self.current_epoch + 1,
            'training_time': time.time() - self.start_time
        }
        
        info_path = os.path.join(self.config['save_dir'], 'training_info.json')
        with open(info_path, 'w') as f:
            json.dump(training_info, f, indent=2)
            
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 处理DDP模型的state_dict加载
            if self.is_distributed:
                # 如果当前是DDP模型，但检查点是单GPU模型
                if not any(key.startswith('module.') for key in checkpoint['model_state_dict'].keys()):
                    # 为检查点的键添加'module.'前缀
                    new_state_dict = {}
                    for key, value in checkpoint['model_state_dict'].items():
                        new_state_dict[f'module.{key}'] = value
                    checkpoint['model_state_dict'] = new_state_dict
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 如果当前是单GPU模型，但检查点是DDP模型
                if any(key.startswith('module.') for key in checkpoint['model_state_dict'].keys()):
                    # 移除检查点键的'module.'前缀
                    new_state_dict = {}
                    for key, value in checkpoint['model_state_dict'].items():
                        if key.startswith('module.'):
                            new_state_dict[key[7:]] = value  # 移除'module.'前缀
                        else:
                            new_state_dict[key] = value
                    checkpoint['model_state_dict'] = new_state_dict
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.current_epoch = checkpoint['epoch']
            self.best_loss = checkpoint['best_loss']
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            
            if self.rank == 0:
                self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
            return True
        return False
        
    def train(self):
        """主训练循环"""
        self.start_time = time.time()
        if self.rank == 0:
            self.logger.info("Starting training...")
        
        # 尝试加载之前的检查点
        checkpoint_path = os.path.join(self.config['save_dir'], 'checkpoints', 'latest.pth')
        if self.config['resume'] and self.load_checkpoint(checkpoint_path):
            if self.rank == 0:
                self.logger.info("Resumed training from checkpoint")
        
        for epoch in range(self.current_epoch, self.config['num_epochs']):
            self.current_epoch = epoch
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            # val_loss = self.validate_epoch()
            
            # 检查是否是最佳模型
            # if val_loss is not None:
            #     is_best = val_loss < self.best_loss
            #     if is_best:
            #         self.best_loss = val_loss
            # else:
            #     is_best = False
            
            # 保存检查点
            # self.save_checkpoint(is_best)
            
            # 保存训练信息
            if self.rank == 0:
                self.save_training_info()
            
            # 日志记录（只在主进程）
            if self.rank == 0:
                # val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
                self.logger.info(
                    f"Epoch {epoch+1}/{self.config['num_epochs']} - "
                    f"Train Loss: {train_loss:.4f}"
                    f"Best Loss: {self.best_loss:.4f}"
                )
            
            # 早停检查（可选）
            # if self.config.get('early_stopping_patience'):
            #     if len(self.val_losses) > self.config['early_stopping_patience']:
            #         recent_losses = self.val_losses[-self.config['early_stopping_patience']:]
            #         if all(loss >= self.best_loss for loss in recent_losses):
            #             self.logger.info("Early stopping triggered")
            #             break
        
        total_time = time.time() - self.start_time
        if self.rank == 0:
            self.logger.info(f"Training completed in {total_time/3600:.2f} hours")
            
            # 关闭TensorBoard writer
            if self.writer is not None:
                self.writer.close()

def setup_distributed(rank, world_size, backend='nccl'):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 设置NCCL超时和调试选项
    os.environ['TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC'] = '1800'  # 30分钟超时
    os.environ['TORCH_NCCL_ENABLE_MONITORING'] = '1'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_TIMEOUT'] = '1800'
    
    # 设置当前设备（在初始化进程组之前）
    torch.cuda.set_device(rank)
    
    # 初始化进程组，指定device_id避免GPU映射警告
    dist.init_process_group(
        backend=backend, 
        rank=rank, 
        world_size=world_size,
        device_id=torch.device(f'cuda:{rank}'),
        timeout=timedelta(seconds=1800)  # 30分钟超时
    )

def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def train_worker(rank, world_size, config):
    """分布式训练工作进程"""
    # 初始化分布式环境
    setup_distributed(rank, world_size)
    
    try:
        # 创建训练器并开始训练
        trainer = VLATrainer(config, rank=rank, world_size=world_size)
        trainer.train()
    finally:
        # 清理分布式环境
        cleanup_distributed()

def main():
    # 训练配置
    config = {
        # 数据配置
        'data_root': './',
        'batch_size': 3,
        'num_workers': 4,
        'image_size': (480, 640),
        
        # 模型配置
        'num_seg_classes': 2, 
        'num_det_classes': 3,  # bottle, brush, cube
        'num_seg_queries': 100,
        'num_det_queries': 100,
        'num_points': 5,
        'action_dim': 7,
        'seq_len': 30,
        'feature_dim': 256,
        'use_dual_view': True,
        
        # 训练配置
        'num_epochs': 50,
        'learning_rate': 1e-5,
        'backbone_lr': 1e-6,
        'weight_decay': 1e-5,
        'warmup_epochs': 5,
        'grad_clip_norm': 1.0,
        
        # 保存配置
        'save_dir': './experiments/' + datetime.now().strftime('%Y%m%d_%H%M%S'),
        'save_freq': 10,  # 每10个epoch保存一次
        'resume': False,
        
        # 其他配置
        'device': 'cuda',
        'early_stopping_patience': 20,
        
        # 分布式训练配置
        'distributed': True,  # 是否使用分布式训练
        'world_size': torch.cuda.device_count() if torch.cuda.is_available() else 1  # GPU数量
    }
    
    # 检查是否使用分布式训练
    if config['distributed'] and config['world_size'] > 1:
        print(f"Starting distributed training on {config['world_size']} GPUs")
        # 使用spawn方法启动多进程分布式训练
        mp.spawn(train_worker, args=(config['world_size'], config), nprocs=config['world_size'], join=True)
    else:
        print("Starting single GPU training")
        # 单GPU训练
        trainer = VLATrainer(config)
        trainer.train()

if __name__ == '__main__':
    main()
