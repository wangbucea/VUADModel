import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
from tqdm import tqdm

from vla_dataset_loader import create_vla_dataloader, VLADataset
from VLAModel import RobotUniADModel
from compute import HungarianMatcher, DETRLossComputer

class VLATrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        
        # 先创建保存目录
        self.setup_directories()
        
        # 再设置日志
        self.setup_logging()
        
        # 初始化模型
        self.model = self.build_model()
        
        # 初始化数据加载器
        self.train_loader, self.val_loader = self.build_dataloaders()
        
        # 初始化优化器和调度器
        self.optimizer, self.scheduler = self.build_optimizer_and_scheduler()
        
        # 初始化TensorBoard
        self.writer = SummaryWriter(log_dir=os.path.join(self.config['save_dir'], 'tensorboard'))
        
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
        
        # 打印模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model
        
    def build_dataloaders(self):
        """构建数据加载器"""
        # 图像变换
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 训练数据加载器
        train_loader = create_vla_dataloader(
            data_root=self.config['data_root'],
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            transform=transform,
            stride=1
        )
        
        # 验证数据加载器（这里简化为使用相同数据，实际应该分离）
        val_loader = create_vla_dataloader(
            data_root=self.config['data_root'],
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            transform=transform,
            stride=1
        )
        
        self.logger.info(f"Train dataset size: {len(train_loader.dataset)}")
        self.logger.info(f"Validation dataset size: {len(val_loader.dataset)}")
        
        return train_loader, val_loader
        
    def build_optimizer_and_scheduler(self):
        """构建优化器和学习率调度器，实现差异化学习率"""
        # 分离DINOv2骨干网络参数和其他参数
        backbone_params = []
        other_params = []
        
        # 获取DINOv2参数
        if hasattr(self.model.visual_encoder, 'dinov2_view1'):
            # 双视角模型
            backbone_params.extend(list(self.model.visual_encoder.dinov2_view1.parameters()))
            backbone_params.extend(list(self.model.visual_encoder.dinov2_view2.parameters()))
        elif hasattr(self.model.visual_encoder, 'dinov2'):
            # 单视角模型
            backbone_params.extend(list(self.model.visual_encoder.dinov2.parameters()))
        
        # 获取其他所有参数
        backbone_param_ids = set(id(p) for p in backbone_params)
        for name, param in self.model.named_parameters():
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
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
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
            
            # 梯度裁剪
            if self.config['grad_clip_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip_norm'])
            
            self.optimizer.step()
            self.scheduler.step()
            
            # 记录损失
            epoch_losses.append(total_loss.item())
            
            # 更新进度条
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
        
        avg_loss = np.mean(epoch_losses)
        self.train_losses.append(avg_loss)
        
        return avg_loss
        
    def validate_epoch(self):
        """验证一个epoch"""
        self.model.eval()
        epoch_losses = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
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
        
        # 记录到TensorBoard
        self.writer.add_scalar('Val/Total_Loss', avg_loss, self.current_epoch)
        
        return avg_loss
        
    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
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
        if (self.current_epoch + 1) % self.config['save_freq'] == 0:
            epoch_path = os.path.join(self.config['save_dir'], 'checkpoints', f'epoch_{self.current_epoch+1}.pth')
            torch.save(checkpoint, epoch_path)
            
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
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.current_epoch = checkpoint['epoch']
            self.best_loss = checkpoint['best_loss']
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            
            self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
            return True
        return False
        
    def train(self):
        """主训练循环"""
        self.start_time = time.time()
        self.logger.info("Starting training...")
        
        # 尝试加载之前的检查点
        checkpoint_path = os.path.join(self.config['save_dir'], 'checkpoints', 'latest.pth')
        if self.config['resume'] and self.load_checkpoint(checkpoint_path):
            self.logger.info("Resumed training from checkpoint")
        
        for epoch in range(self.current_epoch, self.config['num_epochs']):
            self.current_epoch = epoch
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            # val_loss = self.validate_epoch()
            
            # 检查是否是最佳模型
            # is_best = val_loss < self.best_loss
            # if is_best:
            #     self.best_loss = val_loss
            
            # 保存检查点
            self.save_checkpoint()
            
            # 保存训练信息
            self.save_training_info()
            
            # 日志记录
            self.logger.info(
                f"Epoch {epoch+1}/{self.config['num_epochs']} - "
                f"Train Loss: {train_loss:.4f}"
                # f"Best Loss: {self.best_loss:.4f}"
            )
            
            # 早停检查（可选）
            # if self.config.get('early_stopping_patience'):
            #     if len(self.val_losses) > self.config['early_stopping_patience']:
            #         recent_losses = self.val_losses[-self.config['early_stopping_patience']:]
            #         if all(loss >= self.best_loss for loss in recent_losses):
            #             self.logger.info("Early stopping triggered")
            #             break
        
        total_time = time.time() - self.start_time
        self.logger.info(f"Training completed in {total_time/3600:.2f} hours")
        
        # 关闭TensorBoard writer
        self.writer.close()

def main():
    # 训练配置
    config = {
        # 数据配置
        'data_root': './',
        'batch_size': 3,
        'num_workers': 2,
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
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'backbone_lr': 1e-5,
        'weight_decay': 1e-5,
        'warmup_epochs': 10,
        'grad_clip_norm': 1.0,
        
        # 保存配置
        'save_dir': './experiments/' + datetime.now().strftime('%Y%m%d_%H%M%S'),
        'save_freq': 500,  # 每10个epoch保存一次
        'resume': False,
        
        # 其他配置
        'device': 'cuda',
        'early_stopping_patience': 20
    }
    
    # 创建训练器并开始训练
    trainer = VLATrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
