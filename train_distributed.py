#!/usr/bin/env python3
"""
分布式训练启动脚本
支持多GPU训练的VLA模型
"""

import os
import argparse
import torch
import torch.multiprocessing as mp
from train_vla import train_worker, setup_distributed, cleanup_distributed
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='VLA分布式训练')
    
    # 数据配置
    parser.add_argument('--data_root', type=str, default='./',
                        help='数据根目录')
    parser.add_argument('--batch_size', type=int, default=3,
                        help='每个GPU的批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载器工作进程数')
    parser.add_argument('--stride', type=int, default=1,
                        help='序列采样步长')
    
    # 模型配置
    parser.add_argument('--image_size', type=int, nargs=2, default=[480, 640],
                        help='图像尺寸 [height, width]')
    parser.add_argument('--use_dual_view', action='store_true',
                        help='是否使用双视角')
    
    # 训练配置
    parser.add_argument('--num_epochs', type=int, default=40,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='学习率')
    parser.add_argument('--backbone_lr', type=float, default=3e-5,
                        help='骨干网络学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0,
                        help='梯度裁剪范数')
    
    # 分布式配置
    parser.add_argument('--world_size', type=int, default=None,
                        help='GPU数量，默认使用所有可用GPU')
    parser.add_argument('--master_addr', type=str, default='localhost',
                        help='主节点地址')
    parser.add_argument('--master_port', type=str, default='12355',
                        help='主节点端口')
    parser.add_argument('--backend', type=str, default='nccl',
                        choices=['nccl', 'gloo'],
                        help='分布式后端')
    
    # 保存配置
    parser.add_argument('--save_dir', type=str, default=None,
                        help='模型保存目录')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='保存频率（每N个epoch）')
    parser.add_argument('--resume', action='store_true',
                        help='是否从检查点恢复训练')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 设置环境变量
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    
    # 确定GPU数量
    if args.world_size is None:
        world_size = torch.cuda.device_count()
    else:
        world_size = args.world_size
    
    if world_size == 0:
        raise RuntimeError("没有可用的GPU")
    
    # 设置保存目录
    if args.save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = f'./experiments/distributed_{timestamp}'
    else:
        save_dir = args.save_dir
    
    # 构建配置
    config = {
        # 数据配置
        'data_root': args.data_root,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'image_size': tuple(args.image_size),
        'stride': args.stride,
        
        # 模型配置
        'num_seg_classes': 2,
        'num_det_classes': 3,
        'num_seg_queries': 100,
        'num_det_queries': 100,
        'num_points': 5,
        'action_dim': 7,
        'seq_len': 30,
        'feature_dim': 256,
        'use_dual_view': args.use_dual_view,
        
        # 训练配置
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'backbone_lr': args.backbone_lr,
        'weight_decay': args.weight_decay,
        'warmup_epochs': 5,
        'grad_clip_norm': args.grad_clip_norm,
        
        # 保存配置
        'save_dir': save_dir,
        'save_freq': args.save_freq,
        'resume': args.resume,
        
        # 其他配置
        'device': 'cuda',
        'early_stopping_patience': 20,
        
        # 分布式配置
        'distributed': True,
        'world_size': world_size,
        'backend': args.backend
    }
    
    print(f"=== VLA分布式训练配置 ===")
    print(f"GPU数量: {world_size}")
    print(f"批次大小: {args.batch_size} (每GPU)")
    print(f"总批次大小: {args.batch_size * world_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"骨干网络学习率: {args.backbone_lr}")
    print(f"序列采样步长: {args.stride}")
    print(f"保存目录: {save_dir}")
    print(f"分布式后端: {args.backend}")
    print(f"========================")
    
    if world_size > 1:
        print(f"启动分布式训练，使用 {world_size} 个GPU")
        # 使用spawn方法启动多进程分布式训练
        mp.spawn(train_worker, args=(world_size, config), nprocs=world_size, join=True)
    else:
        print("启动单GPU训练")
        # 单GPU训练
        from train_vla import VLATrainer
        trainer = VLATrainer(config)
        trainer.train()
    
    print("训练完成！")

if __name__ == '__main__':
    main()
