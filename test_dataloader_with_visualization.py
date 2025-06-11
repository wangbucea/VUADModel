import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision import transforms
from vla_dataset_loader import create_vla_dataloader, VLADataset
import json

def denormalize_image(tensor_img):
    """
    反归一化图像用于可视化
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # 反归一化
    img = tensor_img * std + mean
    img = torch.clamp(img, 0, 1)
    
    # 转换为numpy并调整维度顺序
    img = img.permute(1, 2, 0).numpy()
    return img

def visualize_detection_and_segmentation():
    """
    可视化目标检测和语义分割数据
    """
    # 定义图像变换
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据加载器
    print("创建数据加载器...")
    dataloader = create_vla_dataloader(
        data_root="c:/DiskD/trae_doc/VLA",
        batch_size=2,
        shuffle=True,
        transform=transform
    )
    
    print(f"数据集大小: {len(dataloader.dataset)}")
    
    # 类别名称映射
    class_names = {0: 'bottle', 1: 'brush', 2: 'cube'}
    colors = ['red', 'green', 'blue']  # 对应不同类别的颜色
    
    # 获取第一个batch进行可视化
    for batch_idx, batch in enumerate(dataloader):
        print(f"\n=== 可视化 Batch {batch_idx} ===")
        
        batch_size = batch['chest_images'].shape[0]
        
        for sample_idx in range(batch_size):
            sample_id = batch['sample_id'][sample_idx]
            print(f"\n--- 可视化样本 {sample_idx} ({sample_id}) ---")
            
            # 获取数据
            chest_img = batch['chest_images'][sample_idx]  # [3, 224, 224]
            head_img = batch['head_images'][sample_idx]    # [3, 224, 224]
            boxes = batch['detection_boxes'][sample_idx]   # [N, 4]
            labels = batch['detection_labels'][sample_idx] # [N]
            object_ids = batch['detection_object_ids'][sample_idx] # [N]
            seg_mask = batch['segmentation_mask'][sample_idx] # [224, 224]
            key_frames = batch['key_frame_labels'][sample_idx] # [5, 7]
            
            # 反归一化图像
            chest_img_vis = denormalize_image(chest_img)
            head_img_vis = denormalize_image(head_img)
            
            # 创建可视化图像
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'样本 {sample_id} - 数据可视化', fontsize=16)
            
            # 1. Chest摄像头原图
            axes[0, 0].imshow(chest_img_vis)
            axes[0, 0].set_title('Chest摄像头原图')
            axes[0, 0].axis('off')
            
            # 2. Chest摄像头 + 目标检测框
            axes[0, 1].imshow(chest_img_vis)
            axes[0, 1].set_title('Chest摄像头 + 目标检测')
            
            # 绘制检测框
            valid_boxes = 0
            for i, (box, label, obj_id) in enumerate(zip(boxes, labels, object_ids)):
                if torch.sum(box) > 0:  # 非零框
                    x1, y1, x2, y2 = box.cpu().numpy()
                    width = x2 - x1
                    height = y2 - y1
                    
                    # 创建矩形框
                    rect = patches.Rectangle((x1, y1), width, height, 
                                           linewidth=2, edgecolor=colors[label.item() % len(colors)], 
                                           facecolor='none')
                    axes[0, 1].add_patch(rect)
                    
                    # 添加标签
                    class_name = class_names.get(label.item(), 'unknown')
                    axes[0, 1].text(x1, y1-5, f'{class_name}({obj_id.item()})', 
                                   color=colors[label.item() % len(colors)], fontsize=10, 
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                    valid_boxes += 1
            
            axes[0, 1].axis('off')
            print(f"  检测到 {valid_boxes} 个有效目标")
            
            # 3. 语义分割掩码
            axes[0, 2].imshow(seg_mask.cpu().numpy(), cmap='tab10')
            axes[0, 2].set_title('语义分割掩码')
            axes[0, 2].axis('off')
            
            unique_values = torch.unique(seg_mask)
            print(f"  分割掩码唯一值: {unique_values.tolist()}")
            
            # 4. Head摄像头原图
            axes[1, 0].imshow(head_img_vis)
            axes[1, 0].set_title('Head摄像头原图')
            axes[1, 0].axis('off')
            
            # 5. 分割掩码叠加在Chest图像上
            axes[1, 1].imshow(chest_img_vis)
            
            # 创建彩色掩码
            mask_colored = np.zeros((*seg_mask.shape, 4))  # RGBA
            for class_id in unique_values:
                if class_id > 0:  # 忽略背景
                    mask_area = (seg_mask == class_id).cpu().numpy()
                    color_idx = (class_id.item() - 1) % len(colors)
                    if colors[color_idx] == 'red':
                        mask_colored[mask_area] = [1, 0, 0, 0.5]
                    elif colors[color_idx] == 'green':
                        mask_colored[mask_area] = [0, 1, 0, 0.5]
                    elif colors[color_idx] == 'blue':
                        mask_colored[mask_area] = [0, 0, 1, 0.5]
            
            axes[1, 1].imshow(mask_colored)
            axes[1, 1].set_title('分割掩码叠加')
            axes[1, 1].axis('off')
            
            # 6. 关键帧标签可视化
            axes[1, 2].bar(range(5), [np.linalg.norm(kf) for kf in key_frames.cpu().numpy()])
            axes[1, 2].set_title('关键帧标签强度')
            axes[1, 2].set_xlabel('关键帧索引')
            axes[1, 2].set_ylabel('标签向量模长')
            axes[1, 2].grid(True, alpha=0.3)
            
            print(f"  关键帧标签形状: {key_frames.shape}")
            print(f"  关键帧标签范围: [{key_frames.min().item():.3f}, {key_frames.max().item():.3f}]")
            
            plt.tight_layout()
            
            # 保存图像
            save_path = f"c:/DiskD/trae_doc/VLA/visualization_sample_{sample_id}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  可视化结果已保存到: {save_path}")
            
            plt.show()
        
        if batch_idx == 0:  # 只可视化第一个batch
            break
    
    print("\n可视化完成！")

def visualize_raw_data():
    """
    可视化原始数据文件
    """
    print("\n=== 可视化原始数据文件 ===")
    
    sample_id = "0000"
    data_root = "c:/DiskD/trae_doc/VLA"
    
    # 加载原始图像
    chest_img_path = os.path.join(data_root, "james", sample_id, "Chest", "image_1.jpg")
    if os.path.exists(chest_img_path):
        chest_img = Image.open(chest_img_path)
        
        # 加载检测数据
        json_path = os.path.join(data_root, "merged_data", "merged_jsons", sample_id, "1.json")
        detection_data = None
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                detection_data = json.load(f)
        
        # 加载分割掩码
        mask_path = os.path.join(data_root, "merged_data", "merged_masks", sample_id, "1.png")
        seg_mask = None
        if os.path.exists(mask_path):
            seg_mask = Image.open(mask_path)
        
        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'原始数据可视化 - 样本 {sample_id}, 帧 1', fontsize=16)
        
        # 原始图像
        axes[0].imshow(chest_img)
        axes[0].set_title(f'原始图像\n尺寸: {chest_img.size}')
        axes[0].axis('off')
        
        # 图像 + 检测框
        axes[1].imshow(chest_img)
        if detection_data:
            class_names = {"bottle": 0, "brush": 1, "cube": 2}
            colors = ['red', 'green', 'blue']
            
            for obj in detection_data.get("objects", []):
                x1, y1 = obj["top_left"]
                x2, y2 = obj["bottom_right"]
                width = x2 - x1
                height = y2 - y1
                
                color_idx = class_names.get(obj["label"], 0)
                rect = patches.Rectangle((x1, y1), width, height, 
                                       linewidth=3, edgecolor=colors[color_idx], 
                                       facecolor='none')
                axes[1].add_patch(rect)
                
                axes[1].text(x1, y1-10, f'{obj["label"]}({obj["object_id"]})', 
                           color=colors[color_idx], fontsize=12, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        axes[1].set_title('原始图像 + 检测框')
        axes[1].axis('off')
        
        # 分割掩码
        if seg_mask:
            axes[2].imshow(seg_mask, cmap='tab10')
            mask_array = np.array(seg_mask)
            unique_vals = np.unique(mask_array)
            axes[2].set_title(f'分割掩码\n唯一值: {unique_vals}')
        else:
            axes[2].text(0.5, 0.5, '无分割掩码', ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('分割掩码')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # 保存原始数据可视化
        save_path = f"c:/DiskD/trae_doc/VLA/raw_data_visualization_{sample_id}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"原始数据可视化已保存到: {save_path}")
        
        plt.show()
    else:
        print(f"图像文件不存在: {chest_img_path}")

def compare_multiple_samples():
    """
    比较多个样本的数据
    """
    print("\n=== 比较多个样本 ===")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = VLADataset(data_root="c:/DiskD/trae_doc/VLA", transform=transform)
    
    # 选择前4个样本进行比较
    num_samples = min(4, len(dataset))
    
    fig, axes = plt.subplots(2, num_samples, figsize=(5*num_samples, 10))
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle('多样本数据比较', fontsize=16)
    
    for i in range(num_samples):
        try:
            sample = dataset[i]
            
            # Chest图像
            chest_img = denormalize_image(sample['chest_images'][2])  # 取中间帧
            axes[0, i].imshow(chest_img)
            axes[0, i].set_title(f'样本 {sample["sample_id"]}\nChest摄像头')
            axes[0, i].axis('off')
            
            # 分割掩码
            seg_mask = sample['segmentation_mask'].numpy()
            axes[1, i].imshow(seg_mask, cmap='tab10')
            unique_vals = np.unique(seg_mask)
            non_zero_pixels = np.sum(seg_mask > 0)
            axes[1, i].set_title(f'分割掩码\n唯一值: {len(unique_vals)}\n非零像素: {non_zero_pixels}')
            axes[1, i].axis('off')
            
            print(f"样本 {i} ({sample['sample_id']}): 检测框数量={len(sample['detection_boxes'])}, 分割类别数={len(unique_vals)}")
            
        except Exception as e:
            print(f"样本 {i} 加载失败: {e}")
            axes[0, i].text(0.5, 0.5, f'加载失败\n{str(e)[:50]}', ha='center', va='center', transform=axes[0, i].transAxes)
            axes[1, i].text(0.5, 0.5, f'加载失败\n{str(e)[:50]}', ha='center', va='center', transform=axes[1, i].transAxes)
    
    plt.tight_layout()
    
    save_path = "c:/DiskD/trae_doc/VLA/multi_sample_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"多样本比较已保存到: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    print("开始VLA数据可视化测试...")
    
    # 1. 可视化原始数据文件
    visualize_raw_data()
    
    # 2. 可视化数据加载器输出
    visualize_detection_and_segmentation()
    
    # 3. 比较多个样本
    compare_multiple_samples()
    
    print("\n所有可视化完成！")
    print("生成的可视化文件:")
    print("- raw_data_visualization_0000.png: 原始数据可视化")
    print("- visualization_sample_*.png: 数据加载器输出可视化")
    print("- multi_sample_comparison.png: 多样本比较")
