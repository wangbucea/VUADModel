import os
import json
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import glob

class VLADataset(Dataset):
    """
    VLA多任务数据集加载器
    支持：双摄像头图像、机械臂动作序列、目标检测、语义分割、关键帧预测
    """
    
    def __init__(self, 
                 data_root: str = "c:/DiskD/trae_doc/VLA",
                 sequence_length: int = 5,
                 action_sequence_length: int = 30,
                 key_frames: List[int] = [128, 129, 130, 131, 134],
                 key_frame_shuffle_range: int = 15,
                 transform=None):
        """
        初始化数据集
        
        Args:
            data_root: 数据根目录
            sequence_length: 图像序列长度（默认5帧）
            action_sequence_length: 动作序列长度（默认30帧）
            key_frames: 关键帧索引列表
            key_frame_shuffle_range: 关键帧shuffle范围
            transform: 图像变换
        """
        self.data_root = data_root
        self.sequence_length = sequence_length
        self.action_sequence_length = action_sequence_length
        self.key_frames = key_frames
        self.key_frame_shuffle_range = key_frame_shuffle_range
        self.transform = transform
        
        # 类别映射
        self.class_to_idx = {'bottle': 0, 'brush': 1, 'cube': 2}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # 加载所有样本路径
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[str]:
        """加载所有样本路径并排序"""
        james_dir = os.path.join(self.data_root, "james")
        samples = []
        
        # 获取所有样本目录
        for sample_dir in os.listdir(james_dir):
            sample_path = os.path.join(james_dir, sample_dir)
            if os.path.isdir(sample_path):
                samples.append(sample_dir)
        
        # 按名称排序
        samples.sort()
        return samples
    
    def _load_action_data(self, sample_id: str) -> Dict:
        """加载动作序列数据"""
        data_path = os.path.join(self.data_root, "james", sample_id, "data.json")
        
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading action data for {sample_id}: {e}")
            return {"slave": [], "master": []}
    
    def _get_image_paths(self, sample_id: str, camera: str) -> List[str]:
        """获取指定摄像头的所有图像路径"""
        image_dir = os.path.join(self.data_root, "james", sample_id, camera)
        image_paths = []
        
        # 获取所有图像文件并按数字排序
        for img_file in os.listdir(image_dir):
            if img_file.endswith('.jpg'):
                image_paths.append(os.path.join(image_dir, img_file))
        
        # 按图像编号排序
        image_paths.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        return image_paths
    
    def _load_detection_labels(self, sample_id: str, frame_idx: int) -> Dict:
        """加载目标检测标签"""
        json_path = os.path.join(self.data_root, "merged_data", "merged_jsons", 
                                sample_id, f"{frame_idx}.json")
        
        if not os.path.exists(json_path):
            return {"boxes": [], "labels": [], "object_ids": []}
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            boxes = []
            labels = []
            object_ids = []
            
            for obj in data.get("objects", []):
                # 转换bbox格式 [x1, y1, x2, y2]
                top_left = obj["top_left"]
                bottom_right = obj["bottom_right"]
                box = [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]
                
                boxes.append(box)
                labels.append(self.class_to_idx[obj["label"]])
                object_ids.append(obj["object_id"])
            
            return {
                "boxes": np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32),
                "labels": np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64),
                "object_ids": np.array(object_ids, dtype=np.int64) if object_ids else np.zeros((0,), dtype=np.int64)
            }
        except Exception as e:
            print(f"Error loading detection labels for {sample_id}, frame {frame_idx}: {e}")
            return {"boxes": np.zeros((0, 4), dtype=np.float32), 
                   "labels": np.zeros((0,), dtype=np.int64),
                   "object_ids": np.zeros((0,), dtype=np.int64)}
    
    def _load_segmentation_mask(self, sample_id: str, frame_idx: int) -> Optional[np.ndarray]:
        """加载语义分割掩码"""
        mask_path = os.path.join(self.data_root, "merged_data", "merged_masks", 
                                sample_id, f"{frame_idx}.png")
        
        if not os.path.exists(mask_path):
            return None
        
        try:
            mask = Image.open(mask_path)
            return np.array(mask)
        except Exception as e:
            print(f"Error loading segmentation mask for {sample_id}, frame {frame_idx}: {e}")
            return None
    
    def _pad_sequence(self, sequence: List, target_length: int, pad_value=None):
        """填充序列到目标长度"""
        if len(sequence) >= target_length:
            return sequence[:target_length]
        
        if pad_value is None:
            pad_value = sequence[-1] if sequence else 0
        
        padded = sequence.copy()
        while len(padded) < target_length:
            padded.append(pad_value)
        
        return padded
    
    def _shuffle_key_frames(self, key_frames: List[int]) -> List[int]:
        """在指定范围内shuffle关键帧"""
        shuffled_frames = []
        for frame in key_frames:
            # 在±shuffle_range范围内随机选择
            min_frame = max(0, frame - self.key_frame_shuffle_range)
            max_frame = frame + self.key_frame_shuffle_range
            shuffled_frame = random.randint(min_frame, max_frame)
            shuffled_frames.append(shuffled_frame)
        
        return shuffled_frames
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample_id = self.samples[idx]
        
        # 加载动作数据
        action_data = self._load_action_data(sample_id)
        slave_actions = action_data.get("slave", [])
        master_actions = action_data.get("master", [])
        
        # 获取图像路径
        chest_paths = self._get_image_paths(sample_id, "Chest")
        head_paths = self._get_image_paths(sample_id, "Head")
        
        # 随机选择起始帧
        max_start_frame = max(0, min(len(chest_paths), len(head_paths), len(slave_actions)) - self.sequence_length)
        start_frame = random.randint(0, max_start_frame) if max_start_frame > 0 else 0
        
        # 加载图像序列
        chest_images = []
        head_images = []
        
        for i in range(self.sequence_length):
            frame_idx = start_frame + i
            
            # Chest摄像头图像
            if frame_idx < len(chest_paths):
                chest_img = Image.open(chest_paths[frame_idx]).convert('RGB')
            else:
                chest_img = Image.open(chest_paths[-1]).convert('RGB')  # 使用最后一帧填充
            
            if self.transform:
                chest_img = self.transform(chest_img)
            chest_images.append(chest_img)
            
            # Head摄像头图像
            if frame_idx < len(head_paths):
                head_img = Image.open(head_paths[frame_idx]).convert('RGB')
            else:
                head_img = Image.open(head_paths[-1]).convert('RGB')  # 使用最后一帧填充
            
            if self.transform:
                head_img = self.transform(head_img)
            head_images.append(head_img)
        
        # 获取动作序列（从起始帧开始的30帧）
        action_start_idx = start_frame
        action_end_idx = action_start_idx + self.action_sequence_length
        
        # Slave动作序列（输入）
        slave_sequence = slave_actions[action_start_idx:action_end_idx] if action_start_idx < len(slave_actions) else []
        slave_sequence = self._pad_sequence(slave_sequence, self.action_sequence_length)
        
        # Master动作序列（标签）
        master_sequence = master_actions[action_start_idx:action_end_idx] if action_start_idx < len(master_actions) else []
        master_sequence = self._pad_sequence(master_sequence, self.action_sequence_length)
        
        # 机械臂当前状态（slave序列的第一帧）
        current_state = slave_sequence[0] if slave_sequence else [0] * 7  # 假设7维状态
        
        # 加载目标检测标签（使用序列中间帧）
        detection_frame_idx = start_frame + self.sequence_length // 2 + 1  # +1因为检测标签从1开始编号
        detection_labels = self._load_detection_labels(sample_id, detection_frame_idx)
        
        # 加载语义分割掩码
        segmentation_mask = self._load_segmentation_mask(sample_id, detection_frame_idx)
        
        # 生成关键帧标签 - 修改为[5, 7]形状
        shuffled_key_frames = self._shuffle_key_frames(self.key_frames)
        
        # 创建[5, 7]形状的关键帧标签
        # 5表示5个关键帧，7表示每个关键帧的状态维度（与机械臂状态维度一致）
        key_frame_labels = np.zeros((5, 7), dtype=np.float32)
        
        # 为每个关键帧设置对应的状态标签
        for i, frame_idx in enumerate(shuffled_key_frames[:5]):  # 确保只取前5个关键帧
            if frame_idx < len(master_sequence) and i < 5:
                # 使用对应帧的master动作作为关键帧标签
                if isinstance(master_sequence[frame_idx], (list, np.ndarray)):
                    key_frame_state = master_sequence[frame_idx][:7]  # 取前7维
                    # 如果维度不足7，用0填充
                    if len(key_frame_state) < 7:
                        key_frame_state = list(key_frame_state) + [0] * (7 - len(key_frame_state))
                    key_frame_labels[i] = key_frame_state[:7]
                else:
                    # 如果master_sequence[frame_idx]是标量，创建7维向量
                    key_frame_labels[i] = [master_sequence[frame_idx]] + [0] * 6
        
        return {
            'sample_id': sample_id,
            'chest_images': torch.stack(chest_images) if chest_images else torch.zeros(self.sequence_length, 3, 640, 480),
            'head_images': torch.stack(head_images) if head_images else torch.zeros(self.sequence_length, 3, 640, 480),
            'current_state': torch.tensor(current_state, dtype=torch.float32),
            'slave_actions': torch.tensor(slave_sequence, dtype=torch.float32),
            'master_actions': torch.tensor(master_sequence, dtype=torch.float32),
            'detection_boxes': torch.tensor(detection_labels['boxes'], dtype=torch.float32),
            'detection_labels': torch.tensor(detection_labels['labels'], dtype=torch.long),
            'detection_object_ids': torch.tensor(detection_labels['object_ids'], dtype=torch.long),
            'segmentation_mask': torch.tensor(segmentation_mask, dtype=torch.long) if segmentation_mask is not None else torch.zeros(640, 480, dtype=torch.long),
            'key_frame_labels': torch.tensor(key_frame_labels, dtype=torch.float32),  # 现在形状是[5, 7]
            'start_frame': start_frame
        }

def collate_fn(batch):
    """
    自定义collate函数，处理变长的检测标签
    """
    # 分离不同类型的数据
    sample_ids = [item['sample_id'] for item in batch]
    chest_images = torch.stack([item['chest_images'] for item in batch])
    head_images = torch.stack([item['head_images'] for item in batch])
    current_states = torch.stack([item['current_state'] for item in batch])
    slave_actions = torch.stack([item['slave_actions'] for item in batch])
    master_actions = torch.stack([item['master_actions'] for item in batch])
    segmentation_masks = torch.stack([item['segmentation_mask'] for item in batch])
    key_frame_labels = torch.stack([item['key_frame_labels'] for item in batch])
    start_frames = [item['start_frame'] for item in batch]
    
    # 处理变长的检测标签
    detection_boxes = [item['detection_boxes'] for item in batch]
    detection_labels = [item['detection_labels'] for item in batch]
    detection_object_ids = [item['detection_object_ids'] for item in batch]
    
    return {
        'sample_id': sample_ids,
        'chest_images': chest_images,
        'head_images': head_images,
        'current_state': current_states,
        'slave_actions': slave_actions,
        'master_actions': master_actions,
        'detection_boxes': detection_boxes,  # 保持为列表
        'detection_labels': detection_labels,  # 保持为列表
        'detection_object_ids': detection_object_ids,  # 保持为列表
        'segmentation_mask': segmentation_masks,
        'key_frame_labels': key_frame_labels,
        'start_frame': start_frames
    }

def create_vla_dataloader(data_root: str = "c:/DiskD/trae_doc/VLA",
                         batch_size: int = 4,
                         shuffle: bool = True,
                         num_workers: int = 4,
                         transform=None):
    """
    创建VLA数据加载器
    
    Args:
        data_root: 数据根目录
        batch_size: 批次大小
        shuffle: 是否随机打乱
        num_workers: 工作进程数
        transform: 图像变换
    
    Returns:
        DataLoader对象
    """
    dataset = VLADataset(data_root=data_root, transform=transform)
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn  # 添加自定义collate函数
    )
    
    return dataloader

# 使用示例
if __name__ == "__main__":
    from torchvision import transforms
    
    # 定义图像变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据加载器
    dataloader = create_vla_dataloader(
        data_root="c:/DiskD/trae_doc/VLA",
        batch_size=2,
        transform=transform
    )
    
    # 测试数据加载
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Chest images shape: {batch['chest_images'].shape}")
        print(f"  Head images shape: {batch['head_images'].shape}")
        print(f"  Current state shape: {batch['current_state'].shape}")
        print(f"  Slave actions shape: {batch['slave_actions'].shape}")
        print(f"  Master actions shape: {batch['master_actions'].shape}")
        print(f"  Detection boxes shape: {batch['detection_boxes']}")
        print(f"  Detection labels shape: {batch['detection_labels']}")
        print(f"  Segmentation mask shape: {batch['segmentation_mask'].shape}")
        print(f"  Key frame labels shape: {batch['key_frame_labels'].shape}")
        
        if batch_idx == 0:  # 只打印第一个batch
            break
