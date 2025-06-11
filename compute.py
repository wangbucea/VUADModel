import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import Dict, List, Tuple, Optional

class HungarianMatcher(nn.Module):
    """匈牙利匹配算法实现"""
    
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """
        Args:
            cost_class: 分类损失权重
            cost_bbox: 边界框L1损失权重  
            cost_giou: GIoU损失权重
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "所有成本不能都为0"
    
    @torch.no_grad()
    def forward(self, outputs: Dict, targets: List[Dict]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        执行匈牙利匹配
        
        Args:
            outputs: 模型输出字典，包含:
                - pred_logits: [B, num_queries, num_classes] 类别预测
                - pred_boxes: [B, num_queries, 4] 边界框预测
            targets: 目标列表，每个元素包含:
                - labels: [num_objects] 目标类别
                - boxes: [num_objects, 4] 目标边界框(cx, cy, w, h格式，归一化)
        
        Returns:
            匹配索引列表，每个元素为(pred_indices, target_indices)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # 展平预测结果
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [bs*num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [bs*num_queries, 4]
        
        # 拼接所有目标
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        
        # 计算分类成本 (负对数似然)
        cost_class = -out_prob[:, tgt_ids]
        
        # 计算L1成本
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # 计算GIoU成本
        cost_giou = -self._generalized_box_iou(
            self._box_cxcywh_to_xyxy(out_bbox),
            self._box_cxcywh_to_xyxy(tgt_bbox)
        )
        
        # 最终成本矩阵
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        
        # 为每个batch执行匈牙利算法
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) 
                for i, j in indices]
    
    @staticmethod
    def _box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
        """将中心点格式转换为左上右下格式"""
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)
    
    @staticmethod
    def _box_area(boxes: torch.Tensor) -> torch.Tensor:
        """计算边界框面积"""
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    def _box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算IoU"""
        area1 = self._box_area(boxes1)
        area2 = self._box_area(boxes2)
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
        
        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
        
        union = area1[:, None] + area2 - inter
        iou = inter / (union + 1e-6)  # 避免除零
        
        return iou, union
    
    def _generalized_box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """计算Generalized IoU"""
        assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), "boxes1格式错误"
        assert (boxes2[:, 2:] >= boxes2[:, :2]).all(), "boxes2格式错误"
        
        iou, union = self._box_iou(boxes1, boxes2)
        
        lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
        
        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        area = wh[:, :, 0] * wh[:, :, 1]
        
        return iou - (area - union) / (area + 1e-6)


class DETRLossComputer(nn.Module):
    """DETR损失计算器"""
    
    def __init__(self, 
                 num_classes: int,
                 matcher: HungarianMatcher,
                 weight_dict: Optional[Dict[str, float]] = None,
                 eos_coef: float = 0.1,
                 losses: List[str] = None):
        """
        Args:
            num_classes: 类别数量（不包括背景类）
            matcher: 匈牙利匹配器
            weight_dict: 损失权重字典
            eos_coef: 背景类权重
            losses: 要计算的损失类型列表
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict or {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
        self.eos_coef = eos_coef
        self.losses = losses or ['labels', 'boxes']
        
        # 创建类别权重，降低背景类的权重
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef  # 背景类索引为num_classes
        self.register_buffer('empty_weight', empty_weight)
    
    def loss_labels(self, outputs: Dict, targets: List[Dict], 
                   indices: List[Tuple], num_boxes: int) -> Dict[str, torch.Tensor]:
        """计算分类损失"""
        assert 'pred_logits' in outputs, "输出中缺少pred_logits"
        src_logits = outputs['pred_logits']  # [B, num_queries, num_classes+1]
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                  dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {'loss_ce': loss_ce}
    
    def loss_boxes(self, outputs: Dict, targets: List[Dict], 
                  indices: List[Tuple], num_boxes: int) -> Dict[str, torch.Tensor]:
        """计算边界框回归损失"""
        assert 'pred_boxes' in outputs, "输出中缺少pred_boxes"
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]  # 匹配的预测边界框
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        # L1损失
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / num_boxes
        
        # GIoU损失
        loss_giou = 1 - torch.diag(self.matcher._generalized_box_iou(
            self.matcher._box_cxcywh_to_xyxy(src_boxes),
            self.matcher._box_cxcywh_to_xyxy(target_boxes)
        ))
        loss_giou = loss_giou.sum() / num_boxes
        
        return {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}
    
    def _get_src_permutation_idx(self, indices: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取源序列的排列索引"""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def get_loss(self, loss: str, outputs: Dict, targets: List[Dict], 
                indices: List[Tuple], num_boxes: int, **kwargs) -> Dict[str, torch.Tensor]:
        """获取指定类型的损失"""
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'不支持的损失类型: {loss}'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    
    def forward(self, outputs: Dict, targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        
        Args:
            outputs: 模型输出字典，包含:
                - pred_logits: [B, num_queries, num_classes+1]
                - pred_boxes: [B, num_queries, 4]
                - aux_outputs: (可选) 辅助输出列表
            targets: 目标列表，每个元素包含:
                - labels: [num_objects] 目标类别
                - boxes: [num_objects, 4] 目标边界框
        
        Returns:
            损失字典
        """
        # 排除辅助输出
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
        # 匈牙利匹配
        indices = self.matcher(outputs_without_aux, targets)
        
        # 计算目标数量（用于归一化）
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, 
                                  device=next(iter(outputs.values())).device)
        
        # 分布式训练支持
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_boxes)
            num_boxes = num_boxes / torch.distributed.get_world_size()
        
        num_boxes = torch.clamp(num_boxes, min=1).item()
        
        # 计算主要损失
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        
        # 计算辅助损失（如果存在）
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        return losses
    
    def compute_total_loss(self, loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        根据权重字典计算总损失
        
        Args:
            loss_dict: 损失字典
        
        Returns:
            总损失值
        """
        total_loss = sum(loss_dict[k] * self.weight_dict[k] 
                        for k in loss_dict.keys() 
                        if k in self.weight_dict)
        return total_loss
