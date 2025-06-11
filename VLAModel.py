import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, List
from transformers import Dinov2Model, Dinov2Config
from DeformableTransformer import *
from compute import *

class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DualViewDINOv2VisualEncoder(nn.Module):
    """双视角DINOv2视觉编码器"""
    def __init__(self, model_name="facebook/dinov2-base", freeze_backbone=False, image_size=(640, 480)):
        super().__init__()
        
        # 第一个视角的DINOv2编码器
        self.dinov2_view1 = Dinov2Model.from_pretrained(model_name)
        # 第二个视角的DINOv2编码器
        self.dinov2_view2 = Dinov2Model.from_pretrained(model_name)
        
        if freeze_backbone:
            for param in self.dinov2_view1.parameters():
                param.requires_grad = False
            for param in self.dinov2_view2.parameters():
                param.requires_grad = False
        
        # 特征维度适配
        self.feature_dim = self.dinov2_view1.config.hidden_size  # 768 for base model
        self.feature_adapter_view1 = nn.Linear(self.feature_dim, 256)
        self.feature_adapter_view2 = nn.Linear(self.feature_dim, 256)
        
        # 多尺度特征投影
        self.level_embed = nn.Parameter(torch.Tensor(4, 256))
        nn.init.normal_(self.level_embed)
        
        # 视角融合模块
        self.view_fusion = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            batch_first=True
        )
        
        # 视角标识嵌入
        self.view1_embed = nn.Parameter(torch.randn(1, 1, 256))
        self.view2_embed = nn.Parameter(torch.randn(1, 1, 256))
        
        # 融合后的特征投影
        self.fusion_projection = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
    def encode_single_view(self, images, dinov2_model, feature_adapter, view_embed):
        """
        编码单个视角的图像
        Args:
            images: [B, T, C, H, W] 单个视角的输入图像
            dinov2_model: DINOv2模型
            feature_adapter: 特征适配器
            view_embed: 视角嵌入
        Returns:
            Vfeatures_list: List[[B, \sum H_i*W_i, 256]] 每个时间步的多尺度特征
        """
        B, T, C, H, W = images.shape
        Vfeatures_list = []
        
        # 分别处理每个时间步的图像
        for t in range(T):
            current_images = images[:, t, :, :, :]  # [B, C, H, W]
            
            with torch.no_grad() if hasattr(self, 'freeze_backbone') else torch.enable_grad():
                outputs = dinov2_model(current_images)
                # 获取patch embeddings
                patch_embeddings = outputs.last_hidden_state  # [B, N+1, 768]
                # 移除CLS token
                base_features = patch_embeddings[:, 1:, :]  # [B, N, 768]
            
            # 特征维度适配
            base_features = feature_adapter(base_features)  # [B, N, 256]
            
            # 添加视角嵌入
            base_features = base_features + view_embed.expand(B, base_features.shape[1], -1)
            
            # 动态计算patch的空间维度
            num_patches = base_features.shape[1]  # N
            patch_h = patch_w = int(math.sqrt(num_patches))
            
            # 如果不是完全平方数，需要根据实际的图像尺寸比例计算
            if patch_h * patch_w != num_patches:
                # 根据输入图像的宽高比计算patch维度
                aspect_ratio = W / H  # 640/480 = 4/3
                patch_h = int(math.sqrt(num_patches / aspect_ratio))
                patch_w = int(num_patches / patch_h)
                
                # 确保patch_h * patch_w == num_patches
                if patch_h * patch_w != num_patches:
                    # 如果仍然不匹配，使用最接近的平方根
                    patch_h = patch_w = int(math.sqrt(num_patches))
                    # 截断多余的patch
                    base_features = base_features[:, :patch_h*patch_w, :]
            
            # 重塑为空间特征图
            spatial_features = base_features.view(B, patch_h, patch_w, 256).permute(0, 3, 1, 2)  # [B, 256, H, W]
            
            # 生成多尺度特征
            multi_scale_features = []
            
            # 4个尺度的特征
            scales = [1, 2, 4, 8]  # 下采样倍数
            
            for i, scale in enumerate(scales):
                if scale == 1:
                    feat = spatial_features
                else:
                    feat = F.avg_pool2d(spatial_features, kernel_size=scale, stride=scale)
                
                # 展平并添加位置编码
                feat_flat = feat.flatten(2).transpose(1, 2)  # [B, H*W, 256]
                feat_flat = feat_flat + self.level_embed[i].view(1, 1, -1)
                multi_scale_features.append(feat_flat)
            
            # 拼接所有尺度的特征
            Vfeatures = torch.cat(multi_scale_features, dim=1)  # [B, \sum H_i*W_i, 256]
            Vfeatures_list.append(Vfeatures)
        
        return Vfeatures_list
    
    def forward(self, images):
        """
        Args:
            images[0]: [B, T, C, H, W] 第一个视角的输入图像，T=2表示相邻两个时间步
            images[1]: [B, T, C, H, W] 第二个视角的输入图像，T=2表示相邻两个时间步
        Returns:
            Vfeatures_list: List[[B, \sum H_i*W_i, 256]] 每个时间步的融合多尺度特征
            spatial_shapes: [n_levels, 2] 每个尺度的空间形状
            level_start_index: [n_levels] 每个尺度的起始索引
            valid_ratios: [B, n_levels, 2] 有效区域比例
        """
        B, T, C, H, W = images[0].shape
        # assert T == 2, "Expected 2 time steps for temporal processing"
        assert images[0].shape == images[1].shape, "Both views must have the same shape"
        
        # 编码两个视角
        Vfeatures_list_view1 = self.encode_single_view(
            images[0], self.dinov2_view1, self.feature_adapter_view1, self.view1_embed
        )
        Vfeatures_list_view2 = self.encode_single_view(
            images[1], self.dinov2_view2, self.feature_adapter_view2, self.view2_embed
        )
        
        # 融合两个视角的特征
        fused_Vfeatures_list = []
        for t in range(T):
            view1_features = Vfeatures_list_view1[t]  # [B, N, 256]
            view2_features = Vfeatures_list_view2[t]  # [B, N, 256]
            
            # 使用交叉注意力融合两个视角
            # view1作为query，view2作为key和value
            fused_features_1, _ = self.view_fusion(
                query=view1_features,
                key=view2_features,
                value=view2_features
            )
            
            # view2作为query，view1作为key和value
            fused_features_2, _ = self.view_fusion(
                query=view2_features,
                key=view1_features,
                value=view1_features
            )
            
            # 平均融合两个方向的注意力结果
            fused_features = (fused_features_1 + fused_features_2) / 2
            
            # 残差连接和投影
            fused_features = fused_features + view1_features + view2_features
            fused_features = self.fusion_projection(fused_features)
            
            fused_Vfeatures_list.append(fused_features)
        
        # 计算spatial_shapes和level_start_index（基于第一个视角）
        # 这些参数在两个视角中应该是相同的
        spatial_shapes = []
        level_start_index = []
        scales = [1, 2, 4, 8]
        
        # 使用第一个视角的图像计算空间形状
        sample_image = images[0][:1, 0, :, :, :]  # [1, C, H, W]
        with torch.no_grad():
            outputs = self.dinov2_view1(sample_image)
            patch_embeddings = outputs.last_hidden_state[:, 1:, :]
            num_patches = patch_embeddings.shape[1]
            patch_h = patch_w = int(math.sqrt(num_patches))
            
            if patch_h * patch_w != num_patches:
                aspect_ratio = W / H
                patch_h = int(math.sqrt(num_patches / aspect_ratio))
                patch_w = int(num_patches / patch_h)
                if patch_h * patch_w != num_patches:
                    patch_h = patch_w = int(math.sqrt(num_patches))
        
        start_idx = 0
        for i, scale in enumerate(scales):
            if scale == 1:
                h, w = patch_h, patch_w
            else:
                h, w = max(1, patch_h // scale), max(1, patch_w // scale)
            
            spatial_shapes.append([h, w])
            level_start_index.append(start_idx)
            start_idx += h * w
        
        # 转换为tensor格式
        spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=images[0].device)
        level_start_index = torch.tensor(level_start_index, dtype=torch.long, device=images[0].device)
        
        # 有效区域比例（假设全部有效）
        valid_ratios = torch.ones(B, len(scales), 2, device=images[0].device)
        
        return fused_Vfeatures_list, spatial_shapes, level_start_index, valid_ratios

class DINOv2VisualEncoder(nn.Module):
    """基于DINOv2的视觉编码器"""
    def __init__(self, model_name="facebook/dinov2-base", freeze_backbone=False, image_size=(640, 480)):
        super().__init__()
        # 加载预训练的DINOv2模型
        self.dinov2 = Dinov2Model.from_pretrained(model_name)
        
        if freeze_backbone:
            for param in self.dinov2.parameters():
                param.requires_grad = False
        
        # 特征维度适配
        self.feature_dim = self.dinov2.config.hidden_size  # 768 for base model
        self.feature_adapter = nn.Linear(self.feature_dim, 256)
        
        # 多尺度特征投影
        self.level_embed = nn.Parameter(torch.Tensor(4, 256))
        nn.init.normal_(self.level_embed)
        
    def forward(self, images):
        """
        Args:
            images: [B, T, C, H, W] 输入图像，T=2表示相邻两个时间步
        Returns:
            Vfeatures_list: List[[B, \sum H_i*W_i, 256]] 每个时间步的多尺度特征
            spatial_shapes: [n_levels, 2] 每个尺度的空间形状
            level_start_index: [n_levels] 每个尺度的起始索引
            valid_ratios: [B, n_levels, 2] 有效区域比例
        """
        B, T, C, H, W = images.shape
        assert T == 2, "Expected 2 time steps for temporal processing"
        
        Vfeatures_list = []
        spatial_shapes = []  # 存储[h, w]对
        level_start_index = []
        
        # 分别处理每个时间步的图像
        for t in range(T):
            current_images = images[:, t, :, :, :]  # [B, C, H, W]
            
            with torch.no_grad() if hasattr(self, 'freeze_backbone') else torch.enable_grad():
                outputs = self.dinov2(current_images)
                # 获取patch embeddings
                patch_embeddings = outputs.last_hidden_state  # [B, N+1, 768]
                # 移除CLS token
                base_features = patch_embeddings[:, 1:, :]  # [B, N, 768]
            
            # 特征维度适配
            base_features = self.feature_adapter(base_features)  # [B, N, 256]
            
            # 动态计算patch的空间维度
            num_patches = base_features.shape[1]  # N
            patch_h = patch_w = int(math.sqrt(num_patches))
            
            # 如果不是完全平方数，需要根据实际的图像尺寸比例计算
            if patch_h * patch_w != num_patches:
                # 根据输入图像的宽高比计算patch维度
                aspect_ratio = W / H  # 640/480 = 4/3
                patch_h = int(math.sqrt(num_patches / aspect_ratio))
                patch_w = int(num_patches / patch_h)
                
                # 确保patch_h * patch_w == num_patches
                if patch_h * patch_w != num_patches:
                    # 如果仍然不匹配，使用最接近的平方根
                    patch_h = patch_w = int(math.sqrt(num_patches))
                    # 截断多余的patch
                    base_features = base_features[:, :patch_h*patch_w, :]
            
            # 重塑为空间特征图
            spatial_features = base_features.view(B, patch_h, patch_w, 256).permute(0, 3, 1, 2)  # [B, 256, H, W]
            
            # 生成多尺度特征
            multi_scale_features = []
            
            # 4个尺度的特征
            scales = [1, 2, 4, 8]  # 下采样倍数
            start_idx = 0
            
            for i, scale in enumerate(scales):
                if scale == 1:
                    feat = spatial_features
                else:
                    feat = F.avg_pool2d(spatial_features, kernel_size=scale, stride=scale)
                
                h, w = feat.shape[2], feat.shape[3]
                if t == 0:  # 只在第一个时间步记录spatial_shapes和level_start_index
                    spatial_shapes.append([h, w])  # 添加[h, w]对
                    level_start_index.append(start_idx)
                start_idx += h * w
                
                # 展平并添加位置编码
                feat_flat = feat.flatten(2).transpose(1, 2)  # [B, H*W, 256]
                feat_flat = feat_flat + self.level_embed[i].view(1, 1, -1)
                multi_scale_features.append(feat_flat)
            
            # 拼接所有尺度的特征
            Vfeatures = torch.cat(multi_scale_features, dim=1)  # [B, \sum H_i*W_i, 256]
            Vfeatures_list.append(Vfeatures)
        
        # 转换为正确的2D tensor格式
        spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=images.device)  # [n_levels, 2]
        level_start_index = torch.tensor(level_start_index, dtype=torch.long, device=images.device)  # [n_levels]
        
        # 有效区域比例（假设全部有效）
        valid_ratios = torch.ones(B, len(scales), 2, device=images.device)
        
        return Vfeatures_list, spatial_shapes, level_start_index, valid_ratios

class TemporalSpatialFusion(nn.Module):
    """时空特征融合模块"""
    def __init__(self, feature_dim=256, num_heads=8, num_layers=2):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 时序自注意力
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # 空间交叉注意力
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 层归一化
        self.temporal_norm = nn.LayerNorm(feature_dim)
        self.spatial_norm = nn.LayerNorm(feature_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(0.1)
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, Vfeatures_list):
        """
        Args:
            Vfeatures_list: List[[B, N, 256]] 包含T个时间步的视觉特征列表
        Returns:
            fused_features: [B, N, 256] 融合后的特征
        """
        # 将列表中的特征拼接成时序维度
        # Vfeatures_list: [tensor1[B, N, 256], tensor2[B, N, 256]]
        # 转换为: [B, T, N, 256]
        temporal_features = torch.stack(Vfeatures_list, dim=1)  # [B, T, N, 256]
        B, T, N, D = temporal_features.shape
        
        # 重塑为 [B*N, T, 256] 以便进行时序注意力
        temporal_features = temporal_features.permute(0, 2, 1, 3).contiguous()  # [B, N, T, 256]
        temporal_features = temporal_features.view(B * N, T, D)  # [B*N, T, 256]
        
        # 时序自注意力
        temporal_out, _ = self.temporal_attention(temporal_features, temporal_features, temporal_features)
        temporal_features = self.temporal_norm(temporal_features + temporal_out)
        
        # 取最后一个时间步的特征或者平均池化
        # 这里我们取最后一个时间步（当前时刻）
        current_features = temporal_features[:, -1, :]  # [B*N, 256]
        current_features = current_features.view(B, N, D)  # [B, N, 256]
        
        # 空间自注意力
        spatial_out, _ = self.spatial_attention(current_features, current_features, current_features)
        current_features = self.spatial_norm(current_features + spatial_out)
        
        # 前馈网络
        ffn_out = self.ffn(current_features)
        fused_features = self.ffn_norm(current_features + ffn_out)
        
        return fused_features

class TrajectoryAutoregressiveModel(nn.Module):
    """自回归动作轨迹生成模型，替换扩散模型"""
    def __init__(self, 
                 action_dim=7, 
                 seq_len=30, 
                 feature_dim=256, 
                 hidden_dims=[128, 256, 512],
                 num_heads=8,
                 num_layers=6):
        super().__init__()
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # 输入状态序列编码器
        self.action_embed = nn.Linear(action_dim, feature_dim)
        self.pos_embed = nn.Parameter(torch.randn(seq_len, feature_dim))
        self.state_pos_embed = nn.Parameter(torch.randn(1, feature_dim))

        self.state_encoder = nn.Linear(feature_dim, feature_dim)
        
        # 条件编码器（多模态输入融合）
        self.seg_encoder = nn.Linear(feature_dim, feature_dim)
        self.det_encoder = nn.Linear(feature_dim, feature_dim)
        self.visual_encoder = nn.Linear(feature_dim, feature_dim)
        self.point_encoder = nn.Linear(feature_dim, feature_dim)
        
        # 自回归解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.autoregressive_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 输出头
        self.output_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, action_dim)
        )
        
        # 开始标记和结束标记
        self.start_token = nn.Parameter(torch.randn(1, 1, action_dim))
        
    def encode_input_state(self, state):
        """
        对输入的完整动作序列进行编码
        Args:
            action_sequence: [B, action_dim] 输入动作序列
        Returns:
            encoded_sequence: [B, feature_dim] 编码后的序列特征
        """
        B = state.shape[0]
        
        # 动作嵌入
        action_emb = self.action_embed(state)  # [B, feature_dim]
        
        # 位置编码
        pos_emb = self.state_pos_embed.repeat(B, 1)  # [B, feature_dim]
        # 序列编码
        sequence_input = action_emb + pos_emb
        encoded_sequence = self.state_encoder(sequence_input)
        
        return encoded_sequence
        
    def forward(self, state, seg_features, det_features, Vfeatures, point_features, target_actions=None):
        """
        Args:
            state: [B, action_dim] 机械臂状态
            seg_features: [B, num_seg_queries, 256] 语义分割特征
            det_features: [B, num_det_queries, 256] 目标检测特征
            Vfeatures: [B, N, 256] 视觉特征
            point_features: [B, num_points, 256] 点预测特征
            target_actions: [B, seq_len, action_dim] 目标动作序列（训练时使用）
        Returns:
            output_actions: [B, seq_len, action_dim] 生成的动作序列
        """
        B = state.shape[0]
        
        # 1. 编码输入状态
        encoded_input = self.encode_input_state(state)  # [B, feature_dim]
        encoded_input = encoded_input.unsqueeze(1)  # [B, 1, feature_dim]
        encoded_input = encoded_input.repeat(1, self.seq_len, 1)  # [B, seq_len, feature_dim]

        # 2. 多模态条件编码
        seg_encoded = self.seg_encoder(seg_features)
        det_encoded = self.det_encoder(det_features)
        visual_encoded = self.visual_encoder(Vfeatures)
        point_encoded = self.point_encoder(point_features)
        
        # 拼接所有条件特征作为memory
        condition_memory = torch.cat([
            seg_encoded, det_encoded, visual_encoded, point_encoded
        ], dim=1)  # [B, total_features, feature_dim]
        
        if self.training and target_actions is not None:
            # 训练模式：teacher forcing
            # 准备解码器输入（添加开始标记）
            start_tokens = self.start_token.expand(B, 1, -1)  # [B, 1, action_dim]
            decoder_input_actions = torch.cat([start_tokens, target_actions[:, :-1]], dim=1)  # [B, seq_len, action_dim]
            
            # 嵌入解码器输入
            decoder_input_emb = self.action_embed(decoder_input_actions)
            pos_emb = self.pos_embed.unsqueeze(0).repeat(B, 1, 1)
            decoder_queries = decoder_input_emb + pos_emb + encoded_input
            
            # 创建因果掩码
            tgt_mask = self._generate_square_subsequent_mask(self.seq_len).to(state.device)
            
            # 自回归解码
            decoded_features = self.autoregressive_decoder(
                tgt=decoder_queries,
                memory=condition_memory,
                tgt_mask=tgt_mask
            )
            
            # 生成输出动作
            output_actions = self.output_head(decoded_features)
            
        else:
            # 推理模式：自回归生成
            output_actions = self._autoregressive_generate(
                encoded_input, condition_memory, B
            )
            
        return output_actions
    
    def _generate_square_subsequent_mask(self, sz):
        """生成因果掩码"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    @torch.no_grad()
    def _autoregressive_generate(self, encoded_input, condition_memory, batch_size):
        """自回归生成动作序列"""
        device = encoded_input.device
        
        # 初始化输出序列
        generated_actions = []
        
        # 开始标记
        current_input = self.start_token.expand(batch_size, 1, -1)  # [B, 1, action_dim]
        
        for step in range(self.seq_len):
            # 嵌入当前输入
            current_emb = self.action_embed(current_input)  # [B, step+1, feature_dim]
            
            # 位置编码
            current_len = current_emb.shape[1]
            pos_emb = self.pos_embed[:current_len].unsqueeze(0).repeat(batch_size, 1, 1)
            
            # 结合编码的输入序列特征
            if step < self.seq_len:
                input_context = encoded_input[:, :step+1]  # [B, step+1, feature_dim]
            else:
                input_context = encoded_input
            
            decoder_queries = current_emb + pos_emb + input_context
            
            # 解码
            decoded_features = self.autoregressive_decoder(
                tgt=decoder_queries,
                memory=condition_memory
            )
            
            # 生成下一个动作
            next_action = self.output_head(decoded_features[:, -1:])  # [B, 1, action_dim]
            generated_actions.append(next_action)
            
            # 更新输入序列
            current_input = torch.cat([current_input, next_action], dim=1)
        
        return torch.cat(generated_actions, dim=1)  # [B, seq_len, action_dim]

    def generate(self, action_sequence, seg_features, det_features, Vfeatures, point_features):
        """推理接口"""
        return self.forward(
            action_sequence, seg_features, det_features, Vfeatures, point_features
        )
class PixelTransformerDecoder(nn.Module):
    """基于Transformer的像素级解码器"""
    def __init__(self, feature_dim=256, num_classes=10, image_size=(640, 480)):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.image_size = image_size
        
        # 明确定义：image_size = (width, height) = (640, 480)
        self.image_width = image_size[0]   # 640
        self.image_height = image_size[1]  # 480
        
        # 使用少量可学习的分割查询而不是每像素查询
        self.num_seg_queries = 100
        self.seg_queries = nn.Parameter(torch.randn(self.num_seg_queries, feature_dim))
        self.seg_pos_embed = nn.Parameter(torch.randn(self.num_seg_queries, feature_dim))
        
        # Transformer解码器层（减少层数）
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=feature_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        
        # 上采样网络：将查询特征上采样到像素级
        self.upsampling_layers = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, feature_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_dim // 2, feature_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_dim // 4, feature_dim // 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_dim // 8, feature_dim // 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_dim // 16, self.num_classes, kernel_size=4, stride=2, padding=1)
        )
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, seg_features, visual_memory):
        """
        Args:
            seg_features: [B, num_seg_queries, feature_dim] 分割特征
            visual_memory: [B, N, feature_dim] 视觉记忆特征
        Returns:
            pixel_logits: [B, num_classes, H, W] 像素级分类结果
        """
        batch_size = seg_features.shape[0]
        device = seg_features.device
        
        # 获取分割查询
        seg_queries = self.seg_queries.unsqueeze(0).repeat(batch_size, 1, 1)
        seg_queries_with_pos = seg_queries + self.seg_pos_embed.unsqueeze(0)
        
        # 构建记忆：分割特征 + 视觉特征
        memory = torch.cat([seg_features, visual_memory], dim=1)
        
        # Transformer解码
        decoded_features = self.transformer_decoder(
            tgt=seg_queries_with_pos,
            memory=memory
        )
        
        # 特征融合
        fused_features = self.feature_fusion(decoded_features)
        
        # 计算初始特征图尺寸
        init_h = max(1, self.image_height // 64)  # 480//64 = 7.5 -> 8
        init_w = max(1, self.image_width // 64)   # 640//64 = 10
        
        # 将查询特征重塑为特征图
        feature_map = fused_features.transpose(1, 2)  # [B, feature_dim, num_seg_queries]
        feature_map = F.adaptive_avg_pool1d(feature_map, init_h * init_w)
        feature_map = feature_map.view(batch_size, self.feature_dim, init_h, init_w)
        
        # 上采样到目标尺寸
        pixel_logits = self.upsampling_layers(feature_map)
        
        # 确保输出尺寸正确：(B, C, H, W) = (B, num_classes, 640, 480)
        target_size = self.image_size
        if pixel_logits.shape[2:] != target_size:
            pixel_logits = F.interpolate(
                pixel_logits, 
                size=target_size,  # (640, 480) = (H, W)
                mode='bilinear', 
                align_corners=False
            )
        
        return pixel_logits

class UnifiedTransformerHead(nn.Module):
    """统一的Transformer头，整合语义分割、目标检测和点预测"""
    def __init__(self, 
                 feature_dim=256, 
                 num_seg_queries=256,
                 num_det_queries=256, 
                 num_point_queries=3,
                 num_seg_classes=10,
                 num_det_classes=10,
                 num_layers=6,
                 n_levels=4,
                 image_size=(640, 480)):
        super().__init__()
        
        self.num_seg_queries = num_seg_queries
        self.num_det_queries = num_det_queries
        self.num_point_queries = num_point_queries
        self.num_seg_classes = num_seg_classes
        self.num_det_classes = num_det_classes
        self.image_size = image_size
        
        # 统一的查询嵌入
        total_queries = num_seg_queries + num_det_queries + num_point_queries
        self.unified_query_embed = nn.Embedding(total_queries, feature_dim)
        self.unified_query_pos = nn.Embedding(total_queries, feature_dim)
        
        # 任务特定的查询类型嵌入
        self.task_type_embed = nn.Embedding(3, feature_dim)  # 3种任务类型
        
        # 统一的Transformer解码器
        decoder_layer = DeformableTransformerDecoderLayer(
            d_model=feature_dim,
            n_levels=n_levels,
            n_heads=8,
            n_points=4
        )
        self.unified_decoder = DeformableTransformerDecoder(
            decoder_layer, num_layers, return_intermediate=True
        )
        
        # 跨任务注意力模块
        self.cross_task_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )
        self.cross_task_norm = nn.LayerNorm(feature_dim)
        
        # 任务特定的输出头
        # 1. 语义分割特征头
        self.seg_feature_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 2. 基于Transformer的像素级分割解码器
        self.pixel_transformer_decoder = PixelTransformerDecoder(
            feature_dim=feature_dim,
            num_classes=num_seg_classes,
            image_size=image_size
        )
        
        # 3. 目标检测头
        self.det_class_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_det_classes + 1)
        )
        
        self.det_bbox_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 4)
        )
        
        # 4. 点预测头
        self.point_coord_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 7)  # 抓住物体时候机械臂的状态
        )
        
        # 参考点初始化
        self.reference_points = nn.Linear(feature_dim, 4)
        
        # 查询间交互的权重
        self.seg_to_det_proj = nn.Linear(feature_dim, feature_dim)
        self.det_to_seg_proj = nn.Linear(feature_dim, feature_dim)
        self.seg_to_point_proj = nn.Linear(feature_dim, feature_dim)
        self.det_to_point_proj = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, Vfeatures, spatial_shapes, level_start_index, valid_ratios, padding_mask=None):
        """
        Args:
            Vfeatures: [B, \sum H_i*W_i, 256] 多尺度特征
            spatial_shapes: [n_levels, 2] 每个尺度的空间形状
            level_start_index: [n_levels] 每个尺度的起始索引
            valid_ratios: [B, n_levels, 2] 有效区域比例
            padding_mask: [B, \sum H_i*W_i] 填充掩码
        Returns:
            unified_outputs: 包含所有任务输出的字典
        """
        B = Vfeatures.shape[0]
        
        # 构建统一的查询嵌入
        unified_queries = self.unified_query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        unified_query_pos = self.unified_query_pos.weight.unsqueeze(0).repeat(B, 1, 1)
        
        # 添加任务类型嵌入
        task_embeds = []
        # 分割任务嵌入
        seg_task_embed = self.task_type_embed.weight[0].unsqueeze(0).repeat(B, self.num_seg_queries, 1)
        task_embeds.append(seg_task_embed)
        # 检测任务嵌入
        det_task_embed = self.task_type_embed.weight[1].unsqueeze(0).repeat(B, self.num_det_queries, 1)
        task_embeds.append(det_task_embed)
        # 点预测任务嵌入
        point_task_embed = self.task_type_embed.weight[2].unsqueeze(0).repeat(B, self.num_point_queries, 1)
        task_embeds.append(point_task_embed)
        
        task_type_embeds = torch.cat(task_embeds, dim=1)  # [B, total_queries, 256]
        
        # 融合查询嵌入和任务类型嵌入
        enhanced_queries = unified_queries + unified_query_pos + task_type_embeds
        
        # 初始化参考点
        reference_points = self.reference_points(enhanced_queries).sigmoid()
        
        # 统一的可变形解码器
        hs, inter_references = self.unified_decoder(
            tgt=enhanced_queries,
            reference_points=reference_points,
            src=Vfeatures,
            src_spatial_shapes=spatial_shapes,
            src_level_start_index=level_start_index,
            src_valid_ratios=valid_ratios,
            query_pos=unified_query_pos,
            src_padding_mask=padding_mask
        )
        
        # 使用最后一层的输出
        unified_features = hs[-1]  # [B, total_queries, 256]
        
        # 分离不同任务的特征
        seg_start = 0
        seg_end = self.num_seg_queries
        det_start = seg_end
        det_end = det_start + self.num_det_queries
        point_start = det_end
        point_end = point_start + self.num_point_queries
        
        seg_features = unified_features[:, seg_start:seg_end, :]  # [B, num_seg_queries, 256]
        det_features = unified_features[:, det_start:det_end, :]   # [B, num_det_queries, 256]
        point_features = unified_features[:, point_start:point_end, :]  # [B, num_point_queries, 256]
        
        # 跨任务信息交互
        # 分割 <-> 检测
        seg_enhanced, _ = self.cross_task_attention(
            query=seg_features,
            key=det_features,
            value=det_features
        )
        seg_features = self.cross_task_norm(seg_features + seg_enhanced)
        
        det_enhanced, _ = self.cross_task_attention(
            query=det_features,
            key=seg_features,
            value=seg_features
        )
        det_features = self.cross_task_norm(det_features + det_enhanced)
        
        # 点预测与分割、检测的交互
        seg_to_point = self.seg_to_point_proj(seg_features.mean(dim=1, keepdim=True))  # [B, 1, 256]
        det_to_point = self.det_to_point_proj(det_features.mean(dim=1, keepdim=True))  # [B, 1, 256]
        
        point_enhanced = point_features + seg_to_point + det_to_point
        
        # 任务特定的输出预测
        # 1. 语义分割输出
        seg_query_features = self.seg_feature_head(seg_features)
        
        # 使用Transformer解码器进行像素级分割预测
        seg_logits = self.pixel_transformer_decoder(
            seg_features=seg_query_features,
            visual_memory=Vfeatures
        )  # [B, num_classes, H, W] - 直接输出到目标尺寸
        
        # 2. 目标检测输出
        det_class_logits = self.det_class_head(det_features)  # [B, num_det_queries, num_classes+1]
        det_bbox_coords = self.det_bbox_head(det_features).sigmoid()  # [B, num_det_queries, 4] 应用sigmoid
        
        # 3. 点预测输出
        point_coords = self.point_coord_head(point_enhanced)  # [B, num_points, 7]
        
        return {
            'seg_features': seg_query_features,
            'seg_logits': seg_logits,
            'det_features': det_features,
            'class_logits': det_class_logits,
            'bbox_coords': det_bbox_coords,
            'point_features': point_enhanced,
            'point_coords': point_coords,
            'unified_features': unified_features  # 用于轨迹预测的统一特征
        }

class RobotUniADModel(nn.Module):
    def __init__(self, 
                 num_seg_classes=10,
                 num_det_classes=10, 
                 num_seg_queries=256,
                 num_det_queries=256,
                 num_points=3,
                 action_dim=7,
                 seq_len=30,
                 feature_dim=256,
                 image_size=(640, 480),
                 use_markov_regularizer=True,
                 markov_weight=0.1,
                 use_dual_view=True):
        super().__init__()
        
        # 1. 视觉编码器
        if use_dual_view:
            self.visual_encoder = DualViewDINOv2VisualEncoder(image_size=image_size)
        else:
            self.visual_encoder = DINOv2VisualEncoder(image_size=image_size)
        
        # 2. 时空特征融合
        self.temporal_spatial_fusion = TemporalSpatialFusion(feature_dim)
        
        # 3. 统一的Transformer头
        self.unified_transformer = UnifiedTransformerHead(
            feature_dim=feature_dim,
            num_seg_queries=num_seg_queries,
            num_det_queries=num_det_queries,
            num_point_queries=num_points,
            num_seg_classes=num_seg_classes,
            num_det_classes=num_det_classes,
            image_size=image_size
        )
        
        # 4. 动作轨迹模型
        self.trajectory_autoregressive = TrajectoryAutoregressiveModel(
            action_dim=action_dim,
            seq_len=seq_len,
            feature_dim=feature_dim
        )
        # 5. DETR风格的损失计算组件
        self.matcher = HungarianMatcher()
        self.criterion = DETRLossComputer(
            num_classes=num_det_classes,
            matcher=self.matcher,
            weight_dict={'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2},
            eos_coef=0.1,
            losses=['labels', 'boxes']
        )
        
    def forward(self, images, input_actions=None, target_actions=None, det_targets=None):
        """
        Args:
            images: [B, T, C, H, W] 输入图像序列
            target_actions: [B, seq_len, action_dim] 目标动作序列（训练时使用）
            det_targets: List[Dict] 检测目标标注（训练时使用）
        Returns:
            outputs: 包含所有任务输出的字典
        """
        # 1. 视觉编码
        Vfeatures_list, spatial_shapes, level_start_index, valid_ratios = self.visual_encoder(images)
        
        # 2. 时空特征融合
        Vfeatures = self.temporal_spatial_fusion(Vfeatures_list)  # [B, 512, 256]
        
        # 3. 统一Transformer处理所有任务（传递检测targets）
        unified_outputs = self.unified_transformer(
            Vfeatures, spatial_shapes, level_start_index, valid_ratios
        )
        
        # 4. 自回归模型
        if input_actions is not None:
            trajectory_output = self.trajectory_autoregressive(
                input_actions,
                unified_outputs['seg_features'], 
                unified_outputs['det_features'], 
                Vfeatures, 
                unified_outputs['point_features'],
                target_actions=target_actions if self.training else None
            )
        else:
            # 如果没有输入动作序列，使用零序列
            B = images.shape[0]
            zero_actions = torch.zeros(B, self.trajectory_autoregressive.seq_len, 
                                     self.trajectory_autoregressive.action_dim, 
                                     device=images.device)
            trajectory_output = self.trajectory_autoregressive(
                zero_actions,
                unified_outputs['seg_features'], 
                unified_outputs['det_features'], 
                Vfeatures, 
                unified_outputs['point_features']
            )
        
        # 整合所有输出
        outputs = {
            'seg_logits': unified_outputs['seg_logits'],
            'class_logits': unified_outputs['class_logits'],
            'bbox_coords': unified_outputs['bbox_coords'],
            'point_coords': unified_outputs['point_coords'],
            'trajectory_actions': trajectory_output,  # [B, seq_len, action_dim]
            'det_features': unified_outputs['det_features']
        }
        return outputs
    
    def compute_loss(self, outputs, targets):
        """
        计算多任务损失
        Args:
            outputs: 模型输出
            targets: 目标标签字典，包含：
                - seg_masks: [B, H, W] 语义分割掩码
                - det_targets: 检测目标列表
                - point_coords: [B, num_points, 3] 抓取点坐标
                - actions: [B, seq_len, action_dim] 动作序列
                - noise: [B, seq_len, action_dim] 扩散噪声（训练时）
        """
        losses = {}
        
        # 1. 语义分割损失（像素级）
        if 'seg_masks' in targets:
            # 调整seg_logits的空间维度顺序：从(W, H)转换为(H, W)
            seg_logits_adjusted = outputs['seg_logits'].permute(0, 1, 3, 2)  # [B, C, H, W]
            
            seg_loss = F.cross_entropy(
                seg_logits_adjusted, 
                targets['seg_masks'].long()
            )
            losses['seg_loss'] = seg_loss
        
        # 2. 目标检测损失（DETR风格）
        if 'det_targets' in targets:
            det_outputs = {
                'pred_logits': outputs['class_logits'],
                'pred_boxes': outputs['bbox_coords']
            }
            det_losses = self.criterion(det_outputs, targets['det_targets'])
            losses.update(det_losses)
        
        # 3. 抓取点损失
        if 'point_coords' in targets:
            point_loss = F.l1_loss(
                outputs['point_coords'], 
                targets['point_coords']
            )
            losses['point_loss'] = point_loss
        
        # 4. 生成损失
        if 'actions' in targets:
            trajectory_loss = F.mse_loss(
                outputs['trajectory'], 
                targets['actions']
            )
            losses['trajectory_loss'] = trajectory_loss
        
        # 总损失
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses

# 示例使用
if __name__ == "__main__":
    # 创建统一Transformer模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobotUniADModel(
        num_seg_classes=1,
        num_det_classes=3,
        num_seg_queries=50,
        num_det_queries=50,
        num_points=5,
        action_dim=7,
        seq_len=30,
        image_size=(640, 480)
    )
    model = model.to(device)
    print(model)
    # 测试前向传播
    batch_size = 2
    timesteps = 2
    images = torch.randn(batch_size, timesteps, 3, 640, 480)
    images = images.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        
        print("\n统一Transformer模型输出形状:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape}")
        
        # 验证分割输出尺寸
        print(f"\n输入图像尺寸: {images.shape[2:]}")
        print(f"分割输出尺寸: {outputs['seg_logits'].shape[2:]}")
        print(f"尺寸匹配: {outputs['seg_logits'].shape[2:] == images.shape[2:]}")

    # # 6. 马尔科夫链正则化器（新增）
    # if use_markov_regularizer:
    #     from MarkovRegularizer import MarkovChainRegularizer
    #     self.markov_regularizer = MarkovChainRegularizer(
    #         action_dim=action_dim,
    #         seq_len=seq_len,
    #         key_frame_indices=[4, 9, 14, 19, 24, 29],  # 可以根据需要调整
    #         state_dim=32,
    #         num_discrete_states=16,
    #         hidden_dim=128
    #     )
    #     self.use_markov_regularizer = True
    #     self.markov_weight = markov_weight
    # else:
    #     self.use_markov_regularizer = False
    
    # def compute_loss(self, outputs, targets):
    #     """
    #     计算多任务损失（修改版本）
    #     """
    #     losses = {}
        
    #     # 2. 目标检测损失（DETR风格）
    #     if 'det_targets' in targets:
    #         det_outputs = {
    #             'pred_logits': outputs['class_logits'],
    #             'pred_boxes': outputs['bbox_coords']
    #         }
    #         det_losses = self.criterion(det_outputs, targets['det_targets'])
    #         losses.update(det_losses)
        
    #     # 3. 抓取点损失
    #     if 'point_coords' in targets:
    #         point_loss = F.l1_loss(
    #             outputs['point_coords'], 
    #             targets['point_coords']
    #         )
    #         losses['point_loss'] = point_loss
        
    #     # 4. 生成损失
    #     if 'noise' in targets:
    #         trajectory_loss = F.mse_loss(
    #             outputs['trajectory'], 
    #             targets['trajectory']
    #         )
    #         losses['trajectory_loss'] = trajectory_loss
        
    #     # 总损失
    #     total_loss = sum(losses.values())
    #     losses['total_loss'] = total_loss
        
    #     return losses
