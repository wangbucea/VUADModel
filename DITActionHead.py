import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, List

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

class TimestepEmbedder(nn.Module):
    """时间步嵌入器"""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class LabelEmbedder(nn.Module):
    """条件标签嵌入器"""
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            approx_gelu(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Self-attention with adaLN
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_output
        
        # MLP with adaLN
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_output = self.mlp(x_norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_output
        
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, action_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, action_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x

class DITActionHead(nn.Module):
    """
    Diffusion Transformer for Action Generation
    """
    def __init__(
        self,
        action_dim=7,
        seq_len=30,
        hidden_size=512,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        feature_dim=256,
        num_classes=1000,  # 用于条件生成的类别数
        learn_sigma=True,
        dropout_prob=0.1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.learn_sigma = learn_sigma
        self.out_channels = action_dim * 2 if learn_sigma else action_dim
        self.num_heads = num_heads

        # 输入嵌入层
        self.x_embedder = nn.Linear(action_dim, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, dropout_prob)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_size), requires_grad=False)
        
        # 条件特征编码器
        self.seg_encoder = nn.Linear(feature_dim, hidden_size)
        self.det_encoder = nn.Linear(feature_dim, hidden_size)
        self.visual_encoder = nn.Linear(feature_dim, hidden_size)
        self.point_encoder = nn.Linear(feature_dim, hidden_size)
        self.state_encoder = nn.Linear(action_dim, hidden_size)
        
        # 条件特征融合
        self.condition_fusion = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        self.condition_norm = nn.LayerNorm(hidden_size)
        
        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        
        # 最终输出层
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize position embeddings
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.seq_len)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def encode_conditions(self, state, seg_features, det_features, visual_features, point_features):
        """
        编码多模态条件特征
        """
        B = state.shape[0]
        
        # 编码各种条件特征
        state_encoded = self.state_encoder(state)  # [B, hidden_size]
        seg_encoded = self.seg_encoder(seg_features.mean(dim=1))  # [B, hidden_size]
        det_encoded = self.det_encoder(det_features.mean(dim=1))  # [B, hidden_size]
        visual_encoded = self.visual_encoder(visual_features.mean(dim=1))  # [B, hidden_size]
        point_encoded = self.point_encoder(point_features.mean(dim=1))  # [B, hidden_size]
        
        # 堆叠所有条件特征
        condition_features = torch.stack([
            state_encoded, seg_encoded, det_encoded, visual_encoded, point_encoded
        ], dim=1)  # [B, 5, hidden_size]
        
        # 使用注意力机制融合条件特征
        fused_conditions, _ = self.condition_fusion(
            condition_features, condition_features, condition_features
        )
        fused_conditions = self.condition_norm(condition_features + fused_conditions)
        
        # 平均池化得到全局条件向量
        global_condition = fused_conditions.mean(dim=1)  # [B, hidden_size]
        
        return global_condition

    def forward(self, x, t, state, seg_features, det_features, visual_features, point_features, y=None):
        """
        Forward pass of DiT.
        x: (B, seq_len, action_dim) tensor of noisy actions
        t: (B,) tensor of diffusion timesteps
        state: (B, action_dim) current robot state
        seg_features: (B, num_seg_queries, feature_dim) segmentation features
        det_features: (B, num_det_queries, feature_dim) detection features  
        visual_features: (B, N, feature_dim) visual features
        point_features: (B, num_points, feature_dim) point features
        y: (B,) tensor of class labels (optional)
        """
        B, seq_len, _ = x.shape
        
        # 输入嵌入
        x = self.x_embedder(x) + self.pos_embed  # (B, seq_len, hidden_size)
        
        # 时间步嵌入
        t = self.t_embedder(t)  # (B, hidden_size)
        
        # 条件嵌入
        condition_emb = self.encode_conditions(
            state, seg_features, det_features, visual_features, point_features
        )  # (B, hidden_size)
        
        # 类别嵌入（如果提供）
        if y is not None:
            y = self.y_embedder(y, self.training)  # (B, hidden_size)
            c = t + condition_emb + y
        else:
            c = t + condition_emb
        
        # DiT blocks
        for block in self.blocks:
            x = block(x, c)  # (B, seq_len, hidden_size)
        
        # 最终输出
        x = self.final_layer(x, c)  # (B, seq_len, out_channels)
        
        return x

    def forward_with_cfg(self, x, t, state, seg_features, det_features, visual_features, point_features, y, cfg_scale):
        """
        Forward pass with classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[:len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, state, seg_features, det_features, visual_features, point_features, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.action_dim], model_out[:, self.action_dim:]
        eps, rest = model_out[:, :, :self.action_dim], model_out[:, :, self.action_dim:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=-1)

def get_1d_sincos_pos_embed(embed_dim, length):
    """
    Get 1D sine-cosine positional embedding.
    """
    pos = np.arange(0, length, dtype=np.float32)
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

# 扩散调度器
class DDPMScheduler(nn.Module):
    """
    DDPM噪声调度器
    """
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        
        # 线性beta调度
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        
        # 注册为buffer，这样会自动移动到正确的设备
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # 计算用于采样的系数
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
    def add_noise(self, original_samples, noise, timesteps):
        """
        添加噪声到原始样本
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # 广播到正确的形状
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
            
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def step(self, model_output, timestep, sample):
        """
        执行一步去噪
        """
        t = timestep
        
        # 获取系数
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod_prev[t]
        beta_prod_t = 1 - alpha_prod_t
        
        # 广播到正确的形状
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
            alpha_prod_t_prev = alpha_prod_t_prev.unsqueeze(-1)
            beta_prod_t = beta_prod_t.unsqueeze(-1)
        
        # 计算预测的原始样本
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        
        # 计算前一步的样本
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        return prev_sample
