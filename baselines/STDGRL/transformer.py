import math

import torch.nn as nn
import torch
from typing import List, Optional


def generate_spatial_attention_mask(
        agents_per_scene: List[int],
        device: Optional[torch.device] = None
) -> torch.BoolTensor:
    """
    生成块对角空间注意力掩码，用于限制不同场景的智能体之间的交互

    Args:
        agents_per_scene: 每个场景包含的智能体数量列表，例如 [2, 3] 表示两个场景分别有2、3个智能体
        device: 输出张量的设备位置 (e.g., 'cuda', 'cpu')

    Returns:
        mask: 布尔类型掩码张量，形状为 [total_agents, total_agents]
              - True 表示允许注意力交互
              - False 表示屏蔽注意力交互
    Example:
        >>> generate_spatial_attention_mask([2, 3])
        tensor([[ True,  True, False, False, False],
                [ True,  True, False, False, False],
                [False, False,  True,  True,  True],
                [False, False,  True,  True,  True],
                [False, False,  True,  True,  True]])
    """
    # 计算累积智能体索引
    cumulative_agents = torch.cumsum(
        torch.tensor([0] + agents_per_scene, dtype=torch.long),
        dim=0
    )
    total_agents = cumulative_agents[-1].item()

    # 初始化全屏蔽掩码（默认不允许交互）
    mask = torch.zeros(
        (total_agents, total_agents),
        dtype=torch.bool,
        device=device
    )

    # 为每个场景填充允许交互的区域
    for scene_start, scene_end in zip(cumulative_agents[:-1], cumulative_agents[1:]):
        mask[scene_start:scene_end, scene_start:scene_end] = True
    return mask


def generate_spatial_attention_mask_v2(
        agents_per_scene: List[int],
        device: Optional[torch.device] = None
) -> torch.BoolTensor:
    # 生成场景索引标签
    scene_labels = torch.cat([
        torch.full((n,), i, dtype=torch.long)
        for i, n in enumerate(agents_per_scene)
    ]).to(device)

    # 利用广播比较生成掩码
    return scene_labels[None, :] == scene_labels[:, None]


def generate_temporal_attention_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.BoolTensor:
    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    Notes:
        最官方的实现方式：
        Definition >>> attention = nn.MultiheadAttention(d_model, nhead)
        Usage >>> attention(query=feat, key=feat, value=feat, attn_mask=~mask)

    """

    def __init__(self, model_dim, num_heads=4):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value, attn_mask=None):
        # Q    (N, T, model_dim)
        # K, V (N, T, model_dim)
        N, T, D = query.shape

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * N, length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(-1, -2)  # (num_heads * N,  head_dim, src_length)

        attn_score = (query @ key) / self.head_dim**0.5  # (num_heads * N, T, T)

        if attn_mask is not None:
            attn_score.masked_fill_(~attn_mask, -torch.inf)  # fill -inf in-place for mask=False

        attn_score = torch.softmax(attn_score, dim=-1)  # 沿着最后一个维度聚合数据
        out = attn_score @ value  # (num_heads * N, T, head_dim)
        out = torch.cat(torch.split(out, N, dim=0), dim=-1)  # (batch_size, T, head_dim * num_heads = model_dim)

        out = self.out_proj(out)  # 该操作也是挺重要的，可以理解为下游任务通过该MLP对多头信息进行选择或融合

        return out, attn_score


class SelfAttentionLayer(nn.Module):
    def __init__(self, model_dim, feed_forward_dim=2048, num_heads=4, dropout=0):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_dim=-2, attn_mask=None):
        """标准的transformer是沿着倒数第二个维度计算attention score, 当区分空间和时间注意力时, 需要根据
        数据的具体结构, 指定attn_dim, 在实际处理时, 会将该维度与attn_dim 交换。
        """
        x = x.transpose(attn_dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out, _ = self.attn(x, x, x, attn_mask)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(attn_dim, -2)
        return out


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


if __name__ == '__main__':
    T = 20
    mask = torch.ones((T, T), dtype=torch.bool).tril()
    model = SelfAttentionLayer(64, feed_forward_dim=1024, num_heads=4, dropout=0)
    x = torch.randn((6, T, 64))  # [N, T, D]
    out = model(x, attn_dim=1)
    print(out.shape)
    # print(generate_spatial_attention_mask_v2([2, 3, 4, 1]))
