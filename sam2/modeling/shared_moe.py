import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertLayer(nn.Module):
    """ 标准专家层 (1x1 Conv) """

    def __init__(self, dim, hidden_ratio=4):
        super().__init__()
        hidden_dim = int(dim * hidden_ratio)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)


class TopKRouter(nn.Module):
    def __init__(self, dim, num_experts, active_experts):
        super().__init__()
        self.gate = nn.Conv2d(dim, num_experts, kernel_size=1)
        self.active_experts = active_experts

    def forward(self, x):
        logits = self.gate(x)
        scores = F.softmax(logits, dim=1)
        topk_probs, topk_indices = torch.topk(scores, self.active_experts, dim=1)
        topk_probs = topk_probs / topk_probs.sum(dim=1, keepdim=True)  # 归一化
        return topk_probs, topk_indices, scores


class SharedSparseMoEBlock(nn.Module):
    """
    【新设计】包含 1 个共享专家 + N 个路由专家
    Output = Shared(x) + Sum(Router(x) * Experts(x))
    """

    def __init__(self, dim, num_experts=8, active_experts=3):
        super().__init__()
        self.num_experts = num_experts
        self.active_experts = active_experts

        # 1. 共享专家 (Shared Expert) - 永远激活
        self.shared_expert = ExpertLayer(dim)

        # 2. 路由专家组 (Routed Experts)
        self.router = TopKRouter(dim, num_experts, active_experts)
        self.experts = nn.ModuleList([ExpertLayer(dim) for _ in range(num_experts)])

    def forward(self, x):
        B, C, H, W = x.shape

        # --- A. 共享专家路径 ---
        shared_out = self.shared_expert(x)

        # --- B. 路由专家路径 ---
        routing_weights, selected_indices, all_probs = self.router(x)
        moe_out = torch.zeros_like(x)

        expert_mask = torch.zeros(B, self.num_experts, H, W, device=x.device, dtype=x.dtype)
        expert_mask.scatter_(1, selected_indices, routing_weights)

        for i, expert in enumerate(self.experts):
            weight = expert_mask[:, i:i + 1, :, :]
            if weight.sum() > 0:
                moe_out += weight * expert(x)

        # --- C. 辅助损失计算 ---
        mean_prob = all_probs.mean(dim=(0, 2, 3))
        one_hot = torch.zeros_like(all_probs)
        one_hot.scatter_(1, selected_indices, 1.0)
        mean_load = one_hot.mean(dim=(0, 2, 3))
        aux_loss = self.num_experts * torch.sum(mean_prob * mean_load)

        # 最终输出 = 原输入(残差) + 共享专家输出 + 路由专家输出
        return x + shared_out + moe_out, aux_loss