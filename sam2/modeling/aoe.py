import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertLayer(nn.Module):
    """ 标准专家层 (用于共享专家) """

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


class AoERouter(nn.Module):
    """ AoE Router (核心路由逻辑) """

    def __init__(self, d_model, num_experts, d_low, topk=2):
        super().__init__()
        self.num_experts = num_experts
        self.d_low = d_low
        self.topk = topk
        self.d_model = d_model

        # 1. 专家生成器
        self.w_down = nn.Linear(d_model, num_experts * d_low, bias=False)

        # 2. 门控网络
        self.router = nn.Linear(d_low, 1, bias=False)

        # 3. 专家执行 (Up Projection)
        self.w_up = nn.Parameter(torch.randn(num_experts, d_low, d_model))
        self.act = nn.GELU()

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.w_down.weight, std=0.02)
        nn.init.normal_(self.w_up, std=0.02)
        nn.init.normal_(self.router.weight, std=0.02)

    def _sparse_w_up(self, feats, indices):
        selected_w_up = self.w_up[indices]
        output = torch.matmul(feats.unsqueeze(2), selected_w_up).squeeze(2)
        return output

    def forward(self, x):
        """ x: [Batch, Tokens, d_model] """
        batch_size, num_tokens, d_model = x.shape
        # 【关键修复】使用 reshape 处理非连续内存
        x_flat = x.reshape(-1, d_model)

        # Step 1: Generate
        all_expert_feats = self.w_down(x_flat).view(-1, self.num_experts, self.d_low)

        # Step 2: Routing
        logits = self.router(all_expert_feats).squeeze(-1)

        # Add Noise (Training only)
        if self.training:
            logits = logits + torch.randn_like(logits) * (1.0 / self.num_experts)

        router_probs = F.softmax(logits, dim=-1)

        # Top-K
        topk_weights, topk_indices = torch.topk(router_probs, self.topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Step 3: Execution
        gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, self.d_low)
        selected_feats = torch.gather(all_expert_feats, 1, gather_indices)

        activated_feats = self.act(selected_feats)
        expert_outputs = self._sparse_w_up(activated_feats, topk_indices)

        weighted_outputs = (expert_outputs * topk_weights.unsqueeze(-1)).sum(dim=1)

        # Step 4: Aux Loss
        mean_prob = router_probs.mean(dim=0)
        one_hot = torch.zeros_like(router_probs).scatter_(1, topk_indices, 1.0)
        mean_load = one_hot.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(mean_prob * mean_load)

        # 【关键修复】输出使用 reshape
        return weighted_outputs.reshape(batch_size, num_tokens, d_model), aux_loss


class AoEBlock(nn.Module):
    """
    AoE Block (集成共享专家)
    注意：我保留了 AoEBlock 这个类名，但是加入了 Shared Expert 逻辑
    这样你不用改 import 语句。
    """

    def __init__(self, dim, num_experts=8, active_experts=2):
        super().__init__()

        # 1. 共享专家 (Shared Expert)
        self.shared_expert = ExpertLayer(dim)

        # 2. AoE 路由专家
        self.aoe = AoERouter(
            d_model=dim,
            num_experts=num_experts,
            d_low=dim // 4,
            topk=active_experts
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # A. 共享路径 (Shared Path)
        shared_out = self.shared_expert(x)

        # B. AoE 路径 (Routing Path)
        # 【关键修复】transpose 导致不连续，必须配合 reshape 使用
        x_tokens = x.flatten(2).transpose(1, 2)
        aoe_out, aux_loss = self.aoe(x_tokens)
        aoe_out = aoe_out.transpose(1, 2).reshape(B, C, H, W)

        # C. 融合: 原图(残差) + 共享专家 + AoE专家
        return x + shared_out + aoe_out, aux_loss