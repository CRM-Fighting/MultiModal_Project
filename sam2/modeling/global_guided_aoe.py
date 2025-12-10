import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalGuidedAoERouter(nn.Module):
    """
    【纯净版】Global Guided AoE Router
    - 保留核心：Local Linear Scorer + Global MLP Bias
    - 移除：Tanh限制、训练噪声
    - 修复：维度匹配、显存优化
    """

    def __init__(self, d_model, num_experts, d_low, topk=3, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.d_low = d_low
        self.topk = topk
        self.d_model = d_model

        # 1. AoE 特征生成
        self.w_down = nn.Linear(d_model, num_experts * d_low, bias=False)

        # 2. 交互感知模块
        # 【关键修复】使用 3D 维度 [1, N, d_low] 防止报错
        self.expert_pos_embed = nn.Parameter(torch.randn(1, num_experts, d_low))
        self.global_proj = nn.Linear(d_model, d_low)

        self.interaction_attn = nn.MultiheadAttention(
            embed_dim=d_low, num_heads=4, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_low)

        # 3. 双路门控 (Dual-Path)
        # 路径 A: 局部打分 (Linear)
        self.local_scorer = nn.Linear(d_low, 1)

        # 路径 B: 全局偏置
        self.global_gate_mlp = nn.Sequential(
            nn.Linear(d_low, d_low * 2),
            nn.GELU(),
            nn.Linear(d_low * 2, num_experts)
        )

        # 4. 执行
        self.act = nn.GELU()
        self.w_up = nn.Parameter(torch.randn(num_experts, d_low, d_model))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.w_down.weight, std=0.02)
        nn.init.normal_(self.expert_pos_embed, std=0.02)
        nn.init.normal_(self.w_up, std=0.02)
        nn.init.normal_(self.local_scorer.weight, std=0.02)

    def _sparse_w_up(self, feats, indices):
        selected_w_up = self.w_up[indices]
        output = torch.matmul(feats.unsqueeze(2), selected_w_up).squeeze(2)
        return output

    def forward(self, x):
        batch_size, num_tokens, d_model = x.shape
        # 使用 reshape 避免显存不连续报错
        x_flat = x.reshape(-1, d_model)

        # Step 1: AoE 生成
        all_expert_feats_flat = self.w_down(x_flat)
        expert_feats = all_expert_feats_flat.view(-1, self.num_experts, self.d_low)

        # Step 2: 交互
        global_ctx_raw = x.mean(dim=1)
        global_ctx = self.global_proj(global_ctx_raw)

        global_ctx_expanded = global_ctx.unsqueeze(1).expand(batch_size, num_tokens, -1).reshape(-1, 1, self.d_low)

        # [B*T, N, D] + [1, N, D] -> [B*T, N, D] (正常广播)
        expert_feats_with_pos = expert_feats + self.expert_pos_embed

        interaction_seq = torch.cat([global_ctx_expanded, expert_feats_with_pos], dim=1)

        interacted_seq, _ = self.interaction_attn(interaction_seq, interaction_seq, interaction_seq)
        interacted_seq = self.norm(interacted_seq + interaction_seq)
        refined_feats = interacted_seq[:, 1:, :]

        # Step 3: 决策 (纯净版)
        local_logits = self.local_scorer(refined_feats).squeeze(-1)

        global_bias = self.global_gate_mlp(global_ctx)
        # 【回归本源】不加 Tanh 限制，直接使用
        global_bias_expanded = global_bias.unsqueeze(1).expand(-1, num_tokens, -1).reshape(-1, self.num_experts)

        final_logits = local_logits + global_bias_expanded

        # 【回归本源】不加训练噪声
        router_probs = F.softmax(final_logits, dim=-1)

        # Step 4: 执行
        topk_weights, topk_indices = torch.topk(router_probs, self.topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, self.d_low)
        selected_feats = torch.gather(expert_feats, 1, gather_indices)

        activated_feats = self.act(selected_feats)
        expert_outputs = self._sparse_w_up(activated_feats, topk_indices)

        weighted_outputs = (expert_outputs * topk_weights.unsqueeze(-1)).sum(dim=1)

        # Aux Loss (标准版)
        mean_prob = router_probs.mean(dim=0)
        one_hot = torch.zeros_like(router_probs).scatter_(1, topk_indices, 1.0)
        mean_load = one_hot.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(mean_prob * mean_load)

        return weighted_outputs.reshape(batch_size, num_tokens, d_model), aux_loss


class GlobalGuidedAoEBlock(nn.Module):
    """
    【纯净版 Block】
    移除 Shared Expert，只有 Global Guided AoE + 残差连接
    """

    def __init__(self, dim, num_experts=8, active_experts=3):
        super().__init__()

        # 路由专家
        # d_low = dim // 4
        self.moe = GlobalGuidedAoERouter(
            d_model=dim,
            num_experts=num_experts,
            d_low=dim // 4,
            topk=active_experts
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # 变形
        x_tokens = x.flatten(2).transpose(1, 2)

        # 进 MoE
        moe_out, aux_loss = self.moe(x_tokens)

        # 还原
        moe_out = moe_out.transpose(1, 2).reshape(B, C, H, W)

        # 残差连接 (原图 + MoE)
        return x + moe_out, aux_loss