import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GlobalGuidedAoERouter(nn.Module):
    """
    【老师建议版】Global Guided AoE Router
    配置：8 Experts, Top-3 Active
    逻辑：
    1. Interaction: 9人圆桌会议 (Self-Attention) 增强特征
    2. Routing: Global Query 点积 Refined Expert Key (内容感知路由)
    """

    def __init__(self, d_model, num_experts, d_low, topk=3, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.d_low = d_low
        self.topk = topk
        self.d_model = d_model

        # 1. 特征生成 (AoE)
        self.w_down = nn.Linear(d_model, num_experts * d_low, bias=False)

        # 2. 交互感知 (Self-Attention)
        self.expert_pos_embed = nn.Parameter(torch.randn(1, num_experts, d_low))
        self.global_proj = nn.Linear(d_model, d_low)

        self.interaction_attn = nn.MultiheadAttention(
            embed_dim=d_low, num_heads=4, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_low)

        # 3. 动态路由投影层 (Router Projections)
        # Global -> Query
        self.router_query_proj = nn.Sequential(
            nn.Linear(d_low, d_low),
            nn.LayerNorm(d_low),
            nn.GELU(),
            nn.Linear(d_low, d_low)
        )
        # Expert -> Key
        self.router_key_proj = nn.Linear(d_low, d_low)

        # 4. 执行层
        self.act = nn.GELU()
        self.w_up = nn.Parameter(torch.randn(num_experts, d_low, d_model))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.w_down.weight, std=0.02)
        nn.init.normal_(self.expert_pos_embed, std=0.02)
        nn.init.normal_(self.w_up, std=0.02)
        nn.init.xavier_uniform_(self.router_key_proj.weight)
        for m in self.router_query_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def _sparse_w_up(self, feats, indices):
        selected_w_up = self.w_up[indices]
        output = torch.matmul(feats.unsqueeze(2), selected_w_up).squeeze(2)
        return output

    def forward(self, x):
        batch_size, num_tokens, d_model = x.shape
        x_flat = x.reshape(-1, d_model)

        # --- Step 1: Generate ---
        all_expert_feats_flat = self.w_down(x_flat)
        expert_feats = all_expert_feats_flat.view(-1, self.num_experts, self.d_low)

        # --- Step 2: Interaction ---
        # 准备全局上下文
        global_ctx_raw = x.mean(dim=1)
        # [B*T, 1, D]
        global_ctx = self.global_proj(global_ctx_raw).unsqueeze(1).expand(batch_size, num_tokens, -1).reshape(-1, 1,
                                                                                                              self.d_low)

        # 准备专家
        expert_feats_with_pos = expert_feats + self.expert_pos_embed

        # 交互
        interaction_seq = torch.cat([global_ctx, expert_feats_with_pos], dim=1)
        interacted_seq, _ = self.interaction_attn(interaction_seq, interaction_seq, interaction_seq)
        interacted_seq = self.norm(interacted_seq + interaction_seq)

        # 提取增强后的专家特征
        refined_experts = interacted_seq[:, 1:, :]

        # --- Step 3: Routing (Dot-Product) ---
        # Query: 来自原始全局上下文 (稳定锚点)
        router_query = self.router_query_proj(global_ctx)  # [B*T, 1, D]

        # Key: 来自交互后的专家特征 (动态能力)
        router_key = self.router_key_proj(refined_experts)  # [B*T, N, D]

        # Dot Product: [B*T, 1, D] @ [B*T, D, N] -> [B*T, 1, N]
        routing_logits = torch.matmul(router_query, router_key.transpose(1, 2))
        routing_logits = routing_logits.squeeze(1) / math.sqrt(self.d_low)

        # Softmax
        router_probs = F.softmax(routing_logits, dim=-1)

        # --- Step 4: Execution ---
        topk_weights, topk_indices = torch.topk(router_probs, self.topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, self.d_low)
        selected_feats = torch.gather(expert_feats, 1, gather_indices)

        activated_feats = self.act(selected_feats)
        expert_outputs = self._sparse_w_up(activated_feats, topk_indices)

        # 乘权重后相加
        weighted_outputs = (expert_outputs * topk_weights.unsqueeze(-1)).sum(dim=1)

        # Aux Loss
        mean_prob = router_probs.mean(dim=0)
        one_hot = torch.zeros_like(router_probs).scatter_(1, topk_indices, 1.0)
        mean_load = one_hot.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(mean_prob * mean_load)

        return weighted_outputs.reshape(batch_size, num_tokens, d_model), aux_loss


class GlobalGuidedAoEBlock(nn.Module):
    """ Block Wrapper """

    def __init__(self, dim, num_experts=8, active_experts=3):
        super().__init__()
        # 8选3配置
        self.moe = GlobalGuidedAoERouter(
            d_model=dim,
            num_experts=num_experts,
            d_low=dim // 4,
            topk=active_experts
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_tokens = x.flatten(2).transpose(1, 2)
        moe_out, aux_loss = self.moe(x_tokens)
        moe_out = moe_out.transpose(1, 2).reshape(B, C, H, W)
        return x + moe_out, aux_loss