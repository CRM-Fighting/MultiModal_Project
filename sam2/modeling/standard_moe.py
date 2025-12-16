import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertLayer(nn.Module):
    """ 标准专家层 (与之前保持一致，确保公平对比) """

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


class StandardRouter(nn.Module):
    """
    【Baseline】Standard Noisy Top-K Router
    逻辑：
    1. Gating: Logits = Linear(x) + Noise
    2. Routing: Top-K Softmax
    """

    def __init__(self, d_model, num_experts, topk=3):
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.d_model = d_model

        # 简单的线性门控层
        self.gate = nn.Linear(d_model, num_experts)

        # 专家网络列表
        # 注意：这里专家是 Conv2d，需要把 Token 变回 Image 形状处理，或者用 Linear 模拟
        # 为了和您的 AoE 专家结构保持完全一致（1x1 Conv），这里我们在 forward 里做一下变换
        self.experts = nn.ModuleList([
            ExpertLayer(d_model) for _ in range(num_experts)
        ])

    def forward(self, x):
        """ x: [Batch, Tokens, d_model] """
        batch_size, num_tokens, d_model = x.shape
        x_flat = x.reshape(-1, d_model)

        # 1. 计算路由得分
        # [B*T, N]
        clean_logits = self.gate(x_flat)

        # 加噪声 (Noisy Gating) - 标准 MoE 防坍塌机制
        if self.training:
            noise = torch.randn_like(clean_logits) * (1.0 / self.num_experts)
            logits = clean_logits + noise
        else:
            logits = clean_logits

        router_probs = F.softmax(logits, dim=-1)

        # 2. Top-K 选择
        topk_weights, topk_indices = torch.topk(router_probs, self.topk, dim=-1)
        # 归一化权重
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # 3. 执行专家 (Dispatch & Combine)
        # 这是一个显式的循环实现，虽然慢一点，但逻辑最清晰，适合 Baseline
        # 初始化输出
        final_output = torch.zeros_like(x_flat)

        # 为了复用 Conv2d 专家，我们需要把 flat 的 x 变回 [Total, C, 1, 1] 哪怕是伪维
        # 或者为了效率，我们只计算被选中的。
        # 这里为了代码简单且保持专家结构一致，我们对所有专家进行前向传播，然后Mask (Standard MoE 常见写法)

        # 将输入变回 Conv 格式: [B*T, C, 1, 1]
        x_img = x_flat.unsqueeze(-1).unsqueeze(-1).permute(0, 3, 1, 2)  # [BT, 1, C, 1] -> [BT, C, 1, 1]
        x_img = x_flat.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)  # [C, BT, 1, 1] -> 不对，Conv是 [N, C, H, W]

        # 正确做法：把 x_flat 视为 [B*T, C, 1, 1] 的图片
        x_as_img = x_flat.unsqueeze(-1).unsqueeze(-1)  # [BT, C, 1, 1]

        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # Out: [BT, C, 1, 1] -> [BT, C]
            out = expert(x_as_img).squeeze(-1).squeeze(-1)
            expert_outputs.append(out)

        # Stack: [BT, N, C]
        expert_outputs = torch.stack(expert_outputs, dim=1)

        # 4. 加权融合
        # 根据 topk_indices 选出对应的专家输出
        # gather_indices: [BT, K, C]
        gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, d_model)

        # 从 [BT, N, C] 中选出 [BT, K, C]
        selected_expert_outs = torch.gather(expert_outputs, 1, gather_indices)

        # 加权: [BT, K, C] * [BT, K, 1] -> Sum -> [BT, C]
        weighted_out = (selected_expert_outs * topk_weights.unsqueeze(-1)).sum(dim=1)

        # 5. Aux Loss (Load Balancing)
        mean_prob = router_probs.mean(dim=0)
        one_hot = torch.zeros_like(router_probs).scatter_(1, topk_indices, 1.0)
        mean_load = one_hot.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(mean_prob * mean_load)

        return weighted_out.reshape(batch_size, num_tokens, d_model), aux_loss


class StandardMoEBlock(nn.Module):
    def __init__(self, dim, num_experts=8, active_experts=3):
        super().__init__()
        self.moe = StandardRouter(
            d_model=dim,
            num_experts=num_experts,
            topk=active_experts
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # 变形为序列 [B, T, C]
        x_tokens = x.flatten(2).transpose(1, 2)

        # MoE 前向
        moe_out, aux_loss = self.moe(x_tokens)

        # 变回图片 [B, C, H, W]
        moe_out = moe_out.transpose(1, 2).reshape(B, C, H, W)

        # 残差连接
        return x + moe_out, aux_loss