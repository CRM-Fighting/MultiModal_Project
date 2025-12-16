import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.modeling.fusion import SimpleFusionModule
from sam2.modeling.segformer_head import SegFormerHead


class SerialSAM2Backbone(nn.Module):
    """
    【共享 MoE 版】串行骨干网
    结构：Hiera Stage -> Shared MoE -> Hiera Stage -> ...
    RGB 和 IR 共用同一组 MoE 参数。
    """

    def __init__(self, base_sam, moe_class, feature_channels=[96, 192, 384, 768], num_experts=8, active_experts=3):
        super().__init__()
        self.base_sam = base_sam

        # 1. 冻结原始 SAM2
        for param in self.base_sam.parameters():
            param.requires_grad = False

        # 2. 构建共享的串行 MoE 层 (Shared MoE)
        # 只有一组 MoE，RGB 和 IR 都要过这里
        self.shared_moe_layers = nn.ModuleList([
            moe_class(dim=ch, num_experts=num_experts, active_experts=active_experts)
            for ch in feature_channels
        ])

    def run_serial_stream(self, image, moe_layers):
        """ 单模态串行前向传播 """
        trunk = self.base_sam.image_encoder.trunk
        x = trunk.patch_embed(image)
        if trunk.pos_embed is not None:
            x = x + trunk.pos_embed

        features = []
        total_aux_loss = 0.0

        # 逐阶段穿插执行
        for i, stage in enumerate(trunk.stages):
            # 1. 跑 Hiera Stage (冻结)
            x = stage(x)

            # 2. 跑 MoE (共享参数)
            x_in = x.permute(0, 3, 1, 2)
            # 这里调用的是共享的 MoE
            x_out, aux_loss = moe_layers[i](x_in)
            total_aux_loss += aux_loss

            # 3. 残差连接
            x = x + x_out.permute(0, 2, 3, 1)

            # 4. 收集特征
            features.append(x.permute(0, 3, 1, 2))

        return features, total_aux_loss

    def forward(self, img_rgb, img_ir):
        # RGB 和 IR 都传入 self.shared_moe_layers
        feats_rgb, loss_rgb = self.run_serial_stream(img_rgb, self.shared_moe_layers)
        feats_ir, loss_ir = self.run_serial_stream(img_ir, self.shared_moe_layers)

        # MoE Loss 是两者的总和
        return feats_rgb, feats_ir, (loss_rgb + loss_ir)


class SerialSegModel(nn.Module):
    """ 纯分割模型 (适配共享 MoE Backbone) """

    def __init__(self, base_sam, moe_class, num_classes=9):
        super().__init__()
        # Backbone 内部已经改为了共享 MoE
        self.backbone = SerialSAM2Backbone(base_sam, moe_class)
        channels = [96, 192, 384, 768]

        self.fusion_layers = nn.ModuleList([SimpleFusionModule(ch) for ch in channels])
        self.segformer_head = SegFormerHead(in_channels=channels, num_classes=num_classes)

    def forward(self, vis, ir):
        feats_rgb, feats_ir, moe_loss = self.backbone(vis, ir)

        fused = [self.fusion_layers[i](feats_rgb[i], feats_ir[i]) for i in range(4)]
        logits = self.segformer_head(fused)
        logits = F.interpolate(logits, size=vis.shape[2:], mode='bilinear')

        return logits, moe_loss