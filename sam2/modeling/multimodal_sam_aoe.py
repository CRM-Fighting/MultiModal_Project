import torch.nn as nn
from .fusion import SimpleFusionModule
from .segformer_head import SegFormerHead
# 引用新的 Shared Block
from .aoe import AoEBlock


class MultiModalSegFormerAoE(nn.Module):
    def __init__(self, sam_model, feature_channels, num_classes=9):
        super().__init__()
        self.sam_model = sam_model

        # 冻结基础部分
        for param in self.sam_model.parameters(): param.requires_grad = False
        self.fusion_layers = nn.ModuleList([SimpleFusionModule(ch) for ch in feature_channels])
        for param in self.fusion_layers.parameters(): param.requires_grad = False
        self.segformer_head = SegFormerHead(in_channels=feature_channels, num_classes=num_classes)
        for param in self.segformer_head.parameters(): param.requires_grad = False

        # === Shared AoE 模块 ===
        # 8 选 3 + 1个共享
        self.rgb_moe = nn.ModuleList([
            AoEBlock(dim=ch, num_experts=8, active_experts=3)
            for ch in feature_channels
        ])
        self.ir_moe = nn.ModuleList([
            AoEBlock(dim=ch, num_experts=8, active_experts=3)
            for ch in feature_channels
        ])

        # 激活训练
        for param in self.rgb_moe.parameters(): param.requires_grad = True
        for param in self.ir_moe.parameters(): param.requires_grad = True

    def forward(self, image_rgb, image_ir):
        # (保持不变) ...
        # 简写:
        import torch
        with torch.no_grad():
            features_rgb = self.extract_features(image_rgb)
            features_ir = self.extract_features(image_ir)

        enhanced_rgb, enhanced_ir = [], []
        total_aux = 0.0

        for i in range(4):
            out_r, aux_r = self.rgb_moe[i](features_rgb[i])
            enhanced_rgb.append(out_r)
            total_aux += aux_r

            out_i, aux_i = self.ir_moe[i](features_ir[i])
            enhanced_ir.append(out_i)
            total_aux += aux_i

        fused = []
        for i in range(4):
            fused.append(self.fusion_layers[i](enhanced_rgb[i], enhanced_ir[i]))

        logits = self.segformer_head(fused)
        logits = torch.nn.functional.interpolate(logits, size=image_rgb.shape[2:], mode='bilinear', align_corners=False)

        return logits, total_aux

    def extract_features(self, image):
        backbone = self.sam_model.image_encoder.trunk
        out = backbone(image)
        if isinstance(out, (list, tuple)):
            features = list(out)
        elif isinstance(out, dict):
            features = [out[k] for k in sorted(out.keys())]
        return features[:4]