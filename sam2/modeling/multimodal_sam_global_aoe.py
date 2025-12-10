import torch
import torch.nn as nn
import torch.nn.functional as F
from .fusion import SimpleFusionModule
from .segformer_head import SegFormerHead
# 引用纯净版 Block
from .global_guided_aoe import GlobalGuidedAoEBlock


class MultiModalSegFormerGlobalAoE(nn.Module):
    def __init__(self, sam_model, feature_channels, num_classes=9):
        super().__init__()
        self.sam_model = sam_model

        for param in self.sam_model.parameters(): param.requires_grad = False
        self.fusion_layers = nn.ModuleList([SimpleFusionModule(ch) for ch in feature_channels])
        for param in self.fusion_layers.parameters(): param.requires_grad = False
        self.segformer_head = SegFormerHead(in_channels=feature_channels, num_classes=num_classes)
        for param in self.segformer_head.parameters(): param.requires_grad = False

        # === 8专家，激活3个，无共享 ===
        self.rgb_moe_layers = nn.ModuleList([
            GlobalGuidedAoEBlock(dim=ch, num_experts=8, active_experts=3)
            for ch in feature_channels
        ])

        self.ir_moe_layers = nn.ModuleList([
            GlobalGuidedAoEBlock(dim=ch, num_experts=8, active_experts=3)
            for ch in feature_channels
        ])

        for param in self.rgb_moe_layers.parameters(): param.requires_grad = True
        for param in self.ir_moe_layers.parameters(): param.requires_grad = True

    def extract_features(self, image):
        backbone = self.sam_model.image_encoder.trunk
        out = backbone(image)
        if isinstance(out, (list, tuple)):
            features = list(out)
        elif isinstance(out, dict):
            features = [out[k] for k in sorted(out.keys())]
        return features[:4]

    def forward(self, image_rgb, image_ir):
        with torch.no_grad():
            raw_rgb = self.extract_features(image_rgb)
            raw_ir = self.extract_features(image_ir)

        enhanced_rgb = []
        enhanced_ir = []
        total_aux_loss = 0.0

        for i in range(4):
            out_rgb, aux_r = self.rgb_moe_layers[i](raw_rgb[i])
            enhanced_rgb.append(out_rgb)
            total_aux_loss += aux_r

            out_ir, aux_i = self.ir_moe_layers[i](raw_ir[i])
            enhanced_ir.append(out_ir)
            total_aux_loss += aux_i

        fused_features = []
        for i in range(4):
            f = self.fusion_layers[i](enhanced_rgb[i], enhanced_ir[i])
            fused_features.append(f)

        logits = self.segformer_head(fused_features)
        logits = torch.nn.functional.interpolate(logits, size=image_rgb.shape[2:], mode='bilinear', align_corners=False)

        return logits, total_aux_loss