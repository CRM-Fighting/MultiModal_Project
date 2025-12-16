import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.modeling.serial_modeling import SerialSegModel


class MultiTaskSerialModel(SerialSegModel):
    """
    多任务模型：Shared Serial MoE + SAM2 Aux (Only Stage 4 Supervision)
    遵照指示：只对最后一层特征做 SAM 不确定性约束。
    """

    def __init__(self, base_sam, moe_class, num_classes=9):
        super().__init__(base_sam, moe_class, num_classes)

        # 冻结 SAM 解码器组件
        for param in self.backbone.base_sam.sam_prompt_encoder.parameters(): param.requires_grad = False
        for param in self.backbone.base_sam.sam_mask_decoder.parameters(): param.requires_grad = False

        # SAM2 适配层 (只保留 Stage 4: 768 -> 256)
        self.sam_proj_s4 = nn.Conv2d(768, 256, kernel_size=1)

    def get_prompt(self, gt):
        B, H, W = gt.shape
        coords, labels = [], []
        for i in range(B):
            y, x = torch.where(gt[i] > 0)
            if len(y) > 0:
                idx = torch.randint(len(y), (1,)).item()
                coords.append([x[idx].item(), y[idx].item()])
                labels.append(1)
            else:
                coords.append([W // 2, H // 2])
                labels.append(0)
        return torch.tensor(coords, device=gt.device).unsqueeze(1).float(), \
            torch.tensor(labels, device=gt.device).unsqueeze(1)

    def run_sam_head(self, feat, gt, proj):
        sam_feat = proj(feat)
        pt_c, pt_l = self.get_prompt(gt)
        sparse, dense = self.backbone.base_sam.sam_prompt_encoder(points=(pt_c, pt_l), boxes=None, masks=None)

        # PE 插值 (Stage 4 通常很小，需要对齐)
        pe = self.backbone.base_sam.sam_prompt_encoder.get_dense_pe()
        if pe.shape[-2:] != sam_feat.shape[-2:]:
            pe = F.interpolate(pe, size=sam_feat.shape[-2:], mode='bilinear')

        low_res, _ = self.backbone.base_sam.sam_mask_decoder(
            image_embeddings=sam_feat, image_pe=pe,
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=False
        )
        return low_res

    def forward(self, vis, ir, gt_semantic=None):
        # 1. 基础分割流 (Shared MoE 串行提取)
        feats_rgb, feats_ir, moe_loss = self.backbone(vis, ir)

        fused = [self.fusion_layers[i](feats_rgb[i], feats_ir[i]) for i in range(4)]
        seg_logits = self.segformer_head(fused)
        seg_logits = F.interpolate(seg_logits, size=vis.shape[2:], mode='bilinear')

        # 2. SAM 辅助流 (只算 Stage 4)
        sam_preds = {}
        if gt_semantic is not None:
            H, W = vis.shape[2:]

            # RGB Stage 4
            # feats_rgb[-1] 就是 Stage 4 的输出 (768通道)
            out_r = self.run_sam_head(feats_rgb[-1], gt_semantic, self.sam_proj_s4)
            sam_preds['rgb_s4'] = F.interpolate(out_r, size=(H, W), mode='bilinear')

            # IR Stage 4
            out_i = self.run_sam_head(feats_ir[-1], gt_semantic, self.sam_proj_s4)
            sam_preds['ir_s4'] = F.interpolate(out_i, size=(H, W), mode='bilinear')

        return seg_logits, sam_preds, moe_loss