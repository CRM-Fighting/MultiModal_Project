import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from PIL import Image
import torch.nn.functional as F

from sam2.build_sam import build_sam2
from sam2.modeling.global_guided_aoe import GlobalGuidedAoEBlock
from sam2.modeling.multitask_sam_serial import MultiTaskSerialModel
from utils.custom_losses import BinarySUMLoss, StandardSegLoss

# --- 配置区域 ---
EXP_NAME = "Exp_MultiTask_SUM_Stage4Only"
# 不确定性图路径
UNCERTAINTY_DIRS = {
    'train': "/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/uncertainty_map/train",
    'test': "/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/uncertainty_map/test"
}

TRAIN_DIRS = {
    'vi': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/train/vi',
    'ir': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/train/ir',
    'label': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/train/Segmentation_labels'
}
VAL_DIRS = {
    'vi': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/test/vi',
    'ir': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/test/ir',
    'label': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/test/Segmentation_labels'
}
SAM_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM_CKPT = "../checkpoints/sam2.1_hiera_tiny.pt"
BATCH_SIZE = 2
EPOCHS = 50


# --- Dataset ---
class MSRSDataset(Dataset):
    def __init__(self, dirs, uncertainty_root=None, is_train=True):
        self.vis = dirs['vi'];
        self.ir = dirs['ir'];
        self.lbl = dirs['label']
        self.uncertainty_root = uncertainty_root  # 指定是 train 还是 test 的文件夹
        self.files = sorted([f for f in os.listdir(self.vis) if f.endswith('.png')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        n = self.files[i]
        v = torch.from_numpy(
            np.array(Image.open(os.path.join(self.vis, n)).convert('RGB').resize((640, 480)))).float().permute(2, 0,
                                                                                                               1) / 255.0
        i = torch.from_numpy(
            np.array(Image.open(os.path.join(self.ir, n)).convert('RGB').resize((640, 480)))).float().permute(2, 0,
                                                                                                              1) / 255.0
        l = torch.from_numpy(np.array(Image.open(os.path.join(self.lbl, n)).resize((640, 480), Image.NEAREST))).long()

        # 读取不确定性图
        s = torch.zeros((1, 480, 640))
        if self.uncertainty_root:
            # 假设文件名一致，扩展名为 .npy
            sp = os.path.join(self.uncertainty_root, n.replace('.png', '.npy'))
            if os.path.exists(sp):
                s = torch.from_numpy(np.load(sp)).float()
                # 确保维度对齐
                if len(s.shape) == 2: s = s.unsqueeze(0)
                if s.shape[-1] != 640: s = F.interpolate(s.unsqueeze(0), (480, 640), mode='bilinear').squeeze(0)
        return v, i, l, s


# --- Evaluator ---
class IOUEvaluator:
    def __init__(self): self.mat = np.zeros((9, 9))

    def add(self, g, p):
        m = (g >= 0) & (g < 9);
        self.mat += np.bincount(9 * g[m].astype(int) + p[m].astype(int), minlength=81).reshape(9, 9)

    def val(self):
        i = np.diag(self.mat);
        u = self.mat.sum(1) + self.mat.sum(0) - i
        return np.nanmean(i / (u + 1e-10))


def train():
    os.makedirs(f"checkpoints/{EXP_NAME}", exist_ok=True)
    base = build_sam2(SAM_CFG, SAM_CKPT, device="cpu")
    model = MultiTaskSerialModel(base, GlobalGuidedAoEBlock, num_classes=9).cuda()

    print(f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    scaler = GradScaler()
    crit_seg = StandardSegLoss(9)
    crit_sam = BinarySUMLoss(theta=0.6)

    # 加载数据集
    train_dl = DataLoader(MSRSDataset(TRAIN_DIRS, UNCERTAINTY_DIRS['train']), batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4)
    # 验证集也可以加载不确定性图（虽然通常验证只看IoU，但为了代码统一性）
    val_dl = DataLoader(MSRSDataset(VAL_DIRS, UNCERTAINTY_DIRS['test'], is_train=False), batch_size=2)

    best_iou = 0
    start = time.time()
    val_miou_history = []

    for ep in range(EPOCHS):
        model.train()
        l_seg_m, l_sam_m = 0, 0

        for v, i, l, s in tqdm(train_dl, desc=f"Ep {ep + 1}"):
            v, i, l, s = v.cuda(), i.cuda(), l.cuda(), s.cuda()
            with autocast():
                seg_out, sam_preds, moe_loss = model(v, i, l)

                # 1. Main Loss
                l_main = crit_seg(seg_out, l)

                # 2. Aux Loss (Only Stage 4)
                # 只有 rgb_s4 和 ir_s4
                l_rgb = crit_sam(sam_preds['rgb_s4'], l, s)
                l_ir = crit_sam(sam_preds['ir_s4'], l, s)
                l_aux = (l_rgb + l_ir) / 2.0

                loss = l_main + 0.5 * l_aux + 0.01 * moe_loss

            scaler.scale(loss).backward()
            scaler.step(opt);
            scaler.update();
            opt.zero_grad()
            l_seg_m += l_main.item();
            l_sam_m += l_aux.item()

        model.eval()
        evaluator = IOUEvaluator()
        with torch.no_grad():
            for v, i, l, _ in val_dl:
                with autocast(): seg, _, _ = model(v.cuda(), i.cuda(), None)
                evaluator.add(l.numpy(), torch.argmax(seg, 1).cpu().numpy())

        miou = evaluator.val()
        val_miou_history.append(miou)

        if miou > best_iou:
            best_iou = miou
            torch.save(model.state_dict(), f"checkpoints/{EXP_NAME}/best_model.pth")
            print(f"★ New Best: {best_iou:.4f}")
        print(
            f"Ep {ep + 1} SegL: {l_seg_m / len(train_dl):.3f} SAML: {l_sam_m / len(train_dl):.3f} Val mIoU: {miou:.4f}")

    print("\n" + "=" * 30)
    print("TRAINING REPORT (Stage 4 Only)")
    print(f"Total Time: {(time.time() - start) / 3600:.2f} hours")
    print(f"Best Val mIoU:  {best_iou:.4f}")
    print(f"Worst Val mIoU: {min(val_miou_history):.4f}")
    print(f"Average Val mIoU: {sum(val_miou_history) / len(val_miou_history):.4f}")
    print("=" * 30)


if __name__ == "__main__": train()