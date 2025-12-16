import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.modeling.standard_moe import StandardMoEBlock
from sam2.modeling.serial_modeling import SerialSegModel
from utils.custom_losses import StandardSegLoss

# 配置
EXP_NAME = "Exp_Standard_Serial"
TRAIN_DIRS = {'vi': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/train/vi',
              'ir': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/train/ir',
              'label': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/train/Segmentation_labels'}
VAL_DIRS = {'vi': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/test/vi',
            'ir': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/test/ir',
            'label': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/test/Segmentation_labels'}
SAM_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM_CKPT = "../checkpoints/sam2.1_hiera_tiny.pt"
BATCH_SIZE = 2
EPOCHS = 50


class MSRSDataset(Dataset):
    def __init__(self, dirs, is_train=True):
        self.vis = dirs['vi'];
        self.ir = dirs['ir'];
        self.lbl = dirs['label']
        self.files = sorted([f for f in os.listdir(self.vis) if f.endswith('.png')])

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        n = self.files[i]
        v = torch.from_numpy(
            np.array(Image.open(os.path.join(self.vis, n)).convert('RGB').resize((640, 480)))).float().permute(2, 0,
                                                                                                               1) / 255.0
        i = torch.from_numpy(
            np.array(Image.open(os.path.join(self.ir, n)).convert('RGB').resize((640, 480)))).float().permute(2, 0,
                                                                                                              1) / 255.0
        l = torch.from_numpy(np.array(Image.open(os.path.join(self.lbl, n)).resize((640, 480), Image.NEAREST))).long()
        return v, i, l


class IOUEvaluator:
    def __init__(self): self.mat = np.zeros((9, 9))

    def reset(self): self.mat = np.zeros((9, 9))

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
    model = SerialSegModel(base, StandardMoEBlock, num_classes=9).cuda()

    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    scaler = GradScaler()
    crit = StandardSegLoss(9)

    train_dl = DataLoader(MSRSDataset(TRAIN_DIRS), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dl = DataLoader(MSRSDataset(VAL_DIRS, is_train=False), batch_size=2)

    best_miou = 0
    start = time.time()
    val_miou_history = []

    for ep in range(EPOCHS):
        model.train()
        l_sum = 0
        for v, i, l in tqdm(train_dl, desc=f"Ep {ep + 1}"):
            v, i, l = v.cuda(), i.cuda(), l.cuda()
            with autocast():
                out, aux = model(v, i)
                loss = crit(out, l) + 0.01 * aux

            scaler.scale(loss).backward()
            scaler.step(opt);
            scaler.update();
            opt.zero_grad()
            l_sum += loss.item()

        model.eval()
        evaluator = IOUEvaluator()
        with torch.no_grad():
            for v, i, l in val_dl:
                with autocast(): out, _ = model(v.cuda(), i.cuda())
                evaluator.add(l.numpy(), torch.argmax(out, 1).cpu().numpy())

        miou = evaluator.val()
        val_miou_history.append(miou)

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), f"checkpoints/{EXP_NAME}/best_model.pth")
            print(f"★ New Best: {best_miou:.4f}")

        print(f"Ep {ep + 1} Loss: {l_sum / len(train_dl):.4f} Val mIoU: {miou:.4f}")

    print("\n" + "=" * 30)
    print("TRAINING REPORT (Standard MoE)")
    print(f"Total Time: {(time.time() - start) / 3600:.2f} hours")
    print(f"Best Val mIoU:  {best_miou:.4f}")
    print(f"Worst Val mIoU: {min(val_miou_history):.4f}")
    print(f"Average Val mIoU: {sum(val_miou_history) / len(val_miou_history):.4f}")
    print("=" * 30)


if __name__ == "__main__": train()