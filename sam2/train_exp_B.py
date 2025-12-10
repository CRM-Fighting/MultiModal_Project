import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import numpy as np

# 【修复1】强制使用 Agg 后端，防止 Tkinter 报错
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2
# 引用 Shared MoE 模型
from sam2.modeling.multimodal_sam_shared import MultiModalSegFormerShared

# ================= 全局配置 =================
# 【修复2】恢复辅助损失权重为 0.5
AUX_WEIGHT = 0.05

# 路径配置
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_CHECKPOINT = "../checkpoints/sam2.1_hiera_tiny.pt"
# 加载 Baseline 权重
PRETRAINED_SEG_PATH = "checkpoints/baseline_stage1/best_msrs_model_new_loss.pth"

# 保存路径
EXP_NAME = f"Exp_SharedMoE_Aux{AUX_WEIGHT}"
CHECKPOINT_DIR = f"checkpoints/{EXP_NAME}"
VIS_DIR = os.path.join(CHECKPOINT_DIR, "vis_plots")

TRAIN_DIRS = {
    'vi': r"/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/train/vi",
    'ir': r"/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/train/ir",
    'label': r"/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/train/Segmentation_labels"
}
VAL_DIRS = {
    'vi': r"/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/test/vi",
    'ir': r"/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/test/ir",
    'label': r"/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/test/Segmentation_labels"
}


# ================= 工具类 =================
class IOUEvaluator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_classes)
        label = self.num_classes * gt_image[mask].astype('int') + pre_image[mask].astype('int')
        count = np.bincount(label, minlength=self.num_classes ** 2)
        return count.reshape(self.num_classes, self.num_classes)

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def get_metrics(self):
        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - intersection
        iou = intersection / (union + 1e-10)
        miou = np.nanmean(iou)
        return miou, iou


def plot_training_curves(history, save_dir):
    # 确保数据长度一致
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(18, 6))

    # 1. Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Total Loss')
    plt.xlabel('Epochs');
    plt.ylabel('Loss');
    plt.legend();
    plt.grid(True)

    # 2. mIoU (修复了这里可能存在的 KeyError)
    plt.subplot(1, 3, 2)
    if 'train_miou' in history and len(history['train_miou']) == len(epochs):
        plt.plot(epochs, history['train_miou'], 'b--', label='Train mIoU')
    plt.plot(epochs, history['val_miou'], 'g-', label='Val mIoU')
    plt.title('mIoU Curve')
    plt.xlabel('Epochs');
    plt.ylabel('mIoU');
    plt.legend();
    plt.grid(True)

    # 3. Aux Loss
    if 'train_aux' in history and len(history['train_aux']) == len(epochs):
        plt.subplot(1, 3, 3)
        plt.plot(epochs, history['train_aux'], 'orange', label='Aux Loss')
        plt.title('Load Balancing Loss')
        plt.xlabel('Epochs');
        plt.legend();
        plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "training_metrics.png")
    plt.savefig(save_path)
    plt.close()
    print(f"曲线图已保存: {save_path}")


# ================= 损失函数 & 数据集 =================
class SegmentationLoss(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.weights = torch.ones(num_classes)
        self.weights[0] = 0.1

    def forward(self, preds, labels):
        if self.weights.device != preds.device:
            self.weights = self.weights.to(preds.device)

        # CE
        loss_ce = F.cross_entropy(preds, labels, weight=self.weights, ignore_index=255)
        # Dice
        preds_soft = F.softmax(preds, dim=1)
        labels_one_hot = F.one_hot(labels, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()
        inter = (preds_soft * labels_one_hot).sum(dim=(2, 3))
        union = preds_soft.sum(dim=(2, 3)) + labels_one_hot.sum(dim=(2, 3))
        loss_dice = 1 - (2. * inter + 1) / (union + 1)

        return 0.3 * loss_ce + 0.7 * loss_dice.mean()


class MSRSDataset(Dataset):
    def __init__(self, root_dirs, limit=None):
        self.vis_dir = root_dirs['vi']
        self.ir_dir = root_dirs['ir']
        self.label_dir = root_dirs['label']
        self.filenames = sorted([f for f in os.listdir(self.vis_dir) if f.endswith('.png')])
        if limit: self.filenames = self.filenames[:limit]

    def __len__(self): return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        vis = Image.open(os.path.join(self.vis_dir, name)).convert('RGB').resize((640, 480))
        ir = Image.open(os.path.join(self.ir_dir, name)).convert('RGB').resize((640, 480))
        label = Image.open(os.path.join(self.label_dir, name)).resize((640, 480), Image.NEAREST)

        vis_t = torch.from_numpy(np.array(vis)).float().permute(2, 0, 1) / 255.0
        ir_t = torch.from_numpy(np.array(ir)).float().permute(2, 0, 1) / 255.0
        label_t = torch.from_numpy(np.array(label)).long()
        return vis_t, ir_t, label_t


# ================= 主程序 =================
def train_shared_moe():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    print(f"=== Running Experiment: {EXP_NAME} ===")
    print(f"=== Aux Loss Weight: {AUX_WEIGHT} ===")

    # 1. 构建模型
    print("Building Model...")
    base_sam = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device="cpu")
    model = MultiModalSegFormerShared(base_sam, [96, 192, 384, 768], num_classes=9)

    # 2. 加载 Baseline 权重
    if os.path.exists(PRETRAINED_SEG_PATH):
        print(f"Loading baseline from {PRETRAINED_SEG_PATH}...")
        pretrained = torch.load(PRETRAINED_SEG_PATH, map_location="cpu")
        model_dict = model.state_dict()
        # 过滤不匹配的层
        filtered = {k: v for k, v in pretrained.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        print(f"Loaded {len(filtered)} layers.")
    else:
        print("Warning: No baseline weights found!")

    model.to(device)

    # 3. 优化器配置
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable Params: {sum(p.numel() for p in trainable) / 1e6:.2f} M")

    optimizer = optim.AdamW(trainable, lr=0.0001)
    criterion = SegmentationLoss(num_classes=9)
    evaluator = IOUEvaluator(num_classes=9)

    # 【修复3】完整初始化记录器
    history = {
        'train_loss': [], 'val_loss': [],
        'train_miou': [], 'val_miou': [],
        'train_aux': []
    }
    best_miou = 0.0

    # 4. 训练循环
    for epoch in range(50):
        print(f"\nEpoch {epoch + 1}/50")

        # --- Train ---
        model.train()
        train_loss_meter = 0
        aux_meter = 0
        evaluator.reset()  # 重置 evaluator

        pbar = tqdm(DataLoader(MSRSDataset(TRAIN_DIRS), batch_size=2, shuffle=True, num_workers=4))

        for vis, ir, label in pbar:
            vis, ir, label = vis.to(device), ir.to(device), label.to(device)
            optimizer.zero_grad()

            preds, aux_loss = model(vis, ir)
            seg_loss = criterion(preds, label)

            # 总损失
            total_loss = seg_loss + AUX_WEIGHT * aux_loss

            total_loss.backward()
            optimizer.step()

            train_loss_meter += total_loss.item()
            aux_meter += aux_loss.item()

            # 计算 Train mIoU
            pred_mask = torch.argmax(preds, dim=1)
            evaluator.add_batch(label.cpu().numpy(), pred_mask.cpu().numpy())

            pbar.set_postfix({"Loss": f"{total_loss.item():.3f}", "Aux": f"{aux_loss.item():.3f}"})

        # 记录 Train 数据
        train_miou, _ = evaluator.get_metrics()
        history['train_loss'].append(train_loss_meter / 542)
        history['train_aux'].append(aux_meter / 542)
        history['train_miou'].append(train_miou)

        # --- Val ---
        model.eval()
        evaluator.reset()
        val_loss_meter = 0

        with torch.no_grad():
            for vis, ir, label in tqdm(DataLoader(MSRSDataset(VAL_DIRS, limit=100), batch_size=2, num_workers=2)):
                vis, ir, label = vis.to(device), ir.to(device), label.to(device)
                preds, _ = model(vis, ir)

                loss = criterion(preds, label)
                val_loss_meter += loss.item()

                pred_mask = torch.argmax(preds, dim=1)
                evaluator.add_batch(label.cpu().numpy(), pred_mask.cpu().numpy())

        val_miou, _ = evaluator.get_metrics()
        history['val_loss'].append(val_loss_meter / 50)
        history['val_miou'].append(val_miou)

        print(f"Val Loss: {val_loss_meter / 50:.4f} | mIoU: {val_miou:.4f}")

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print(f"★ New Best! mIoU: {best_miou:.4f}")

        # 绘图
        plot_training_curves(history, VIS_DIR)

    print(f"Done. Best mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    train_shared_moe()