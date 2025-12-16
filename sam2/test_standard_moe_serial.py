import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.modeling.standard_moe import StandardMoEBlock
from sam2.modeling.serial_modeling import SerialSegModel

EXP_NAME = "Exp_Standard_Serial"
VAL_DIRS = {'vi': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/test/vi',
            'ir': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/test/ir',
            'label': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/test/Segmentation_labels'}
SAM_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM_CKPT = "../checkpoints/sam2.1_hiera_tiny.pt"


class MSRSDatasetTest(torch.utils.data.Dataset):
    def __init__(self, dirs):
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


def test():
    base = build_sam2(SAM_CFG, SAM_CKPT, device="cpu")
    model = SerialSegModel(base, StandardMoEBlock, num_classes=9).cuda()
    model.load_state_dict(torch.load(f"checkpoints/{EXP_NAME}/best.pth"))
    model.eval()

    dl = DataLoader(MSRSDatasetTest(VAL_DIRS), batch_size=1)

    # 全局矩阵
    total_mat = np.zeros((9, 9))
    iou_per_img = []

    print("Testing Standard MoE Serial...")
    with torch.no_grad():
        for v, i, l in tqdm(dl):
            out, _ = model(v.cuda(), i.cuda())
            pred = torch.argmax(out, 1).cpu().numpy()
            gt = l.numpy()

            # 单张计算
            mask = (gt >= 0) & (gt < 9)
            label = 9 * gt[mask].astype(int) + pred[mask].astype(int)
            count = np.bincount(label, minlength=81).reshape(9, 9)

            # 累加全局
            total_mat += count

            # 计算单张 mIoU
            intersect = np.diag(count)
            union = count.sum(1) + count.sum(0) - intersect
            img_iou = np.nanmean(intersect / (union + 1e-10))
            iou_per_img.append(img_iou)

    # 全局指标
    inter = np.diag(total_mat)
    union = total_mat.sum(1) + total_mat.sum(0) - inter
    class_iou = inter / (union + 1e-10)
    miou = np.nanmean(class_iou)

    print("=" * 30)
    print(f"Dataset mIoU: {miou:.4f}")
    print(f"Best Image mIoU: {max(iou_per_img):.4f}")
    print(f"Worst Image mIoU: {min(iou_per_img):.4f}")
    print(f"Average Image mIoU: {np.mean(iou_per_img):.4f}")
    print("Class IoU:", class_iou)
    print("=" * 30)


if __name__ == "__main__": test()