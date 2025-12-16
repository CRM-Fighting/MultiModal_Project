import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sam2.build_sam import build_sam2
from sam2.modeling.global_guided_aoe import GlobalGuidedAoEBlock  # <--- Change
from sam2.modeling.serial_modeling import SerialSegModel
from test_standard_moe_serial import MSRSDatasetTest  # 复用

EXP_NAME = "Exp_Global_Serial"  # <--- Change
VAL_DIRS = {'vi': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/test/vi',
            'ir': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/test/ir',
            'label': '/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/test/Segmentation_labels'}
SAM_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM_CKPT = "../checkpoints/sam2.1_hiera_tiny.pt"


def test():
    base = build_sam2(SAM_CFG, SAM_CKPT, device="cpu")
    model = SerialSegModel(base, GlobalGuidedAoEBlock, num_classes=9).cuda()
    model.load_state_dict(torch.load(f"checkpoints/{EXP_NAME}/best.pth"))
    model.eval()

    dl = DataLoader(MSRSDatasetTest(VAL_DIRS), batch_size=1)
    total_mat = np.zeros((9, 9))
    iou_per_img = []

    print("Testing Global MoE Serial...")
    with torch.no_grad():
        for v, i, l in tqdm(dl):
            out, _ = model(v.cuda(), i.cuda())
            pred = torch.argmax(out, 1).cpu().numpy();
            gt = l.numpy()
            mask = (gt >= 0) & (gt < 9)
            label = 9 * gt[mask].astype(int) + pred[mask].astype(int)
            count = np.bincount(label, minlength=81).reshape(9, 9)
            total_mat += count
            inter = np.diag(count)
            union = count.sum(1) + count.sum(0) - inter
            iou_per_img.append(np.nanmean(inter / (union + 1e-10)))

    inter = np.diag(total_mat)
    union = total_mat.sum(1) + total_mat.sum(0) - inter
    class_iou = inter / (union + 1e-10)
    print("=" * 30)
    print(f"Dataset mIoU: {np.nanmean(class_iou):.4f}")
    print(f"Best Image mIoU: {max(iou_per_img):.4f}")
    print(f"Worst Image mIoU: {min(iou_per_img):.4f}")
    print(f"Class IoU: {class_iou}")
    print("=" * 30)


if __name__ == "__main__": test()