import os
import time
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 引入你的模型定义
# 确保你的文件夹结构里包含 sam2.build_sam 和 sam2.modeling
from sam2.build_sam import build_sam2
from sam2.modeling.multimodal_sam import MultiModalSegFormer

# ================= 配置区域 =================
# 1. 路径设置 (Linux 绝对路径或相对路径)
# 假设脚本运行在 .../sam2/ 目录下
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# 数据集路径
TEST_ROOT = os.path.join(CURRENT_DIR, "data", "MSRS", "test")
DIR_VIS = os.path.join(TEST_ROOT, "vi")
DIR_IR = os.path.join(TEST_ROOT, "ir")
# 注意：确认你的标签文件夹名是 'Segmentation_labels' 还是 'label'
# 根据你上次提供的代码，应该是 Segmentation_labels
DIR_LABEL = os.path.join(TEST_ROOT, "Segmentation_labels")

# 权重路径 (请修改为你训练好的最佳权重路径)
# 假设你上次训练保存到了 checkpoints/best_msrs_model.pth
MODEL_WEIGHT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_msrs_model.pth")

# 结果保存路径
SAVE_DIR = os.path.join(CURRENT_DIR, "vis_results")

# 2. 模型参数
SAM2_CONFIG = os.path.join("sam2.1", "sam2.1_hiera_t.yaml")
SAM2_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "sam2.1_hiera_tiny.pt")
BACKBONE_CHANNELS = [96, 192, 384, 768]
NUM_CLASSES = 9

# 3. 颜色表 (用于将类别索引 0-8 转换成彩色)
# MSRS 9类常用颜色 (背景, 汽车, 行人, 自行车, 曲线, 汽车站, 护栏, 颜色锥, 凸起)
# 这里随机生成了一组颜色，你可以替换成 MSRS 官方配色
PALETTE = np.array([
    [0, 0, 0],  # 0: Background (黑色)
    [128, 0, 0],  # 1: Car (深红)
    [0, 128, 0],  # 2: Person (深绿)
    [128, 128, 0],  # 3: Bike
    [0, 0, 128],  # 4: Curve
    [128, 0, 128],  # 5: Car Stop
    [0, 128, 128],  # 6: Guardrail
    [128, 128, 128],  # 7: Color Cone
    [64, 0, 0]  # 8: Bump
], dtype=np.uint8)


# ============================================

class MSRSTestDataset(Dataset):
    def __init__(self, vis_dir, ir_dir, label_dir, skip_first=100):
        self.vis_dir = vis_dir
        self.ir_dir = ir_dir
        self.label_dir = label_dir

        # 获取排序后的文件名列表
        all_files = sorted([f for f in os.listdir(vis_dir) if f.endswith('.png')])

        # 【关键】跳过前 100 张，取剩余图片
        if len(all_files) > skip_first:
            self.filenames = all_files[skip_first:]
            print(f"原始测试集共 {len(all_files)} 张，跳过前 {skip_first} 张，剩余 {len(self.filenames)} 张用于可视化。")
        else:
            self.filenames = all_files
            print(f"警告：图片总数 ({len(all_files)}) 不足 {skip_first} 张，将使用全部图片。")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        vis_path = os.path.join(self.vis_dir, fname)
        ir_path = os.path.join(self.ir_dir, fname)
        label_path = os.path.join(self.label_dir, fname)

        # 读取原始图片 (用于可视化展示)
        vis_raw = Image.open(vis_path).convert('RGB')
        ir_raw = Image.open(ir_path).convert('L')  # 红外展示用灰度
        label_raw = Image.open(label_path)  # 标签

        # 预处理用于模型输入 (Resize -> Tensor)
        target_size = (640, 480)

        # 1. 原始尺寸图片 (为了画图好看，我们存一下)
        vis_np = np.array(vis_raw.resize(target_size))
        ir_np = np.array(ir_raw.resize(target_size))
        label_np = np.array(label_raw.resize(target_size, Image.NEAREST))

        # 2. 模型输入 Tensor
        vis_t = torch.from_numpy(np.array(vis_raw.convert('RGB').resize(target_size))).float().permute(2, 0, 1) / 255.0
        # IR 转 3通道适配 SAM2
        ir_t = torch.from_numpy(np.array(ir_raw.convert('RGB').resize(target_size))).float().permute(2, 0, 1) / 255.0

        return vis_t, ir_t, vis_np, ir_np, label_np, fname


def colorize_mask(mask, palette):
    """将类别索引掩码 (H, W) 转换为彩色图片 (H, W, 3)"""
    # mask: (H, W) with 0-8
    # 处理 ignore_index (255)
    mask[mask == 255] = 0
    # 查表
    color_mask = palette[mask]
    return color_mask


def main():
    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device} (RTX 2080 应该显示 cuda)")

    # 2. 准备保存目录
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"创建结果目录: {SAVE_DIR}")

    # 3. 加载模型
    print("正在加载模型...")
    # 构建基础 SAM2
    try:
        base_sam = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    except Exception as e:
        print(f"加载 SAM2 失败，请检查路径: {e}")
        return

    # 构建你的多模态模型
    model = MultiModalSegFormer(base_sam, feature_channels=BACKBONE_CHANNELS, num_classes=NUM_CLASSES)

    # 加载训练好的权重
    if os.path.exists(MODEL_WEIGHT_PATH):
        print(f"加载权重: {MODEL_WEIGHT_PATH}")
        checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        print(f"错误: 找不到权重文件 {MODEL_WEIGHT_PATH}，无法进行可视化！")
        return

    model.to(device)
    model.eval()  # 开启评估模式 (关闭 Dropout 等)

    # 4. 准备数据加载器
    test_ds = MSRSTestDataset(DIR_VIS, DIR_IR, DIR_LABEL, skip_first=100)
    # batch_size=1 方便一张张处理和保存
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    print("开始生成可视化结果...")

    with torch.no_grad():  # 不计算梯度，节省显存
        for vis_t, ir_t, vis_np, ir_np, label_np, fname in tqdm(test_loader):
            fname = fname[0]  # 取出文件名字符串
            vis_t = vis_t.to(device)
            ir_t = ir_t.to(device)

            # --- 推理 ---
            # forward: (B, 9, 480, 640)
            logits = model(vis_t, ir_t)

            # --- 后处理 ---
            # 取最大概率的类别 (B, 480, 640)
            pred_mask = torch.argmax(logits, dim=1)
            pred_mask = pred_mask.squeeze(0).cpu().numpy().astype(np.uint8)  # 转回 CPU numpy

            # --- 可视化拼接 ---
            # 1. 真实标签彩色化
            gt_color = colorize_mask(label_np[0].numpy().astype(np.uint8), PALETTE)
            # 2. 预测结果彩色化
            pred_color = colorize_mask(pred_mask, PALETTE)
            # 3. 原始 RGB (去掉 batch 维)
            vis_img = vis_np[0].numpy().astype(np.uint8)
            # 4. 原始 IR (转成 3通道灰度以便拼接)
            ir_img = ir_np[0].numpy().astype(np.uint8)
            ir_img_3c = np.stack([ir_img] * 3, axis=-1)

            # 拼接: 上面是 [RGB, IR], 下面是 [GT, Prediction]
            # 或者横着拼四张: RGB | IR | GT | Pred
            row1 = np.hstack([vis_img, ir_img_3c])
            row2 = np.hstack([gt_color, pred_color])
            final_img = np.vstack([row1, row2])

            # --- 保存 ---
            save_path = os.path.join(SAVE_DIR, f"res_{fname}")
            Image.fromarray(final_img).save(save_path)

    print(f"\n所有结果已保存至: {SAVE_DIR}")
    print("你可以通过 VSCode 的图片查看器或者将文件夹传回本地查看效果。")


if __name__ == "__main__":
    main()

