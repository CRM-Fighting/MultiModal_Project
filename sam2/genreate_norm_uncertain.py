import numpy as np
import torch
import cv2
import os
from pathlib import Path
import sys

# 尝试导入进度条
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("提示: 安装 tqdm 库可以显示进度条 (pip install tqdm)")

# 尝试导入 SAM2
try:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
except ImportError:
    print("错误：无法导入 SAM 2 库，请检查环境。")
    exit()


# ==========================================
#          核心算法：计算 ci (Float32)
# ==========================================

def generate_ci_map(masks, image_shape):
    """
    计算论文公式参数 ci。
    逻辑：Pixel Stack Count / Max Stack Count
    返回: float32 矩阵 (H, W)，范围 0.0 ~ 1.0
    """
    h, w = image_shape[:2]
    # 使用 float32 保证精度
    overlay_count = np.zeros((h, w), dtype=np.float32)

    if len(masks) == 0:
        return overlay_count

    # 1. 累加掩码覆盖次数
    for mask_data in masks:
        overlay_count[mask_data['segmentation']] += 1.0

    # 2. 获取最大堆叠数
    max_stack = np.max(overlay_count)

    # 3. 归一化 (防止除以0)
    if max_stack > 0:
        ci_map = overlay_count / max_stack
    else:
        ci_map = overlay_count  # 全0

    return ci_map.astype(np.float32)


def process_folder(input_dir, output_dir, mask_generator, max_files=None, description=""):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    image_files = sorted([f for f in input_path.iterdir() if f.suffix.lower() in valid_extensions])

    if max_files is not None:
        image_files = image_files[:max_files]

    total = len(image_files)

    print(f"\n任务: {description}")
    print(f"  源: {input_path}")
    print(f"  至: {output_path}")
    print(f"  量: {total} 张")

    iterator = tqdm(image_files, desc="Processing", unit="img") if HAS_TQDM else image_files

    for img_file in iterator:
        try:
            # 1. 读取原始图片
            image = cv2.imread(str(img_file))
            if image is None: continue

            # 2. SAM2 生成掩码
            masks = mask_generator.generate(image)

            # 3. 计算 ci 矩阵 (0.0 - 1.0 float32)
            ci_map = generate_ci_map(masks, image.shape)

            # 4. 保存为 .npy 文件
            # 文件名示例: '00134.png' -> 保存为 '00134.npy'
            # 使用 stem 获取不带后缀的文件名
            save_name = img_file.stem + ".npy"
            save_path = output_path / save_name

            # 核心步骤：保存 NumPy 数组
            np.save(str(save_path), ci_map)

        except Exception as e:
            msg = f"Error processing {img_file.name}: {e}"
            if HAS_TQDM:
                tqdm.write(msg)
            else:
                print(msg)


if __name__ == "__main__":
    # --- 1. 路径配置 ---
    # 根据您的 MSRS 数据集路径设置
    DATASET_ROOT = Path("/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS")

    # 输出目录命名为 'ci_maps_npy' 以示区别
    OUTPUT_ROOT = Path("/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS/uncertainty_map")

    input_paths = {
        "train_ir": DATASET_ROOT / "train/ir",
        "train_vi": DATASET_ROOT / "train/vi",
        "test_ir": DATASET_ROOT / "test/ir",
        "test_vi": DATASET_ROOT / "test/vi"
    }

    # 保持输出目录结构一致
    output_paths = {
        "train_ir": OUTPUT_ROOT / "train/ir",
        "train_vi": OUTPUT_ROOT / "train/vi",
        "test_ir": OUTPUT_ROOT / "test/ir",
        "test_vi": OUTPUT_ROOT / "test/vi"
    }

    # --- 2. 加载 SAM2 模型 ---
    # 请确保 checkpoint 相对路径正确
    checkpoint = "../checkpoints/sam2.1_hiera_tiny.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在加载 SAM2 模型到 {device} ...")

    try:
        model = build_sam2(model_cfg, checkpoint, device=device)
        mask_generator = SAM2AutomaticMaskGenerator(model)
        print("模型加载成功。")
    except Exception as e:
        print(f"模型加载失败: {e}")
        exit()

    # --- 3. 执行任务 ---

    # 处理训练集
    process_folder(input_paths["train_ir"], output_paths["train_ir"], mask_generator, description="[Train] IR Images")
    process_folder(input_paths["train_vi"], output_paths["train_vi"], mask_generator, description="[Train] Vis Images")

    # 处理验证集 (如需全部处理，请删除 max_files 参数)
    process_folder(input_paths["test_ir"], output_paths["test_ir"], mask_generator, max_files=100,
                   description="[Val] IR Images (Top 100)")
    process_folder(input_paths["test_vi"], output_paths["test_vi"], mask_generator, max_files=100,
                   description="[Val] Vis Images (Top 100)")

    print("\n所有任务完成。数据已保存为 .npy 格式。")