import numpy as np
import torch
import cv2
import os
from pathlib import Path

# 假设 sam2 库在您的 Python 路径中
try:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
except ImportError:
    print("错误：无法导入 SAM 2 库。请确保已安装。")
    exit()


def generate_heatmap_uint8(masks, image_shape):
    """
    生成叠加热度图，并归一化为 0-255 的 uint8 图像。
    黑色 (0) 代表无掩码，白色 (255) 代表掩码重叠最多。
    """
    # 初始化累加层 (使用 int32 防止溢出)
    overlay = np.zeros(image_shape[:2], dtype=np.int32)

    if len(masks) == 0:
        return np.zeros(image_shape[:2], dtype=np.uint8)

    # 累加所有 mask
    for mask_data in masks:
        overlay[mask_data['segmentation']] += 1

    # 获取最大重叠数，用于归一化
    max_val = np.max(overlay)

    if max_val == 0:
        return overlay.astype(np.uint8)

    # 归一化到 0-255
    # 公式: (当前值 / 最大值) * 255
    heatmap_norm = (overlay / max_val) * 255

    # 转换为图像格式
    heatmap_uint8 = heatmap_norm.astype(np.uint8)

    return heatmap_uint8


def process_folder(input_dir, output_dir, mask_generator, max_files=None, description=""):
    """
    处理单个文件夹：读取 -> 生成 Mask -> 保存热度图
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 获取所有图片文件 (支持 png, jpg, jpeg, bmp)
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    image_files = sorted([f for f in input_path.iterdir() if f.suffix.lower() in valid_extensions])

    # 如果有数量限制 (例如验证集只取前100)
    if max_files is not None:
        image_files = image_files[:max_files]

    total = len(image_files)
    print(f"\n开始处理 [{description}]: 共有 {total} 张图片 -> 输出至 {output_path}")

    for idx, img_file in enumerate(image_files):
        try:
            # 1. 读取图像
            image = cv2.imread(str(img_file))
            if image is None:
                print(f"  [警告] 无法读取: {img_file.name}")
                continue

            # 2. 生成 Masks
            masks = mask_generator.generate(image)

            # 3. 生成堆叠热度图 (0-255 灰度图)
            heatmap_img = generate_heatmap_uint8(masks, image.shape)

            # 4. 保存图片 (文件名与原图相同)
            save_path = output_path / img_file.name
            cv2.imwrite(str(save_path), heatmap_img)

            if (idx + 1) % 10 == 0:
                print(f"  进度: {idx + 1}/{total} - {img_file.name} 完成")

        except Exception as e:
            print(f"  [错误] 处理 {img_file.name} 时出错: {e}")


if __name__ == "__main__":

    # --- 1. 配置路径 (请根据实际情况调整盘符或路径) ---

    # 输入数据集根目录
    DATASET_ROOT = Path("/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/MSRS")

    # 输出根目录
    OUTPUT_ROOT = Path("/home/mmsys/disk/MCL/MultiModal_Project/sam2/data/overlap_map")

    # 定义具体的输入路径
    input_paths = {
        "train_ir": DATASET_ROOT / "train/ir",
        "train_vi": DATASET_ROOT / "train/vi",
        "test_ir": DATASET_ROOT / "test/ir",
        "test_vi": DATASET_ROOT / "test/vi"
    }

    # 定义输出路径结构
    output_paths = {
        "train_ir": OUTPUT_ROOT / "train/ir",
        "train_vi": OUTPUT_ROOT / "train/vi",  # 注意：输出文件夹名为 vis
        "val_ir": OUTPUT_ROOT / "val/ir",
        "val_vi": OUTPUT_ROOT / "val/vi"
    }

    # --- 2. 初始化模型 ---

    # 自动定位当前脚本目录，寻找 checkpoint
    script_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()

    # 请确保以下相对路径指向正确，或者修改为绝对路径
    checkpoint = "../checkpoints/sam2.1_hiera_tiny.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在使用设备: {device}")

    try:
        model = build_sam2(model_cfg, checkpoint, device=device)
    except Exception as e:
        print(f"加载 SAM2 失败，请检查路径: {e}")
    mask_generator = SAM2AutomaticMaskGenerator(model)
    print("模型加载完成。")

    # --- 3. 执行处理任务 ---

    # 任务 1: 训练集 - 红外 (所有图片)
    # process_folder(
    #     input_dir=input_paths["train_ir"],
    #     output_dir=output_paths["train_ir"],
    #     mask_generator=mask_generator,
    #     max_files=None,  # 不限制数量
    #     description="Train Set - IR"
    # )

    # 任务 2: 训练集 - 可见光 (所有图片)
    process_folder(
        input_dir=input_paths["train_vi"],
        output_dir=output_paths["train_vi"],
        mask_generator=mask_generator,
        max_files=None,  # 不限制数量
        description="Train Set - Visible"
    )

    # 任务 3: 验证集 - 红外 (取 MSRS test 的前 100 张)
    process_folder(
        input_dir=input_paths["test_ir"],
        output_dir=output_paths["val_ir"],
        mask_generator=mask_generator,
        max_files=100,  # 限制前 100 张
        description="Val Set - IR"
    )

    # 任务 4: 验证集 - 可见光 (取 MSRS test 的前 100 张)
    process_folder(
        input_dir=input_paths["test_vi"],
        output_dir=output_paths["val_vi"],
        mask_generator=mask_generator,
        max_files=100,  # 限制前 100 张
        description="Val Set - Visible"
    )

    print("\n所有任务处理完毕！")