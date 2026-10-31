import os
import torch
from monai.data.dataset import Dataset
from monai.transforms import (
    Compose, Spacingd,ScaleIntensityRanged, RandCropByPosNegLabeld, RandFlipd,NormalizeIntensityd,
    RandRotate90d, RandScaleIntensityd, RandShiftIntensityd, ToTensord)
import random
import numpy as np
import nibabel as nib

class LargeMedicalDataSets(Dataset):

    def __init__(self, args):
        self.sample_list = []

        # 递归遍历所有子文件夹，收集 .nii.gz 文件
        for root, dirs, files in os.walk(args.base_dir):
            for filename in files:
                if filename.endswith(".nii.gz"):
                    self.sample_list.append(os.path.join(root, filename))

        # 打乱数据集
        random.shuffle(self.sample_list)

        # 按比例选取部分数据
        self.sample_list = self.sample_list[:int(len(self.sample_list) * args.data_ratio)]
        self.total = len(self.sample_list)

        # 数据增强（只针对 image）
        self.transform = Compose([

            RandCropByPosNegLabeld(spatial_size=(96, 96, 96),
                                   keys=["image"], label_key="image",
                                   pos=6, neg=2, num_samples=8),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image"], prob=0.3, max_k=3),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
            ToTensord(keys=["image"]),
        ])

        print(f"Total: {len(self.sample_list)} 3D volumes loaded.")
        print(self.sample_list[:5], "..." if len(self.sample_list) > 5 else "")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        file_path = self.sample_list[idx]  # 获取文件路径
        try:
            # 读取 nii.gz 文件
            nii_img = nib.load(file_path)
            img_data = nii_img.get_fdata().astype(np.float32)

            # 转换为 Tensor 并添加通道维度 (1, H, W, D)
            img_tensor = torch.tensor(img_data).unsqueeze(0)

            # 应用数据增强
            sample = self.transform({"image": img_tensor})

            return sample
        except Exception as e:
            print(f"Error processing file: {file_path}")
            print(f"Exception: {e}")
            raise

        return sample

