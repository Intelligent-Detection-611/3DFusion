import os
import nibabel as nib
import torch
from torch.utils.data import Dataset
import numpy as np

class NiftiFusionDataset(Dataset):
    """
    数据文件结构:
    dataset_path/
      ct/
        xxx.nii(.gz)
      ir/
        xxx.nii(.gz)
    要求vi和ir下文件名一一对应。
    """

    def __init__(self, dataset_path):
        self.vi_dir = os.path.join(dataset_path, 'ct')
        self.ir_dir = os.path.join(dataset_path, 'mr')
        self.vi_paths = sorted([os.path.join(self.vi_dir, f) for f in os.listdir(self.vi_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
        self.ir_paths = sorted([os.path.join(self.ir_dir, f) for f in os.listdir(self.ir_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
        assert len(self.vi_paths) == len(self.ir_paths), '数量不一致'
        # 可以增加更严格的文件名约束
        for v, i in zip(self.vi_paths, self.ir_paths):
            assert os.path.basename(v) == os.path.basename(i), "文件名需一一对应"

    def __len__(self):
        return len(self.vi_paths)

    def __getitem__(self, idx):
        vi_nii = nib.load(self.vi_paths[idx])
        ir_nii = nib.load(self.ir_paths[idx])
        vi = vi_nii.get_fdata()
        ir = ir_nii.get_fdata()
        # 归一化到[0,1]
        vi = (vi - vi.min()) / (vi.max() - vi.min() + 1e-8)
        ir = (ir - ir.min()) / (ir.max() - ir.min() + 1e-8)
        vi = torch.from_numpy(vi.astype(np.float32)).unsqueeze(0)  # [1, D, H, W]
        ir = torch.from_numpy(ir.astype(np.float32)).unsqueeze(0)
        return vi, ir
