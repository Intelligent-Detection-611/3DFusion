import argparse
import torch
import nibabel as nib
import numpy as np

from models.fusion_model_3D import PIAFusion
from models.common_3D import clamp

def load_nifti_as_tensor(nifti_path):
    nii = nib.load(nifti_path)
    data = nii.get_fdata()
    data = (data - data.min()) / (data.max() - data.min() + 1e-8)
    tensor = torch.from_numpy(data.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    # shape: [1, 1, D, H, W]
    return tensor, nii.affine, nii.header

def save_tensor_as_nifti(tensor, affine, header, save_path):
    # tensor: [1, 1, D, H, W] or [1, D, H, W]
    data = tensor.squeeze().cpu().numpy()
    fused_nii = nib.Nifti1Image(data, affine, header)
    nib.save(fused_nii, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D NIfTI image fusion inference')
    parser.add_argument('--vi_nifti', type=str, required=True, help='Path to visible NIfTI image')
    parser.add_argument('--ir_nifti', type=str, required=True, help='Path to infrared NIfTI image')
    parser.add_argument('--fusion_pretrained', type=str, required=True, help='Path to trained PIAFusion model (.pth)')
    parser.add_argument('--output_nifti', type=str, required=True, help='Output fused NIfTI file')
    parser.add_argument('--cuda', default=True, type=bool, help='Use GPU if available')
    args = parser.parse_args()

    # Load and preprocess inputs
    vi_tensor, vi_affine, vi_header = load_nifti_as_tensor(args.vi_nifti)
    ir_tensor, _, _ = load_nifti_as_tensor(args.ir_nifti)

    # Move to CUDA/CPU
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    vi_tensor = vi_tensor.to(device)
    ir_tensor = ir_tensor.to(device)

    # Load model
    model = PIAFusion()
    model.load_state_dict(torch.load(args.fusion_pretrained, map_location=device))
    model = model.to(device)
    model.eval()

    # Forward pass
    with torch.no_grad():
        fused_tensor = model(vi_tensor, ir_tensor)
        fused_tensor = clamp(fused_tensor)

    # Save as NIfTI
    save_tensor_as_nifti(fused_tensor, vi_affine, vi_header, args.output_nifti)
    print(f'Fused image is saved to {args.output_nifti}')
