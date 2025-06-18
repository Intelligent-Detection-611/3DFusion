import os
import shutil

# 设置原始文件夹路径，和目标文件夹路径
src_folder = r"D:\task\dataset\MambaMorph_Data\volumes_center"
ct_folder = os.path.join(src_folder, "ct")
mr_folder = os.path.join(src_folder, "mr")

# 创建目标文件夹（如果不存在）
os.makedirs(ct_folder, exist_ok=True)
os.makedirs(mr_folder, exist_ok=True)

for filename in os.listdir(src_folder):
    # 只处理nii.gz文件
    if filename.endswith('.nii.gz'):
        if '_ct.nii.gz' in filename:
            shutil.copy2(os.path.join(src_folder, filename), os.path.join(ct_folder, filename))
        elif '_mr.nii.gz' in filename:
            shutil.copy2(os.path.join(src_folder, filename), os.path.join(mr_folder, filename))

print("拷贝完成！")
