import os

def rename_files(folder):
    for filename in os.listdir(folder):
        if filename.endswith('.nii.gz'):
            # 只处理包含_ct或_mr的文件名
            if '_ct.nii.gz' in filename:
                new_name = filename.replace('_ct.nii.gz', '.nii.gz')
            elif '_mr.nii.gz' in filename:
                new_name = filename.replace('_mr.nii.gz', '.nii.gz')
            else:
                continue  # 跳过不符合的文件
            old_path = os.path.join(folder, filename)
            new_path = os.path.join(folder, new_name)
            os.rename(old_path, new_path)
            print(f"已重命名: {filename} -> {new_name}")

# 替换为你的文件夹路径，比如
# ct_folder = r"D:\data\nifti\ct"
# mr_folder = r"D:\data\nifti\mr"

ct_folder = r"D:\task\github\PIAFusion\datasets\nifti_train\ct"
mr_folder = r"D:\task\github\PIAFusion\datasets\nifti_train\mr"

rename_files(ct_folder)
rename_files(mr_folder)

print("所有文件重命名完成！")
