import os
import nibabel as nib
import csv

# 定义根文件夹路径
root_dir = '/home/tzou/Mamba_model/MyoSAIQ/Cascade_Fully_Supervised_ViM_716_3_stage_attn/result_test_attn'

# 定义子文件夹名称
subfolders = ['D8_postMI', 'M1_postMI', 'M12_postMI']

# 定义输出CSV文件路径
output_csv = '/home/tzou/MyoSAIQ/nnUnet/MYOSAIQ_data_test_resampled_shape.csv'

# 打开CSV文件以写入模式
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    # 写入CSV文件的表头
    writer.writerow(['File Path', 'Dimension 1', 'Dimension 2', 'Dimension 3'])

    # 遍历每个子文件夹
    for subfolder in subfolders:
        images_path = os.path.join(root_dir, subfolder)
        if os.path.exists(images_path):
            for file_name in os.listdir(images_path):
                if file_name.endswith('.nii.gz'):
                    file_path = os.path.join(images_path, file_name)
                    img = nib.load(file_path)
                    dimensions = img.shape
                    # 写入文件路径和维度信息到CSV文件
                    writer.writerow([file_path] + list(dimensions))
        else:
            print(f"Path {images_path} does not exist")
