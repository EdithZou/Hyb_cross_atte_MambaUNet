import csv
import os
import subprocess

# 定义CSV文件路径
input_csv = 'file_dimensions.csv'

# 定义C3D可执行文件路径
c3d_executable = '/home/user/Applications/c3d-1.1.0-Linux-gcc64/bin/c3d'

# 定义输出文件夹
output_folder = '/home/tzou/MYOSAIQ_dataset/test_resampled/resampled_images/'

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 读取CSV文件并执行C3D命令
with open(input_csv, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # 跳过表头

    for row in reader:
        source_file = row[0]
        target_resolution = f"{row[1]}x{row[2]}x{row[3]}"
        file_name = os.path.basename(source_file)
        output_file = os.path.join(output_folder, file_name)

        cmd_img = [
            c3d_executable,
            source_file,
            '-interpolation', 'Cubic',
            '-resample', target_resolution,
            '-o', output_file
        ]
        subprocess.run(cmd_img, check=True)
        print(f"Resampled {source_file} to {target_resolution} and saved to {output_file}")
