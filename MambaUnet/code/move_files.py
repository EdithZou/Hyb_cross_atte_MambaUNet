import os
import shutil

# 指定要分类的文件夹路径
source_folder = '/home/tzou/Mamba_model/MyoSAIQ/Cascade_Fully_Supervised_ViM_716_3_stage_attn/mambaunet/mambaunet_predictions'
# 指定目标文件夹路径
dest_folder_D8 = '/home/tzou/Mamba_model/MyoSAIQ/Cascade_Fully_Supervised_ViM_716_3_stage_attn/mambaunet/mambaunet_predictions/D8_postMI'
dest_folder_M1 = '/home/tzou/Mamba_model/MyoSAIQ/Cascade_Fully_Supervised_ViM_716_3_stage_attn/mambaunet/mambaunet_predictions/M1_postMI'
dest_folder_M12 = '/home/tzou/Mamba_model/MyoSAIQ/Cascade_Fully_Supervised_ViM_716_3_stage_attn/mambaunet/mambaunet_predictions/M12_postMI'

# 确保目标文件夹存在，不存在则创建
os.makedirs(dest_folder_D8, exist_ok=True)
os.makedirs(dest_folder_M1, exist_ok=True)
os.makedirs(dest_folder_M12, exist_ok=True)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    # 获取文件的完整路径
    file_path = os.path.join(source_folder, filename)
    # 检查文件是否包含特定字符串，并移动到相应的文件夹
    if 'D8' in filename:
        shutil.move(file_path, os.path.join(dest_folder_D8, filename))
    elif 'M1' in filename and 'M12' not in filename:
        shutil.move(file_path, os.path.join(dest_folder_M1, filename))
    elif 'M12' in filename:
        shutil.move(file_path, os.path.join(dest_folder_M12, filename))

print("文件分类完成。")
