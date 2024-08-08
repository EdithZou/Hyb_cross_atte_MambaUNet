import nibabel as nib
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt

# 加载 .nii.gz 文件
file_path = '/data/MYOSAIQ_dataset/train_resampled/D8_postMI/labels/001_D8.nii.gz'
img = nib.load(file_path)
data = img.get_fdata()
data = data[..., 10]


# 查看数据的形状和标签范围
print(f"Data shape: {data.shape}")
print(f"Unique labels: {np.unique(data)}")

# 提取指定标签的点云 (例如标签1,2,3,4,5)
labels = [1, 2, 3, 4]
points = []

for label in labels:
    coords = np.argwhere(data == label)
    points.append(coords)

# 将点云合并
points = np.vstack(points)

# 创建Rips复形
rips_complex = gd.RipsComplex(points=points, max_edge_length=20.0)
simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)

# 计算持久同调
diag = simplex_tree.persistence()

# 可视化持久同调条形码
gd.plot_persistence_barcode(diag)
plt.show()