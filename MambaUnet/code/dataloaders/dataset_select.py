import os
import cv2
import torch
import random
import numpy as np

import nibabel as nib
import pandas as pd

from glob import glob
from torch.utils.data import Dataset
#import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import augmentations
from augmentations.ctaugment import OPS
import matplotlib.pyplot as plt
from PIL import Image

# 用于MyoSAIQ的数据处理
class BaseDataSets(Dataset):
    def __init__(
        self,
        csv_file,
        base_dir=None,
        split="train",
        num=None,
        transform=None, 
        ops_weak=None,
        ops_strong=None,
        #slice_range = (, 20), # 选择含有较多准确label信息的第3-18层的切片
        random_state = 42
    ):
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong
        self.slice_range_height = 96
        self.slice_range_width = 96
        self.slices = []
        self.slices_mvo = []
        self.volumes = []
        self.volumes_mvo =[]
        
        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        # 从 CSV 文件中读取数据
        data = pd.read_csv(csv_file, header=None, names=["folder_path", "filetype", "subset", "filename"])
                
        subsets = data['subset'].unique()[1:]

        # 将数据分为训练集和验证集
        self.sample_list = []          
        # 这个subset代表的就是不同的子数据集
        for subset in subsets:
            subset_data = data[data['subset'] == subset]                 
            '''
            if split == "train":
                self.sample_list.extend(subset_data.iloc(frac=0.8, random_state=42).values)
            elif split == "valid":
                self.sample_list.extend(subset_data.drop(subset_data.sample(frac=0.8, random_state=42).index).values)            
            '''
            if self.split == "train":
                self.sample_list.extend(subset_data.sample(frac=0.8, random_state = random_state).values)
            elif self.split == "valid":
                self.sample_list.extend(subset_data.drop(subset_data.sample(frac=0.8, random_state=42).index).values)

        print("total {} samples".format(len(self.sample_list)))
        
        self._prepare_slices()

    def _prepare_slices(self):
        if self.split == "train":
            for sample in self.sample_list:
                image_path = sample[0]
                subset = sample[2]
                image_file = sample[3]
                label_path = image_path.replace("images", "labels")
                label_file = image_file

                # 读取图像和标签文件
                image = nib.load(f"{image_path}/{image_file}").get_fdata()
                label = nib.load(f"{label_path}/{label_file}").get_fdata()
                
                                
                # 选择切片范围
                image_slices = image[:, :, :]
                label_slices = label[:, :, :]
                
                # 归一化图像
                image_slices = normalize_image(image_slices)
                
                # 获取裁切的高度和宽度
                crop_size = self.slice_range_height

                # 确保裁切区域在图像的实际尺寸范围内
                for i in range(image_slices.shape[-1]):
                    h, w = image_slices[:, :, i].shape
                    if h > self.slice_range_height and w > self.slice_range_width:
                        img_slice = image_slices[:, :, i]
                        lbl_slice = label_slices[:, :, i]

                        # 计算中心点
                        center_h, center_w = img_slice.shape[0] // 2, img_slice.shape[1] // 2

                        # 计算裁切区域的边界
                        h_start = max(center_h - crop_size // 2, 0)
                        h_end = min(center_h + crop_size // 2, img_slice.shape[0])
                        w_start = max(center_w - crop_size // 2, 0)
                        w_end = min(center_w + crop_size // 2, img_slice.shape[1])

                        # 如果裁切区域超出了图像边界，需要在边界内调整
                        if h_end - h_start < crop_size:
                            if h_start == 0:
                                h_end = min(h_start + crop_size, img_slice.shape[0])
                            else:
                                h_start = max(h_end - crop_size, 0)

                        if w_end - w_start < crop_size:
                            if w_start == 0:
                                w_end = min(w_start + crop_size, img_slice.shape[1])
                            else:
                                w_start = max(w_end - crop_size, 0)

                        # 对每个切片进行裁切
                        img_slice = img_slice[h_start:h_end, w_start:w_end]
                        lbl_slice = lbl_slice[h_start:h_end, w_start:w_end]

                        slice_data = {"image": img_slice, "label": lbl_slice, "subset": subset}
                        # 筛选包含MVO的数据
                        if 4 in lbl_slice:
                            self.slices_mvo.append(slice_data)
                        else:
                            self.slices.append(slice_data)                        
                        
                    elif h < self.slice_range_height or w < self.slice_range_width:
                        # Padding
                        # 裁剪以确保尺寸不大于目标尺寸
                        if h > self.slice_range_height:
                            start_y = (h - self.slice_range_height) // 2
                            img_slice = img_slice[start_y:start_y + self.slice_range_height, :]
                            lbl_slice = lbl_slice[start_y:start_y + self.slice_range_height, :]
                        if w > self.slice_range_width:
                            start_x = (w - self.slice_range_width) // 2
                            img_slice = img_slice[:, start_x:start_x + self.slice_range_width]
                            lbl_slice = lbl_slice[:, start_x:start_x + self.slice_range_width]                        
                        h_c, w_c = img_slice.shape
                        pad_h = max(self.slice_range_height - h_c, 0)
                        pad_w = max(self.slice_range_width - w_c, 0)
                        pad_left = pad_w // 2
                        pad_right = pad_w - pad_left
                        pad_top = pad_h // 2
                        pad_bottom = pad_h - pad_top
                        img_slice = np.pad(img_slice, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
                        lbl_slice = np.pad(lbl_slice, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
                        
                        slice_data = {"image": img_slice, "label": lbl_slice, "subset": subset}
                        # 筛选包含MVO的数据
                        if 4 in lbl_slice:
                            self.slices_mvo.append(slice_data)
                        else:
                            self.slices.append(slice_data)
                            
        elif self.split == "valid":
            for sample in self.sample_list:
                image_path = sample[0]
                subset = sample[2]
                image_file = sample[3]
                label_path = image_path.replace("images", "labels")
                label_file = image_file

                # 读取图像和标签文件
                image = nib.load(f"{image_path}/{image_file}").get_fdata()
                label = nib.load(f"{label_path}/{label_file}").get_fdata()
                
                
                # 归一化图像
                image_volume = normalize_image(image)
                
                # 获取裁切的高度和宽度
                crop_size = self.slice_range_height
                h,w,d = image.shape
            
                if h > self.slice_range_height and w > self.slice_range_width:
                    img_volume = image_volume
                    lbl_volume = label

                    # 计算中心点
                    center_h, center_w = h // 2, w // 2

                    # 计算裁切区域的边界
                    h_start = max(center_h - crop_size // 2, 0)
                    h_end = min(center_h + crop_size // 2, h)
                    w_start = max(center_w - crop_size // 2, 0)
                    w_end = min(center_w + crop_size // 2, w)

                    # 如果裁切区域超出了图像边界，需要在边界内调整
                    if h_end - h_start < crop_size:
                        if h_start == 0:
                            h_end = min(h_start + crop_size, h)
                        else:
                            h_start = max(h_end - crop_size, 0)

                    if w_end - w_start < crop_size:
                        if w_start == 0:
                            w_end = min(w_start + crop_size, w)
                        else:
                            w_start = max(w_end - crop_size, 0)

                    # 对每个切片进行裁切
                    img_volume = img_volume[h_start:h_end, w_start:w_end, :]
                    lbl_volume = lbl_volume[h_start:h_end, w_start:w_end, :]

                    volume_data = {"image": img_volume, "label": lbl_volume, "subset": subset}
                    self.volumes.append(volume_data)
                    
                elif h < self.slice_range_height or w < self.slice_range_width:
                    # Padding
                    # 裁剪以确保尺寸不大于目标尺寸
                    img_volume = image_volume
                    lbl_volume = label
                    if h > self.slice_range_height:
                        start_y = (h - self.slice_range_height) // 2
                        img_volume = img_volume[start_y:start_y + self.slice_range_height, :, :]
                        lbl_volume = lbl_volume[start_y:start_y + self.slice_range_height, :, :]
                    if w > self.slice_range_width:
                        start_x = (w - self.slice_range_width) // 2
                        img_volume = img_volume[:, start_x:start_x + self.slice_range_width, :]
                        lbl_volume = lbl_volume[:, start_x:start_x + self.slice_range_width, :]                        
                    h_c, w_c, _ = img_volume.shape
                    pad_h = max(self.slice_range_height - h_c, 0)
                    pad_w = max(self.slice_range_width - w_c, 0)
                    pad_left = pad_w // 2
                    pad_right = pad_w - pad_left
                    pad_top = pad_h // 2
                    pad_bottom = pad_h - pad_top
                    img_volume = np.pad(img_volume, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
                    lbl_volume = np.pad(lbl_volume, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
                    
                    volume_data = {"image": img_volume, "label": lbl_volume, "subset": subset}
                    self.volumes.append(volume_data)
        random.shuffle(self.slices)
        random.shuffle(self.slices_mvo)
        
    def __len__(self):
        if self.split == "train":
            return len(self.slices)+len(self.slices_mvo)
        else:
            return len(self.volumes)

    def __getitem__(self, idx):
        if self.split == "train":
            if idx < len(self.slices_mvo): 
                sample = self.slices_mvo[idx]
            else:
                idx = idx - (len(self.slices_mvo)) 
                sample = self.slices[idx]
                
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        
        if self.split == "valid":        
            sample = self.volumes[idx]
        return sample




class BaseDataSets_Synapse(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train_slices.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            # # 计算要选择的数据数量（10%）
            # num_samples = len(self.sample_list)
            # num_samples_to_select = int(1 * num_samples)
            # # 随机选择数据
            # selected_samples = random.sample(self.sample_list, num_samples_to_select)
            # self.sample_list = selected_samples

        elif self.split == "val":
            with open(self._base_dir + "/val.txt", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = np.load(self._base_dir + "/train_npz/{}.npz".format(case))
        else:
            if self.split == "val":
                h5f = h5py.File(self._base_dir + "/test_vol_h5/{}.npy.h5".format(case))
            else:
                h5f = h5py.File(self._base_dir + "/test_vol_h5/{}.npy.h5".format(case))
                
        image = np.array(h5f["image"])
        label = np.array(h5f["label"])
        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        return sample

def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)  # 旋转0到3次90度
    image = np.rot90(image, k).copy()
    #axis = np.random.randint(0, 2)  # 随机选择水平或垂直翻转
    #image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k).copy()
        #label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)

def encode_labels(label):
    label_map = {0: 0, 1: 1, 2: 2, 3: 3, 4:3}
    encoded_label = np.zeros_like(label)
    for old_label, new_label in label_map.items():
        encoded_label[label == old_label] = new_label
    return encoded_label

def preprocess_labels(label):
    # 将第五类标签(label=4)重新编码为第四类标签(label=3)
    label[label == 4] = 3
    #label[label == 3] = 2
    return label

def normalize_image(image):
    """将图像数据归一化到 [0, 1]"""
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val > min_val:  # 避免除以零
        image = (image - min_val) / (max_val - min_val)
    else:
        # 如果max_val等于min_val，意味着图像中的所有像素值都相同
        image = np.zeros_like(image)
    return image

class CTATransform(object):
    def __init__(self, output_size, cta):
        self.output_size = output_size
        self.cta = cta

    def __call__(self, sample, ops_weak, ops_strong):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        to_tensor = transforms.ToTensor()

        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        # apply augmentations
        image_weak = augmentations.cta_apply(transforms.ToPILImage()(image), ops_weak)
        image_strong = augmentations.cta_apply(image_weak, ops_strong)
        label_aug = augmentations.cta_apply(transforms.ToPILImage()(label), ops_weak)
        label_aug = to_tensor(label_aug).squeeze(0)
        label_aug = torch.round(255 * label_aug).int()

        sample = {
            "image_weak": to_tensor(image_weak),
            "image_strong": to_tensor(image_strong),
            "label_aug": label_aug,
        }
        return sample

    def cta_apply(self, pil_img, ops):
        if ops is None:
            return pil_img
        for op, args in ops:
            pil_img = OPS[op].f(pil_img, *args)
        return pil_img

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample


class WeakStrongAugment(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        # weak augmentation is rotation / flip
        image_weak, label = random_rot_flip(image, label)
        # strong augmentation is color jitter
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
