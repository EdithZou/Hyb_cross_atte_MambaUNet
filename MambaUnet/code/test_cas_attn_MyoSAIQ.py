import argparse
import os
import shutil
import csv
import pandas as pd
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
torch.cuda.empty_cache()
torch.cuda.set_device(1)
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm

from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', type=str,
                    default='/home/tzou/MyoSAIQ/nnUnet/MYOSAIQ_data_test.csv', help='Path to the CSV file with test data')
parser.add_argument('--exp', type=str,
                    default='MyoSAIQ/Cascade_Fully_Supervised_ViM', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='mambaunet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=3,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=716,
                    help='labeled data')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    return dice

def calculate_metric_percase_train(pred, gt):
    if np.sum(gt) == 0:
        return 0, 0
    # 计算除了背景以外的部分标签的dice
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0


def load_case_paths(csv_path):
    cases = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            folder_path, filetype, subset, filename = row
            case_id = f"{filename.split('.')[0]}"
            if case_id not in cases:
                cases[case_id] = {'image':None, 'label':None}
            if 'image' in filetype:
                cases[case_id]['image'] = os.path.join(folder_path, filename)
                label_path = folder_path.replace('images', 'labels')
                cases[case_id]['label'] = os.path.join(label_path, filename)
    return cases

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

def test_single_volume(case_id, case_paths, net2, net3, test_save_path, FLAGS):
    image_path = case_paths['image']
    #label_path = case_paths['label']
    image_nib = nib.load(image_path)
    image = image_nib.get_fdata()
    #label = nib.load(label_path).get_fdata()
    prediction = np.zeros_like(image)
    #first_metric = []
    #second_metric = []
    #third_metric = []
    #fourth_metric = []
    for ind in range(image.shape[2]):
        slice = image[:, :, ind]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (96 / x, 96 / y), order=0)
        input = normalize_image(slice)
        input = torch.from_numpy(input).unsqueeze(0).unsqueeze(0).float().cuda()
        #net1.eval()
        net2.eval()
        net3.eval()
        with torch.no_grad():
            #output1 = net1(input)
            #s_output1 = torch.softmax(output1, dim=1)
            #s2_, output2 = net2(torch.concat((input, s_output1[:,-1,:,:].unsqueeze(1)), dim=1))
            s2_, output2 = net2(input)
            s_output2 = torch.softmax(output2, dim=1)
            s3_, output3 = net3(torch.concat((input, s_output2[:,2:,:,:]), dim=1))
            if 'M' in case_id:
                out = torch.argmax(torch.softmax(
                    output2, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(out, (x / 96, y / 96), order=0)
                prediction[:,:,ind] = pred
            if 'D' in case_id:
                out = torch.argmax(torch.softmax(
                    output3, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(out, (x / 96, y / 96), order=0)
                prediction[:,:,ind] = pred                

    #first_metric = calculate_metric_percase(prediction == 1, label == 1)
    #second_metric = calculate_metric_percase(prediction == 2, label == 2)
    #third_metric = calculate_metric_percase(prediction == 3, label == 3)
    #fourth_metric = calculate_metric_percase(prediction == 4, label == 4)
    
    #保存metric到CSV文件
    #csv_path = os.path.join(test_save_path, "metrics.csv")
    #with open(csv_path, mode='a', newline='') as file:
    #    writer = csv.writer(file)
    #    writer.writerow([case_id, first_metric, second_metric, third_metric, fourth_metric])
            
    image = torch.from_numpy(image).permute(2,1,0)
    prediction = torch.from_numpy(prediction).permute(2,1,0)
    #label = torch.from_numpy(label).permute(2,1,0)
    image = image.cpu().detach().numpy()
    prediction = prediction.cpu().detach().numpy()
    #label = label.cpu().detach().numpy()
    
    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.int16))
    #lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    #lab_itk.SetSpacing((1.2, 1.2, 5))
    sitk.WriteImage(prd_itk, os.path.join(test_save_path, f"{case_id}_pred.nii.gz"))
    sitk.WriteImage(img_itk, os.path.join(test_save_path, f"{case_id}_img.nii.gz"))
    #sitk.WriteImage(lab_itk, os.path.join(test_save_path, f"{case_id}_gt.nii.gz"))
    #print('Testing Finished!')
    return 0
    #return first_metric, second_metric, third_metric, fourth_metric


def Inference(FLAGS):
    cases = load_case_paths(FLAGS.csv_path)
    snapshot_path = "/home/tzou/Mamba_model/{}_{}_3_stage_attn/{}".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)    
    #snapshot_path = os.path.join("../model", f"{FLAGS.exp}_{FLAGS.labeled_num}_labeled", FLAGS.model)
    test_save_path = os.path.join(snapshot_path, f"{FLAGS.model}_predictions")
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    
    #net1 = net_factory(net_type='ViM_Seg', in_chns=3, class_num=FLAGS.num_classes)
    net2 = net_factory(net_type='hyb_VIM_att', in_chns=3, class_num=FLAGS.num_classes+1)
    net3 = net_factory(net_type='hyb_VIM_att', in_chns=3, class_num=FLAGS.num_classes+2)
    
    #save_mode_path_s1 = os.path.join(snapshot_path, f"{FLAGS.model}_best_model_s1.pth")
    save_mode_path_s2 = os.path.join(snapshot_path, f"{FLAGS.model}_best_model_s2.pth")
    save_mode_path_s3 = os.path.join(snapshot_path, f"{FLAGS.model}_best_model_s3.pth")
    
    #net1.load_state_dict(torch.load(save_mode_path_s1))
    net2.load_state_dict(torch.load(save_mode_path_s2))
    net3.load_state_dict(torch.load(save_mode_path_s3))
    #print("stage1 model init weight from {}".format(save_mode_path_s1))
    print("stage2 model init weight from {}".format(save_mode_path_s2))
    print("stage3 model init weight from {}".format(save_mode_path_s3))
    #net1.eval()
    net2.eval()
    net3.eval()
    
    #first_total = 0.0
    #second_total = 0.0
    #third_total = 0.0
    #fourth_total = 0.0
    
    for case_id, case_paths in tqdm(cases.items()):
        if case_id !='filename':
            #first_metric, second_metric, third_metric, fourth_metric = test_single_volume(case_id, case_paths, net2, 
            #                                                                              net3, test_save_path, FLAGS)    
            test_single_volume(case_id, case_paths, net2, net3, test_save_path, FLAGS)  
            #first_total += np.asarray(first_metric)
            #second_total += np.asarray(second_metric)
            #third_total += np.asarray(third_metric)
            #fourth_total += np.asarray(fourth_metric)
    #avg_metric = [first_total / len(cases), second_total / len(cases), third_total / len(cases), fourth_total / len(cases)]
    #return avg_metric
    return 0

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    #print('result is: \n')
    #print(metric)
    #print('LV, myo, MI, MVO result is:')
    #print((metric[0], metric[1], metric[2], metric[3]))
