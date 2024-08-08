import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom

    
def calculate_metric_percase(pred, gt):
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



def test_single_volume(image, label, net1, net2, net3, classes, patch_size=[96, 96]):
    # 将图像移到CPU进行验证
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    
    s1_prediction = np.zeros_like(label)
    s2_prediction = np.zeros_like(label)
    s3_prediction = np.zeros_like(label)
    
    for ind in range(image.shape[2]):
        slice = image[:, :,ind]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net1.eval()
        net2.eval()
        net3.eval()
        
        #dice_per_slice = []
        with torch.no_grad():
            out1 = net1(input)
            s1_out = torch.argmax(torch.softmax(out1, dim=1), dim=1).squeeze(0)
            s1_out = s1_out.cpu().detach().numpy()
            
            out2, out2_dice, out2_edge = net2(torch.concat((input, torch.softmax(out1, dim=1)[:,-1,:,:].unsqueeze(1)), dim=1))         
            #out2, out2_dice = net2(input)      
            s2_out = torch.argmax(torch.softmax(out2, dim=1), dim=1, keepdim=True).squeeze(0)
            s2_out = s2_out.cpu().detach().numpy()

            out3, out3_dice, out3_edge = net3(torch.concat((input, torch.softmax(out2, dim=1)[:,2:,:,:]), dim=1))         
            s3_out = torch.argmax(torch.softmax(out3, dim=1), dim=1, keepdim=True).squeeze(0)
            s3_out = s3_out.cpu().detach().numpy()
            
            #s1_pred = zoom(s1_out, (x / patch_size[0], y / patch_size[1]), order=0)
            #s1_prediction[:,:,ind] = s1_pred
            
            s2_pred = zoom(s2_out, (x / patch_size[0], y / patch_size[1]), order=0)
            s2_prediction[:,:,ind] = s2_pred
            
            s3_pred = zoom(s3_out, (x / patch_size[0], y / patch_size[1]), order=0)
            s3_prediction[:,:,ind] = s3_pred   
        
    metric_list = []
    for i in range(1, classes):
        metric_s3 = calculate_metric_percase(s3_prediction== i, label == i)
        if metric_s3 is not None:
            metric_list.append(metric_s3)                 
    return metric_list



def test_single_volume_ds(image, label, net, classes, patch_size=[96, 96]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
