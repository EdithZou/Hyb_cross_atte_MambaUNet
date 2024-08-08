# d = sum(A,2); #以矩阵A的每一行为对象，对一行内的数字求和,返回一个列向量：degree vector??
# for ilab = 1 : num_lab #遍历每一张图片的label
#     X_t = double(segmentation2D(:) == ilab) #将分割图片展平成一维向量，数据类型为double
#     gradient_bk = X_t' * A * X_t / (d' * X_t)^2 * d - A * X_t * 2 / (d' * X_t); #A：W配对矩阵，X_t:网络输出分割结果一维向量
#     Ct(:,:,ilab) = reshape(gradient bk, height, width) ;
# end




from torch.autograd import Variable

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("../wrapper/bilateralfilter/build/lib.linux-x86_64-3.6")
from bilateralfilter import bilateralfilter, bilateralfilter_batch
import time
from multiprocessing import Pool
import multiprocessing
from itertools import repeat
import pickle


class NCFunction(Function):

    @staticmethod
    def forward(ctx, images, segmentations, sigma_rgb, sigma_xy, ROIs):
        img = images
        ctx.save_for_backward(segmentations)
        ctx.N, ctx.K, ctx.H, ctx.W = segmentations.shape

        ROIs = ROIs.unsqueeze_(1).repeat(1,ctx.K,1,1)
        segmentations = torch.mul(segmentations.cuda(), ROIs.cuda())
        ctx.ROIs = ROIs

        normalizedcut_loss = 0.0
        images = images.cpu().numpy().flatten()
        segmentations = segmentations.cpu().numpy().flatten()
        # 图之间的边
        AS = np.zeros(segmentations.shape, dtype=np.float32)
        bilateralfilter_batch(images, segmentations, AS, ctx.N, ctx.K, ctx.H, ctx.W, sigma_rgb, sigma_xy)
        
        # 图内部的边
        d = np.ones(segmentations.shape, dtype=np.float32)
        ones = np.ones(segmentations.shape, dtype=np.float32)
        bilateralfilter_batch(images, ones, d, ctx.N, ctx.K, ctx.H, ctx.W, sigma_rgb, sigma_xy)


        normalizedcut_loss -= np.dot(segmentations, AS)/(np.dot(d, segmentations) + 1e-7)
                                                                
        normalizedcut_loss /= ctx.N       
        
        ctx.AS = np.reshape(AS, (ctx.N, ctx.K, ctx.H, ctx.W))
        ctx.d = np.reshape(d, (ctx.N, ctx.K, ctx.H, ctx.W))
        ctx.segmentations = segmentations
        return Variable(torch.tensor([normalizedcut_loss]).to(device=img.device), requires_grad=True)       

    @staticmethod
    def backward(ctx, grad_output):
        tem = np.dot(np.dot(ctx.segmentations, torch.from_numpy(ctx.AS)), ctx.d)
        tem /= np.dot(ctx.d, ctx.segmentations)**2 + 1e-7
        tem += -2*torch.from_numpy(ctx.AS)/(np.dot(ctx.d, ctx.segmentations) + 1e-7)
        grad_segmentation = tem/ctx.N 
        grad_segmentation =  grad_output*((np.dot(np.mul(torch.from_numpy(ctx.AS),ctx.segmentations)),(ctx.d))+1e-5)   \
            /ctx.N/((np.dot(ctx.segmentations,ctx.d))**2 + 1e-5)  \
            -2*(grad_output*torch.from_numpy(ctx.AS)+1e-5)/ctx.N/((np.dot(ctx.segmentations,ctx.d))+1e-5)
        grad_segmentation = grad_segmentation.cuda()
        grad_segmentation = torch.mul(grad_segmentation, ctx.ROIs.cuda())
        return None, grad_segmentation, None, None, None


class NCLoss(nn.Module):
    def __init__(self, weight, sigma_rgb, sigma_xy, scale_factor):
        super(NCLoss, self).__init__()
        self.weight = weight
        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy
        self.scale_factor = scale_factor
    def forward(self, images, segmentations, ROIs):
        scaled_images = F.interpolate(images,scale_factor=self.scale_factor) 
        scaled_segs = F.interpolate(segmentations,scale_factor=self.scale_factor,mode='bilinear',align_corners=False)
        scaled_ROIs = F.interpolate(ROIs.unsqueeze(1),scale_factor=self.scale_factor).squeeze(1)
        # print(scaled_images.sum(),scaled_segs(),scaled_ROIs())
        return self.weight*NCFunction.apply(
                Variable(scaled_images), Variable(scaled_segs), self.sigma_rgb, self.sigma_xy*self.scale_factor, Variable(scaled_ROIs))
        # return self.weight*NCFunction.apply(
        #         scaled_images, scaled_segs, self.sigma_rgb, self.sigma_xy*self.scale_factor, scaled_ROIs)
    def extra_repr(self):
        return 'sigma_rgb={}, sigma_xy={}, weight={}, scale_factor={}'.format(
            self.sigma_rgb, self.sigma_xy, self.weight, self.scale_factor)
                                                                 
