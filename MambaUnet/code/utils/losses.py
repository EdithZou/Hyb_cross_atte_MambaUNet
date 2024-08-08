import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch import einsum
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from scipy.ndimage import distance_transform_edt as distance
import numpy as np


# from metrics import dice_coef
# from metrics import dice
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")



def ConstraLoss(inputs, targets):

    m=nn.AdaptiveAvgPool2d(1)
    input_pro = m(inputs)
    input_pro = input_pro.view(inputs.size(0),-1) #N*C
    targets_pro = m(targets)
    targets_pro = targets_pro.view(targets.size(0),-1)#N*C
    input_normal = nn.functional.normalize(input_pro,p=2,dim=1) # 正则化
    targets_normal = nn.functional.normalize(targets_pro,p=2,dim=1)
    res = (input_normal - targets_normal)
    res = res * res
    loss = torch.mean(res)
    return loss

    
def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def entropy_loss(p, C=2):
    # p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / \
        torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent


def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                       keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss


def softmax_kl_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_log_softmax = torch.log(torch.sigmoid(input_logits))
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='mean')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        #print(f"input_tensor max value: {input_tensor.max().item()}, min value: {input_tensor.min().item()}")

        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        
        loss = 0.0
        for i in range(1, self.n_classes):
            dice = self._dice_loss(inputs[:, i, :, :], target[:, i,:, :])
            class_wise_dice.append(dice.item())
            loss += dice * weight[i]
        total_loss = loss / (self.n_classes - 1)
        return total_loss


import cv2
class GradientLoss(nn.Module):
    def __init__(self, num_classes=5, operator="Sobel", channel_mean=True, loss='dice',use_opencv=False,sigma=5.0):
        r"""
       :param operator: in ['Sobel', 'Prewitt','Roberts','Scharr']
       :param channel_mean: 是否在通道维度上计算均值
       """
        super(GradientLoss, self).__init__()
        assert operator in ['Sobel', 'Prewitt', 'Roberts', 'Scharr'], "Unsupported operator"
        self.num_classes = num_classes
        self.channel_mean = channel_mean
        self.operators = {
            "Sobel": {
                'x': torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float).cuda(),
                'y': torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float).cuda()
            },
            "Prewitt": {
                'x': torch.tensor([[[[1, 0, -1], [1, 0, -1], [1, 0, -1]]]], dtype=torch.float).cuda(),
                'y': torch.tensor([[[[-1, -1, -1], [0, 0, 0], [1, 1, 1]]]], dtype=torch.float).cuda()
            },
            "Roberts": {
                'x': torch.tensor([[[[1, 0], [0, -1]]]], dtype=torch.float).cuda(),
                'y': torch.tensor([[[[0, -1], [1, 0]]]], dtype=torch.float).cuda()
            },
            "Scharr": {
                'x': torch.tensor([[[[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]]], dtype=torch.float).cuda(),
                'y': torch.tensor([[[[-3, 10, -3], [0, 0, 0], [3, 10, 3]]]], dtype=torch.float).cuda()
            },
        }
        self.op_x = self.operators[operator]['x']
        self.op_y = self.operators[operator]['y']
        self.diceloss=DiceLoss(n_classes=self.num_classes)
        #self.ssim_loss=SSIM(di=True)
        self.loss=loss
        self.use_opencv=False
        self.sigma=sigma
 
    def gradients(self, img_tensor):
        """
        img_tensor:n*4*h*w
        """
        grad_img=torch.zeros_like(img_tensor).detach().cpu().numpy()
        if self.use_opencv:
            img_tensor=img_tensor.detach().cpu().numpy().astype('uint8')*255
            for i in img_tensor.shape[0]:
                for k in range(4):
                    grad_img[i][k]=cv2.Canny(img_tensor[i][k],100,100)
            grad_img[grad_img>0]=1
            grad_img=torch.tensor(grad_img).cuda().float()
            return grad_img

        else:
            op_x, op_y = self.op_x, self.op_y
            if self.channel_mean:
                img_tensor = img_tensor.mean(dim=1, keepdim=True)
                groups = 1
            else:
                groups = img_tensor.shape[1]
                op_x = op_x.repeat(groups, 1, 1, 1)
                op_y = op_y.repeat(groups, 1, 1, 1)
            grad_x = F.conv2d(img_tensor, op_x, groups=groups)
            grad_y = F.conv2d(img_tensor, op_y, groups=groups)
            grad_img=grad_x.abs()+grad_y.abs()
            grad_img[grad_img>0]=1.
            return grad_img

    def boundary_loss(self,grad_gt,edge):
        # the boundary loss of class K
        # grad_gt:Tensor B*H*W ,edge:Tensor B*H*W  in Class K return the class K's loss
        B=grad_gt.shape[0]
        loss_gradK=0
        for i in range(B):
            dist_gt=torch.tensor(np.exp(-distance(~grad_gt[i].detach().cpu().numpy().astype('bool'))**2/(self.sigma**2))).cuda()
            loss_gradK+=1-(edge[i]*dist_gt).sum()/torch.max(edge[i].sum(),grad_gt[i].sum())
        return loss_gradK/B

 
    def forward(self, edge, img_gt): # edge:B*n*h*w  img:B*(n+1)*h*w
        img=F.one_hot(img_gt.long(),num_classes=self.num_classes).permute(0,3,1,2).float()
        grad_img=self.gradients(img)
        grad_img=F.interpolate(grad_img,img.shape[-1],mode='nearest')
        # print(edge.shape,grad_img.shape)
        score=0
        loss_sum=0
        for i in range(self.num_classes):
            if self.loss=='boundary':
                edge_i=F.softmax(edge[:,2*i:2*(i+1),...],dim=1)[:,1,...]
                loss_sum+=self.boundary_loss(grad_img[:,i],edge_i)
            elif self.loss=='dice':
                edge_i=torch.round(F.softmax(edge[:,2*i:2*(i+1),...],dim=1)[:,1,...].unsqueeze(1))

                # loss_sum+=(1-self.ssim_loss(edge_i,grad_img[:,i,...].unsqueeze(1)))
                loss_sum+=(1-(2*edge_i*grad_img[:,i,...]).sum()/((edge_i+grad_img[:,i,...]).sum()+1e-10))
            

        return loss_sum/self.num_classes


class SurfaceLoss:
    def __init__(self, idc, num_classes):
        self.idc = idc
        self.num_classes = num_classes
        print(f"Initialized {self.__class__.__name__} with idc={idc}")

    def __call__(self, probs, target):
        # Ensure the probabilities are in the simplex (valid probabilities)
        assert simplex(probs), "Probabilities are not in simplex"
        
        # Ensure the target is not one-hot encoded (ground truth)
        assert not one_hot(target), "Target should not be one-hot encoded"

        # Convert target to one-hot encoding
        one_hot_target = self._one_hot_encoder(target)

        # Compute distance maps from one-hot encoded target
        distance_map = self._compute_distance_map(one_hot_target)

        # Extract the relevant channels based on idc
        pc = probs[:, self.idc, ...].type(torch.float32).unsqueeze(1)
        dc = distance_map[:, self.idc, ...].type(torch.float32).unsqueeze(1)

        # Calculate the loss
        multipled = einsum("bkwh,bkwh->bkwh", pc, dc)
        loss = multipled.mean()

        return loss

    def _one_hot_encoder(self, input_tensor):
        #print(f"input_tensor max value: {input_tensor.max().item()}, min value: {input_tensor.min().item()}")

        tensor_list = []
        for i in range(self.num_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _compute_distance_map(self, one_hot_target):
        # Convert one-hot encoded target to distance maps
        B, C, H, W = one_hot_target.shape
        dist_map = np.zeros((B, C, H, W), dtype=np.float64)
        
        for b in range(B):
            dist_map[b] = self._one_hot2dist(one_hot_target[b].cpu().numpy())
        
        return torch.tensor(dist_map, dtype=torch.float32, device=one_hot_target.device)

    def _one_hot2dist(self, seg: np.ndarray) -> np.ndarray:
        C, H, W = seg.shape
        res = np.zeros_like(seg, dtype=np.float64)
        
        for c in range(C):
            posmask = seg[c].astype(bool)
            
            if posmask.any():
                negmask = ~posmask
                res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
        
        return res

def simplex(tensor) -> bool:
    """ Check if the tensor represents a valid probability distribution (simplex) """
    return torch.allclose(tensor.sum(dim=1), torch.ones_like(tensor[:, 0, :, :]))

def one_hot(tensor) -> bool:
    """ Check if the tensor is one-hot encoded """
    return tensor.dim() > 1 and tensor.size(1) > 1 and (tensor.sum(dim=1) == 1).all()
    

class WeightedFocalLoss(nn.Module):
    "Weighted version of Focal Loss"    
    def __init__(self, n_classes, alpha=.25, gamma=2, weight=None):
            super(WeightedFocalLoss, self).__init__()        
            self.alpha = [0.1, 0.1, 0.2, 0.3, 0.3]
            self.gamma = gamma
            self.weight = weight
            self.n_classes = n_classes
            self.bce_loss = BCEWithLogitsLoss(weight=self.weight)
    
    def _one_hot_encoder(self, input_tensor):
        #print(f"input_tensor max value: {input_tensor.max().item()}, min value: {input_tensor.min().item()}")
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, input, target, softmax=True):
        if softmax:
            input_n = torch.softmax(input, dim=1)
        target = self._one_hot_encoder(target)
        assert input_n.size() == target.size(), 'predict & target shape do not match'
        class_wise_focal_loss = []
        loss = 0.0
        for i in range(1, self.n_classes): 
            CE_loss = self.bce_loss(input[:, i, :, :], target[:, i,:, :]) 
            pt = torch.exp(-CE_loss)        
            F_loss = 1 * (1-pt)**self.gamma * CE_loss
            class_wise_focal_loss.append(F_loss.item())
            loss += F_loss * self.alpha[i]
        return loss



def entropy_minmization(p):
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)
    ent = torch.mean(y1)

    return ent


def entropy_map(p):
    ent_map = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                           keepdim=True)
    return ent_map


def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1),
                      F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1),
                      F.softmax(p, dim=-1), reduction='none')

    # Using function "sum" and "mean" are depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss


###############################################
# BCE = torch.nn.BCELoss()

def weighted_loss(pred, mask):
    BCE = torch.nn.BCELoss(reduction = 'none')
    
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask).float()
    wbce = BCE(pred, mask)
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    
    return (wbce + wiou).mean()  



def calc_loss(pred, target, bce_weight=0.5):
    bce = weighted_loss(pred, target)
    # dl = 1 - dice_coef(pred, target)
    # loss = bce * bce_weight + dl * bce_weight

    return bce


def loss_sup(logit_S1, logit_S2, labels_S1, labels_S2):
    loss1 = calc_loss(logit_S1, labels_S1)
    loss2 = calc_loss(logit_S2, labels_S2)

    return loss1 + loss2



def loss_diff(u_prediction_1, u_prediction_2, batch_size):
    a = weighted_loss(u_prediction_1, Variable(u_prediction_2, requires_grad=False))
#     print('a',a.size())
    a = a.item()

    b = weighted_loss(u_prediction_2, Variable(u_prediction_1, requires_grad=False))
    b = b.item()

    loss_diff_avg = (a + b)
#     print('loss_diff_avg',loss_diff_avg)
#     print('loss_diff batch size',batch_size)
#     return loss_diff_avg / batch_size
    return loss_diff_avg 



###############################################
#contrastive_loss

class ConLoss(torch.nn.Module):
#for unlabel data
    def __init__(self, temperature=0.07, base_temperature=0.07):
        """
        Contrastive Learning for Unpaired Image-to-Image Translation
        models/patchnce.py
        """
        super(ConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.nce_includes_all_negatives_from_minibatch = False
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
#         self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction = 'none')
        self.mask_dtype = torch.bool
#         self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]
#         width = feat_q.shape[2]
        feat_q = feat_q.view(batch_size, dim, -1).permute(0, 2, 1)  #batch * dim * np  # batch * np * dim
        feat_k = feat_k.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_q = F.normalize(feat_q, dim=-1, p=1)
        feat_k = F.normalize(feat_k, dim=-1, p=1)
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.reshape(-1, 1, dim), feat_k.reshape(-1, dim, 1))  #(batch * np) * 1 * dim #(batch * np) * dim * 1  #(batch * np) * 1
        l_pos = l_pos.view(-1, 1) #(batch * np) * 1

        # neg logit
        if self.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = batch_size

        # reshape features to batch size
        feat_q = feat_q.reshape(batch_dim_for_bmm, -1, dim)  #batch * np * dim
        feat_k = feat_k.reshape(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))  # batch * np * np

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]

        l_neg_curbatch.masked_fill_(diagonal, -float('inf'))
        l_neg = l_neg_curbatch.view(-1, npatches)  #(batch * np) * np

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature  #(batch * np) * (np+1)

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss
    
    
# class MocoLoss(torch.nn.Module):
# #for unlabel data
#     def __init__(self, temperature=0.07):

#         super(MocoLoss, self).__init__()
#         self.temperature = temperature
#         self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

#     def forward(self, feat_q, feat_k, queue):
#         assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
#         feat_q = F.normalize(feat_q, dim=-1, p=1)
#         feat_k = F.normalize(feat_k, dim=-1, p=1)
#         batch_size = feat_q.shape[0]
#         dim = feat_q.shape[1]
#         K = len(queue)
# #         print('K',K)

#         feat_k = feat_k.detach()

#         # pos logit
#         l_pos = torch.bmm(feat_q.view(batch_size,1,dim),feat_k.view(batch_size,dim,1))  #batch_size * 1
#         l_pos = l_pos.view(-1, 1)
#         feat_k = feat_k.transpose(0,1)
# #         print('feat_k',feat_k.size())
#         # neg logit
#         if K == 0:
#             l_neg = torch.mm(feat_q.view(batch_size,dim), feat_k)
#         else:
            
#             queue_tensor = torch.cat(queue,dim = 1)
# #             print('queue_tensor.size()',queue_tensor.size())
        
#             l_neg = torch.mm(feat_q.view(batch_size,dim), queue_tensor) #batch_size * K
# #         print(l_pos.size())
# #         print(l_neg.size())

#         out = torch.cat((l_pos, l_neg), dim=1) / self.temperature  #batch_size * (K+1)
        
# #         print(1)
        
#         loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
#                                                         device=feat_q.device))
# #         print(2)
        
#         queue.append(feat_k)
        
#         if K >= 10:
#             queue.pop(0)

#         return loss,queue
    
    
class contrastive_loss_sup(torch.nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        """
        Contrastive Learning for Unpaired Image-to-Image Translation
        models/patchnce.py
        """
        super(contrastive_loss_sup, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.nce_includes_all_negatives_from_minibatch = False
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
#         self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction = 'none')
        self.mask_dtype = torch.bool
#         self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]
#         width = feat_q.shape[2]
        feat_q = feat_q.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_k = feat_k.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_q = F.normalize(feat_q, dim=-1, p=1)
        feat_k = F.normalize(feat_k, dim=-1, p=1)
        feat_k = feat_k.detach()

        # pos logit
#         l_pos = torch.zeros((batch_size*2304,1)).cuda()
#         l_pos = torch.zeros((batch_size*1024,1)).cuda()
#         l_pos = torch.zeros((batch_size*784,1)).cuda()
        # neg logit
        if self.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = batch_size

        # reshape features to batch size
        feat_q = feat_q.reshape(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.reshape(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]

        l_neg_curbatch.masked_fill_(diagonal, -float('inf'))
        l_neg = l_neg_curbatch.view(-1, npatches)
        l_pos = torch.zeros((l_neg.size(0),1)).cuda()
        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss
    
def info_nce_loss(feats1,feats2):
#     imgs, _ = batch
#     imgs = torch.cat(imgs, dim=0)

    # Encode all images
#     feats = self.convnet(imgs)
    # Calculate cosine similarity
    cos_sim = F.cosine_similarity(feats1[:,None,:], feats2[None,:,:], dim=-1)
    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    # Find positive example -> batch_size//2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
    # InfoNCE loss
    cos_sim = cos_sim / 0.07
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll = nll.mean()

    # Logging loss
#     self.log(mode+'_loss', nll)
    # Get ranking position of positive example
#     comb_sim = torch.cat([cos_sim[pos_mask][:,None],  # First position positive example
#                               cos_sim.masked_fill(pos_mask, -9e15)],
#                              dim=-1)
#     sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
#     # Logging ranking metrics
#     self.log(mode+'_acc_top1', (sim_argsort == 0).float().mean())
#     self.log(mode+'_acc_top5', (sim_argsort < 5).float().mean())
#     self.log(mode+'_acc_mean_pos', 1+sim_argsort.float().mean())

    return nll

class contrastive_loss_sup(torch.nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        """
        Contrastive Learning for Unpaired Image-to-Image Translation
        models/patchnce.py
        """
        super(contrastive_loss_sup, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.nce_includes_all_negatives_from_minibatch = False
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
#         self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction = 'none')
        self.mask_dtype = torch.bool
#         self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]
#         width = feat_q.shape[2]
        feat_q = feat_q.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_k = feat_k.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_q = F.normalize(feat_q, dim=-1, p=1)
        feat_k = F.normalize(feat_k, dim=-1, p=1)
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.reshape(-1, 1, dim), feat_k.reshape(-1, dim, 1))  
        l_pos = l_pos.view(-1, 1) 
        # neg logit
        if self.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = batch_size

        # reshape features to batch size
        feat_q = feat_q.reshape(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.reshape(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]

        l_neg_curbatch.masked_fill_(diagonal, -float('inf'))
        l_neg = l_neg_curbatch.view(-1, npatches)
#         l_pos = torch.zeros((l_neg.size(0),1)).cuda()
        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss

class MocoLoss(torch.nn.Module):
    def __init__(self, temperature=0.07, use_queue = True, max_queue = 1):

        super(MocoLoss, self).__init__()
        self.temperature = temperature
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.use_queue = use_queue
        self.mask_dtype = torch.bool
        self.queue = OrderedDict()
        self.idx_list = []
        self.max_queue = max_queue

    def forward(self, feat_q, feat_k, idx):
        num_enqueue = 0
        num_update = 0
        num_dequeue = 0
        mid_pop = 0
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
        dim = feat_q.shape[1]
        batch_size = feat_q.shape[0]
        feat_q = feat_q.reshape(batch_size,-1)  
        feat_k = feat_k.reshape(batch_size,-1)

        K = len(self.queue)
#         print(K)

        feat_k = feat_k.detach()

        # pos logit
        l_pos = F.cosine_similarity(feat_q,feat_k,dim=1)        
        l_pos = l_pos.view(-1, 1)

        # neg logit
        if K == 0 or not self.use_queue:
            l_neg = F.cosine_similarity(feat_q[:,None,:], feat_k[None,:,:], dim=-1)
        else:
            for i in range(0,batch_size):
                if str(idx[i].item()) in self.queue.keys():
                    self.queue.pop(str(idx[i].item()))
                    mid_pop += 1
            queue_tensor = torch.cat(list(self.queue.values()),dim = 0)
            l_neg = F.cosine_similarity(feat_q[:,None,:], queue_tensor.reshape(-1,feat_q.size(1))[None,:,:], dim=-1)

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature  #batch_size * (K+1)
        
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        
        if self.use_queue:
            for i in range(0,batch_size):
                if str(idx[i].item()) not in self.queue.keys():
                    self.queue[str(idx[i].item())] = feat_k[i].clone()[None,:]
                    num_enqueue += 1
                else:
                    self.queue[str(idx[i].item())] = feat_k[i].clone()[None,:]
                    num_update += 1
                if len(self.queue) >= 1056 + 1:
                    self.queue.popitem(False)

                    num_dequeue += 1

#         print('queue length, mid pop, enqueue, update queue, dequeue: ', len(self.queue), mid_pop, num_enqueue, num_update, num_dequeue)

        return loss

class ConLoss_queue(torch.nn.Module):
#for unlabel data
    def __init__(self, temperature=0.07, use_queue = True, max_queue = 1):
        """
        Contrastive Learning for Unpaired Image-to-Image Translation
        models/patchnce.py
        """
        super(ConLoss_queue, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mask_dtype = torch.bool
        self.queue = OrderedDict()
        self.idx_list = []
        self.max_queue = max_queue


    def forward(self, feat_q, feat_k):
        num_enqueue = 0
        num_update = 0
        num_dequeue = 0
        mid_pop = 0
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]
#         width = feat_q.shape[2]
        feat_q = feat_q.view(batch_size, dim, -1).permute(0, 2, 1)  #batch * dim * np  # batch * np * dim
        feat_k = feat_k.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_q = F.normalize(feat_q, dim=-1, p=1)
        feat_k = F.normalize(feat_k, dim=-1, p=1)
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.reshape(-1, 1, dim), feat_k.reshape(-1, dim, 1))  #(batch * np) * 1 * dim #(batch * np) * dim * 1  #(batch * np) * 1
        l_pos = l_pos.view(-1, 1) #(batch * np) * 1

        # neg logit

        # reshape features to batch size
        feat_q = feat_q.reshape(batch_size, -1, dim)  #batch * np * dim
        feat_k = feat_k.reshape(batch_size, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))  # batch * np * np

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]

        l_neg_curbatch.masked_fill_(diagonal, -float('inf'))
        l_neg = l_neg_curbatch.view(-1, npatches)  #(batch * np) * np

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature  #(batch * np) * (np+1)

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss
    

class MocoLoss_list(torch.nn.Module):
    def __init__(self, temperature=0.07, use_queue = True):

        super(MocoLoss_list, self).__init__()
        self.temperature = temperature
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.use_queue = use_queue
        self.queue = []
        self.mask_dtype = torch.bool
        self.idx_list = []

    def forward(self, feat_q, feat_k, idx):
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
        dim = feat_q.shape[1]
        batch_size = feat_q.shape[0]
        feat_q = feat_q.reshape(batch_size,-1)  #转成向量
        feat_k = feat_k.reshape(batch_size,-1)

        K = len(self.queue)
#         print('K',K)

        feat_k = feat_k.detach()

        # pos logit
        l_pos = F.cosine_similarity(feat_q,feat_k,dim=1)        
        l_pos = l_pos.view(-1, 1)

        # neg logit
        if K == 0 or not self.use_queue:
            l_neg = F.cosine_similarity(feat_q[:,None,:], feat_k[None,:,:], dim=-1)
        else:            
            queue_tensor = torch.cat(self.queue,dim = 0)
            print(queue_tensor.size())
            l_neg = F.cosine_similarity(feat_q[:,None,:], queue_tensor.reshape(-1,feat_q.size(1))[None,:,:], dim=-1)

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature  #batch_size * (K+1)
        
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        if self.use_queue:
            self.queue.append(feat_k.clone())
#             for i in range(0,24):
#                 if idx[i] not in self.idx_list and len(self.queue) <512:
# #                     print(idx[i].item())
# #                     print(self.idx_list)
#                     self.idx_list.append(idx[i].item())                    
#                     self.queue.append(feat_k[i].clone()[None,:])
#                     print('LIST',len(self.idx_list))
#                     print('1',feat_k[i][None,:].size())
#                 elif idx[i] in self.idx_list:
#                     print('duplicate')
            if K >= 512:
#                 print('pop')
                self.queue.pop(0)
#                 self.idx_list.pop(0)

        return loss