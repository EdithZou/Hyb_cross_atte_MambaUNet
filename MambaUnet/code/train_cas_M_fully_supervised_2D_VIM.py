import argparse
import logging
import os
import random
import shutil
import sys
import time
import pdb
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
#from networks.vision_transformer import SwinUnet as ViT_seg

# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from networks.vision_mamba import MambaUnet as VIM_seg

from config import get_config

#from dataloaders import utils
from dataloaders.dataset_select import BaseDataSets, RandomGenerator
from dataloaders.balanced_dataloader import BalancedBatchSampler
from utils import losses
from utils.bilateralfilter.DenseCRFLoss import DenseCRFLoss
from val_2D_cas import test_single_volume


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/MYOSAIQ_dataset/train_resampled', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='MyoSAIQ/Cascade_Fully_Supervised_ViM', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='mambaunet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=3,
                    help='output channel of network')
parser.add_argument('--csv_file', type=str, dest='csv_file',
                    default='/home/tzou/MyoSAIQ/nnUnet/MYOSAIQ_data.csv', help='List of Data')
parser.add_argument(
    '--cfg', type=str, default="/home/tzou/MyoSAIQ/Mamba_UNet/code/configs/vmamba_tiny.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')


parser.add_argument('--max_iterations', type=int,
                    default=10000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[96, 96],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=666, help='random seed')
parser.add_argument('--labeled_num', type=int, default=716,
                    help='labeled data')
parser.add_argument("--alpha", type=float, help="weights parameter", dest="alpha", default=1.0)
parser.add_argument("--beta", type=float, help="weights parameter", dest="beta", default=1.0)
parser.add_argument("--gama", type=float, help="weights parameter", dest="gama", default=1.0)

args = parser.parse_args()


config = get_config(args)

def encode_labels(label, label_map = {0: 0, 1: 0, 2: 2, 3: 3, 4:4 }): # 只关注心肌
    encoded_label = torch.zeros_like(label)
    a = torch.max(label)
    for old_label, new_label in label_map.items():
        encoded_label[label == old_label] = new_label
    value_encode_label = torch.max(encoded_label)
    return encoded_label


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    # model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    # 解剖模型
    model_s1 = VIM_seg(config, img_size=args.patch_size,
                     num_classes=args.num_classes).cuda()
    model_s1.load_from(config)
    
    # 细病理模型
    model_s2 = VIM_seg(config, img_size=args.patch_size,
                     num_classes=args.num_classes+1, in_cha=2).cuda()
    model_s2.load_from(config)
    
    # 载入数据集
    db_train = BaseDataSets(base_dir=args.root_path, csv_file=args.csv_file, split="train", 
                            num=358, transform=transforms.Compose([
                            RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, csv_file=args.csv_file, split="valid")
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    #train_sampler = BalancedBatchSampler(db_train, batch_size=batch_size)
    #trainloader = DataLoader(db_train, batch_size=batch_size, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn, sampler=train_sampler)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    model_s1.train()
    model_s2.train()

    # 优化器设置
    optimizer_s1 = optim.SGD(model_s1.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    optimizer = optim.SGD(list(model_s1.parameters())+list(model_s2.parameters()), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)    
    # 损失函数
    ce_loss = CrossEntropyLoss()
    dice_loss_s1 = losses.DiceLoss(num_classes)
    dice_loss_s2= losses.DiceLoss(num_classes+1)
    #boundary_loss = losses.SurfaceLoss(idc=num_classes, num_classes=num_classes+2)
    
    # 训练日志
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            # stage 1
            outputs_s1 = model_s1(volume_batch)
            outputs_soft_s1 = torch.softmax(outputs_s1, dim=1)
            label_batch_s1 = encode_labels(label_batch, label_map = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2})
            
            loss_ce_s1 = ce_loss(outputs_s1, label_batch_s1[:].long())
            loss_dice_s1 = dice_loss_s1(outputs_soft_s1, label_batch_s1.unsqueeze(1))
            loss_s1 = loss_dice_s1 + loss_ce_s1
            myo_mask_s1 = torch.argmax(outputs_soft_s1, dim=1) == 2
            
            # stage 2
            outputs_s2 = model_s2(torch.concat((volume_batch, outputs_soft_s1[:,-1,:,:].unsqueeze(1)), dim=1))
            outputs_soft_s2 = torch.softmax(outputs_s2, dim=1)
            myo_mask_s2 = torch.argmax(outputs_soft_s2, dim=1) >= 2
            label_batch_s2 = encode_labels(label_batch, label_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3})
            
            # 对不同stage的训练进行分段
            if iter_num >= 700:
                args.beta = 1.5
                
            args.beta = min(args.beta, 1.5)
            
            loss_ce_s2 = ce_loss(outputs_s2, label_batch_s2[:].long())
            loss_dice_s2 = dice_loss_s2(outputs_soft_s2, label_batch_s2.unsqueeze(1))
            loss_l1 = F.l1_loss(myo_mask_s1.to(torch.float32), myo_mask_s2.to(torch.float32))
            loss_s2 = loss_ce_s2 + loss_l1 + loss_dice_s2
            
            loss_dice = (loss_dice_s1 + loss_dice_s2)/2
            loss = args.alpha * loss_s1 + args.beta * loss_s2
            
            if args.beta == 0.0:
                optimizer_model = optimizer_s1
            else:
                optimizer_model = optimizer
            
            optimizer_model.zero_grad()
            loss.backward()
            optimizer_model.step()
            
            if iter_num > 2000:
                lr_ = base_lr * (1.0 - (iter_num / 100) / max_iterations) ** 0.9
            else:
                lr_ = base_lr
            for param_group in optimizer_model.param_groups:
                param_group['lr'] = lr_         
            iter_num = iter_num + 1
            
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            #writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', 1-loss_dice, iter_num)
            logging.info(
                'iteration %d : loss : %f, loss_dice: %f, loss_dice_s1: %f, loss_dice_s2: %f' %
                (iter_num, loss.item(), 1-loss_dice.item(), 1-loss_dice_s1.item(), 1-loss_dice_s2.item()))

            # 每两百次迭代后，对模型进行检验
            if iter_num > 0 and iter_num % 200 == 0:
                model_s1.eval()
                model_s2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"], model_s1, model_s2, classes=num_classes+1, patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                
                for class_i in range(num_classes+1-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)
                
                performance = np.mean(metric_list, axis=0)[0]              
                mean_hd95 = np.mean(metric_list, axis=0)[1]
                
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)
                
                if performance > best_performance:
                    best_performance = performance
                    save_mode_path_s1 = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}_s1.pth'.format(
                                                      iter_num, round(performance, 4)))
                    save_mode_path_s2 = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}_s2.pth'.format(
                                                      iter_num, round(performance, 4)))
                    
                    save_best_s1 = os.path.join(snapshot_path,
                                             '{}_best_model_s1.pth'.format(args.model))
                    save_best_s2 = os.path.join(snapshot_path,
                                             '{}_best_model_s2.pth'.format(args.model))
                   
                    torch.save(model_s1.state_dict(), save_mode_path_s1)
                    torch.save(model_s1.state_dict(), save_best_s1)
                    torch.save(model_s2.state_dict(), save_mode_path_s2)
                    torch.save(model_s2.state_dict(), save_best_s2)

                                        

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model_s1.train()
                model_s2.train()
                #model_s3.train()

            '''
            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
            
            '''

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    print('Training Finished!')
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "/home/tzou/Mamba_model/{}_{}_M1_12/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    #if os.path.exists(snapshot_path + '/code'):
    #    shutil.rmtree(snapshot_path + '/code')
    #shutil.copytree('.', snapshot_path + '/code',
    #                shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
