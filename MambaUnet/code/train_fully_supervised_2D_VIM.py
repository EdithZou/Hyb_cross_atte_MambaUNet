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
#from networks.vision_mamba import MambaUnet as VIM_seg
from networks.vision_mamba import CrossAttentionMambaUnet as hyb_VIM_att


from config import get_config

#from dataloaders import utils
from dataloaders.dataset_select import BaseDataSets, RandomGenerator
from dataloaders.balanced_dataloader import BalancedBatchSampler
from utils import losses
from utils.bilateralfilter.NCloss import NCLoss
from val_2D import test_single_volume


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/MYOSAIQ_dataset/train_resampled', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='MyoSAIQ/Fully_Supervised_ViM', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='mambaunet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=5,
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
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[96, 96],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--labeled_num', type=int, default=716,
                    help='labeled data')
args = parser.parse_args()


config = get_config(args)

def patients_to_slices(dataset):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "MYOSAIQ" in dataset:
        ref_dict = {"D8_postMI": 98, "M1_postMI": 155, "M12_postMI": 105}
    else:
        ValueError("Dataset not recognized.")
    return ref_dict


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    # model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)

    model = hyb_VIM_att(config, img_size=args.patch_size,
                     num_classes=args.num_classes).cuda()


    db_train = BaseDataSets(base_dir=args.root_path, csv_file=args.csv_file, split="train", 
                            num=358, transform=transforms.Compose([
                            RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, csv_file=args.csv_file, split="valid")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    #trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
    #                         num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    train_sampler = BalancedBatchSampler(db_train, batch_size=batch_size)
    trainloader = DataLoader(db_train, batch_size=batch_size, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn, sampler=train_sampler)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    model.train()
    # 优化器设置
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    
    # 损失函数
    #ce_loss = CrossEntropyLoss(ignore_index=4)
    weight = torch.tensor([1.0,1.0,2.0,3.0,4.0]).cuda()
    ce_loss = CrossEntropyLoss(weight=weight)
    #weight = torch.tensor([1.0,1.0,2.0,3.0,4.0]).cuda()
    #ce_loss_s3 = CrossEntropyLoss(weight=weight)
    dice_loss = losses.DiceLoss(num_classes)
    #dice_loss_s2= losses.DiceLoss(num_classes+1)
    #dice_loss_s3= losses.DiceLoss(num_classes+2)
    nc_loss = NCLoss(weight=batch_size, sigma_rgb=50, sigma_xy=100, scale_factor=0.6)
    
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

            outputs, outputs_dice = model(volume_batch)
            outputs_dice = torch.softmax(outputs_dice, dim=1)
            outputs_soft = torch.softmax(outputs, dim=1)
            loss_dice_img = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            loss_img = 0.2 * ce_loss(outputs, label_batch[:].long()) + 0.8 * loss_dice_img
            loss_dice_attn = dice_loss(outputs_dice,label_batch.unsqueeze(1))
            loss_lbl = 0.2 * ce_loss(outputs_dice, label_batch[:].long()) + 0.8 * loss_dice_attn
            loss_nc = nc_loss(volume_batch.repeat(1,3,1,1), outputs_dice, label_batch)
            #loss_boundary = boundary_loss(outputs_soft, label_batch.unsqueeze(1))
            loss = loss_img +  loss_lbl + 1 * loss_nc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_dice_img', 1-loss_dice_attn, iter_num)
            writer.add_scalar('info/loss_dice_lbl', 1-loss_dice_img, iter_num)
            writer.add_scalar('info/loss_nc', loss_nc, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_dice_img: %f, loss_dice_lbl: %f, loss_nc: %f' %
                (iter_num, loss.item(), 1-loss_dice_attn.item(), 1-loss_dice_img.item(), loss_nc.item()))
            '''
            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                # 计算预测概率最大的类别对应的输出结果
                labs = label_batch[1, ...].unsqueeze(0) * 60
                writer.add_image('train/GroundTruth', labs, iter_num)
                outputs = torch.argmax(outputs_soft, dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1,...] * 60, iter_num) 
            '''
            # 每两百次迭代后，对模型进行检验
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes, patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
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
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                logging.info(
                    'valid iteration %d : LV_dice: %f, myo_dice: %f, mi_dice: %f, mvo_dice: %f' %
                    (iter_num / 200, metric_list[0, 0], metric_list[1, 0], metric_list[2, 0], metric_list[3, 0]))    
                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

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

    snapshot_path = "/home/tzou/Mamba_model/{}_{}_labeled/{}".format(
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
