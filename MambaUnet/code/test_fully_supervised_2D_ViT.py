import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from config import get_config


import nibabel as nib
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom

from dataloaders import utils
from dataloaders.dataset import BaseDataSets, RandomGenerator, BaseDataSets_Synapse
#from networks.net_factory import net_factory
from networks.vision_transformer import SwinUnet as ViT_seg
# from networks.vision_mamba import MambaUnet as VIM
from utils import losses, metrics, ramps
from val_2D import test_single_volume, test_single_volume_ds

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/tzou/MYOSAIQ_dataset/train_resampled_big', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='MyoSAIQ/Fully_Supervised_ViT', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='trans_unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=3,
                    help='output channel of network')
parser.add_argument('--csv_file', type=str, dest='csv_file',
                    default='/home/tzou/MyoSAIQ/nnUnet/MYOSAIQ_data.csv', help='List of Data')

parser.add_argument('--max_iterations', type=int,
                    default=10000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=5e-3,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[96, 96],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument(
    '--cfg', type=str, default="/home/tzou/MyoSAIQ/Mamba_UNet/code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
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

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=358,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()
config = get_config(args)



def test(args, snapshot_path):
    num_classes = args.num_classes
    batch_size = args.batch_size

    model = ViT_seg(config, img_size=args.patch_size, num_classes=args.num_classes).cuda()
    model.load_state_dict(torch.load(os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))))
    model.eval()

    db_test = BaseDataSets(base_dir=args.root_path, csv_file=args.csv_file, split="test")

    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    metric_list = 0.0
    for i_batch, sampled_batch in enumerate(testloader):
        metric_i = test_single_volume(sampled_batch["image"], model, classes=num_classes)
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(db_test)

    for class_i in range(num_classes-1):
        logging.info('val_{}_dice: {}'.format(class_i+1, metric_list[class_i, 0]))
        logging.info('val_{}_hd95: {}'.format(class_i+1, metric_list[class_i, 1]))

    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('val_mean_dice: {}'.format(performance))
    logging.info('val_mean_hd95: {}'.format(mean_hd95))

    return "Testing Finished!"

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

    snapshot_path = "../model/{}_{}_labeled/{}".format(args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log_test.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    test(args, snapshot_path)

