import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer_vos import trainer_cellseg

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/mnt/parscratch/users/coq20tz/TransUNet/data/cell_arg', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='CellSeg', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='/mnt/parscratch/users/coq20tz/TransUNet/lists/cellseg', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=200, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=256, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--num_workers', type=int,
                    default=16, help='num of workers, default is 16')
parser.add_argument('--start_epoch', type=int,
                    default=100, help='default is 100')
parser.add_argument('--sample_number', type=int, default=100)
parser.add_argument('--select', type=int, default=1000)
parser.add_argument('--sample_from', type=int, default=10000)
parser.add_argument('--loss_type', type=str, default='orig', help='choose from pareto, norm or orig')
parser.add_argument('--use_vos', action="store_true")

args = parser.parse_args()


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
    dataset_name = args.dataset
    # dataset_config = {
    #     'Synapse': {
    #         'root_path': '../data/Synapse/train_npz',
    #         'list_dir': './lists/lists_Synapse',
    #         'num_classes': 9,
    #     },
    # }
    dataset_config = {
        'CellSeg': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
        },
    }
    # args.num_classes = dataset_config[dataset_name]['num_classes']
    # args.root_path = dataset_config[dataset_name]['root_path']
    # args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "/mnt/parscratch/users/coq20tz/TransUNet/model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    if args.use_vos:
        snapshot_path += '_St' + str(args.start_epoch)
        snapshot_path += '_SN' + str(args.sample_number)
        snapshot_path += '_SEL' + str(args.select)
        snapshot_path += '_SF' + str(args.sample_from)
        snapshot_path += '_LT' + args.loss_type
        snapshot_path += '_VOS' if args.use_vos else ""

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    net.load_from(weights=np.load(config_vit.pretrained_path))
    
    ###################### load model for visualize ################################
    # checkpoint_path = '/mnt/parscratch/users/coq20tz/TransUNet/model/TU_CellSeg224/TU_pretrain_R50-ViT-B_16_skip0_epo200_bs128_224_St150_SN100_SEL10000_SF100000_LTorig_VOS/epoch_199.pth'
    # checkpoint = torch.load(checkpoint_path)
    # net.load_state_dict(checkpoint)  # Adjust key if necessary
    # net = net.cuda()  # Move model to GPU
    ###################### load model for visualize ################################
    
    trainer = {'CellSeg': trainer_cellseg,}
    trainer[dataset_name](args, net, snapshot_path)