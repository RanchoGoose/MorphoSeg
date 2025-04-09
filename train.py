import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
# from trainer_vos import trainer_cellseg
from trainer import trainer_cellseg
from datasets.dataset_blood import BloodCell_dataset
from datasets.dataset_livecell import LiveCell_dataset  # Import the new dataset
from ScaleFormer.networks.ScaleFormer import ScaleFormer  # Import the ScaleFormer model
# Import Swin-UNet
from SwinUnet.networks.vision_transformer import SwinUnet
from SwinUnet.config import get_config as get_swin_config
import torch.nn as nn

# Import segmentation_models_pytorch for UNet++
import segmentation_models_pytorch as smp

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/mnt/parscratch/users/coq20tz/TransUNet/data/cell_arg', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='CellSeg', help='dataset to use: CellSeg, BloodCell, or LiveCell')
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
                    default=0, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model, ScaleFormer, SwinUnet, or UnetPlusPlus')
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

parser.add_argument('--use_topo', action="store_true", help='Enable topological consistency loss')
parser.add_argument('--lambda_topo', type=float, default=1.0, help='Weight for topological loss')
parser.add_argument('--topo_size', type=int, default=100, help='Patch size for topological loss computation')
parser.add_argument('--pd_threshold', type=float, default=0.7, help='Persistence threshold for topological loss computation')

# Swin-UNet specific parameters
parser.add_argument('--cfg', type=str, default='./SwinUnet/configs/swin_tiny_patch4_window7_224_lite.yaml', 
                    help='path to config file for SwinUNet')
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory (Swin-UNet)")
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')

# For UNet++ parameters
parser.add_argument('--encoder_name', type=str, default='resnet34', 
                    help='Encoder backbone for UNet++ (e.g., resnet34, resnet50, efficientnet-b0)')
parser.add_argument('--encoder_weights', type=str, default='imagenet',
                    help='Pretrained weights for encoder (imagenet or None)')

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
        'BloodCell': {
            'root_path': '/mnt/parscratch/users/coq20tz/TransUNet/data/BCCD',
            'list_dir': '/mnt/parscratch/users/coq20tz/TransUNet/lists/bloodcell',
            'num_classes': 2,
        },
        'LiveCell': {
            'root_path': '/mnt/parscratch/users/coq20tz/TransUNet/data/livecell',
            'list_dir': '/mnt/parscratch/users/coq20tz/TransUNet/lists/livecell',
            'num_classes': 2,
        },
    }
    # IMPORTANT - Use the dataset-specific configurations
    if dataset_name in dataset_config:
        args.root_path = dataset_config[dataset_name]['root_path']
        args.list_dir = dataset_config[dataset_name]['list_dir']
        args.num_classes = dataset_config[dataset_name]['num_classes']
        print(f"Using dataset config for {dataset_name}: {dataset_config[dataset_name]}")
    # args.num_classes = dataset_config[dataset_name]['num_classes']
    # args.root_path = dataset_config[dataset_name]['root_path']
    # args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "./model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    
    # Add vit_name to snapshot path (will either be a ViT model name or other model name)
    snapshot_path += '_' + args.vit_name
    
    # Only add skip and vitpatch info if using TransUNet (not other models)
    if args.vit_name not in ['ScaleFormer', 'SwinUnet']:
        snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
        snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    
    snapshot_path = snapshot_path + str(args.dataset)
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

    if args.use_topo:
        snapshot_path += '_TOPO'
        snapshot_path += '_LAMBDA' + str(args.lambda_topo)
        snapshot_path += '_TOPO_SIZE' + str(args.topo_size)
        snapshot_path += '_PD_THRESHOLD' + str(args.pd_threshold)
        
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    # Initialize model based on vit_name
    if args.vit_name == 'ScaleFormer':
        net = ScaleFormer(n_classes=args.num_classes).cuda()
        # Apply Kaiming initialization to convolutional layers
        for m in net.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # To prevent NaN issues, make sure all batch norms have momentum and eps
                m.momentum = 0.1
                m.eps = 1e-5
        print("Initialized ScaleFormer with Kaiming initialization")
    elif args.vit_name == 'SwinUnet':
        # Set up Swin-UNet with the existing config
        swin_args = argparse.Namespace()
        swin_args.cfg = args.cfg
        swin_args.opts = None
        swin_args.batch_size = args.batch_size
        swin_args.cache_mode = None
        swin_args.resume = None
        swin_args.accumulation_steps = None
        swin_args.use_checkpoint = args.use_checkpoint
        swin_args.amp_opt_level = None
        swin_args.tag = None
        swin_args.eval = None
        swin_args.throughput = None
        swin_args.n_class = args.num_classes
        # Add missing attributes that are checked in config.py
        swin_args.zip = args.zip if hasattr(args, 'zip') else False
        
        config = get_swin_config(swin_args)
        
        # Update config with correct number of classes
        config.defrost()  # Make config mutable before changing values
        config.MODEL.NUM_CLASSES = args.num_classes
        # Set the pre-trained model path directly in the config
        pretrained_path = '/mnt/parscratch/users/coq20tz/TransUNet/model/swin_tiny_patch4_window7_224.pth'
        config.MODEL.PRETRAIN_CKPT = pretrained_path
        config.freeze()  # Freeze config after changes
        
        # Create the SwinUnet model
        net = SwinUnet(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
        
        if os.path.exists(pretrained_path):
            # Use the model's built-in function to load weights
            print(f"Loading Swin-UNet pretrained weights from: {pretrained_path}")
            net.load_from(config)
        else:
            print(f"Warning: Pre-trained weights not found at {pretrained_path}")
    elif args.vit_name == 'UnetPlusPlus':
        # Create a UNet++ model from segmentation_models_pytorch
        net = smp.UnetPlusPlus(
            encoder_name=args.encoder_name,
            encoder_weights=args.encoder_weights,
            in_channels=1,  # Changed from 3 to 1 for grayscale images
            classes=args.num_classes,
            activation=None  # No activation, we'll use softmax in loss function
        ).cuda()
        
        print(f"Initialized UNet++ with {args.encoder_name} backbone, pretrained: {args.encoder_weights}, input channels: 1")
    else:
        # Use TransUNet with specified ViT backbone
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = args.n_skip
        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
        net.load_from(weights=np.load(config_vit.pretrained_path))
    
    trainer = {
        'CellSeg': trainer_cellseg,
        'BloodCell': trainer_cellseg,  # Using the same trainer for now, modify if needed
        'LiveCell': trainer_cellseg,   # Using the same trainer for LiveCell
    }
    
    if dataset_name not in trainer:
        raise ValueError(f"Dataset {dataset_name} is not supported. Choose from: {list(trainer.keys())}")
    
    trainer[dataset_name](args, net, snapshot_path)