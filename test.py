import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_cellseg import CellSeg_dataset
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from datasets.dataset_blood import BloodCell_dataset
from datasets.dataset_livecell import LiveCell_dataset
from ScaleFormer.networks.ScaleFormer import ScaleFormer  # Import the ScaleFormer model
# Import Swin-UNet
from SwinUnet.networks.vision_transformer import SwinUnet
from SwinUnet.config import get_config as get_swin_config
import datetime  # Add datetime for timestamped filenames
from torchvision import transforms
# Import segmentation_models_pytorch for UNet++
import segmentation_models_pytorch as smp

# Helper function to load pre-trained Swin-UNet weights
def load_swin_weights(model, pretrained_path):
    if not os.path.exists(pretrained_path):
        print(f"Warning: Pre-trained weights not found at {pretrained_path}")
        return False
    
    try:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        # Extract the model weights if they're nested under a key like 'model' or 'state_dict'
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Filter out keys that don't match the model 
        # (ignoring those that start with module. which happens with DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k[7:] if k.startswith('module.') else k
            if k in model.state_dict():
                new_state_dict[k] = v
        
        # Handle missing or extra keys
        missing_keys = [k for k in model.state_dict().keys() if k not in new_state_dict]
        unexpected_keys = [k for k in new_state_dict.keys() if k not in model.state_dict()]
        if missing_keys:
            print(f"Missing keys in pretrained weights: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys in pretrained weights: {unexpected_keys}")
            
        # Load the filtered state dict
        model.load_state_dict(new_state_dict, strict=False)
        print(f"Successfully loaded pre-trained weights from {pretrained_path}")
        return True
    except Exception as e:
        print(f"Error loading pre-trained weights: {e}")
        return False

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='/mnt/parscratch/users/coq20tz/TransUNet/data/cell_arg', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='CellSeg', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='/mnt/parscratch/users/coq20tz/TransUNet/lists/cellseg', help='list dir')

parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=30, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=0, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='ViT-B_16', help='select one vit model, ScaleFormer, SwinUnet, or UnetPlusPlus')

parser.add_argument('--test_save_dir', type=str, default='./predictions', help='saving prediction as png')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
parser.add_argument('--data_split', type=str, default="test", help='choose between train test and eval')
parser.add_argument('--start_epoch', type=int,
                    default=150, help='default is 100')
parser.add_argument('--sample_number', type=int, default=100)
parser.add_argument('--select', type=int, default=1000)
parser.add_argument('--sample_from', type=int, default=10000)
parser.add_argument('--loss_type', type=str, default='orig')
parser.add_argument('--use_vos', action="store_true")
parser.add_argument('--use_topo', action="store_true", help='Enable topological consistency loss')
parser.add_argument('--lambda_topo', type=float, default=1.0, help='Weight for topological loss')
parser.add_argument('--topo_size', type=int, default=100, help='Patch size for topological loss computation')
parser.add_argument('--pd_threshold', type=float, default=0.7, help='Persistence threshold for topological loss computation')

# Swin-UNet specific parameters
parser.add_argument('--cfg', type=str, default='./SwinUnet/configs/swin_tiny_patch4_window7_224_lite.yaml', 
                    help='path to config file for Swin-UNet')
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory (Swin-UNet)")

# For UNet++ parameters
parser.add_argument('--encoder_name', type=str, default='resnet34', 
                    help='Encoder backbone for UNet++ (e.g., resnet34, resnet50, efficientnet-b0)')
parser.add_argument('--encoder_weights', type=str, default='imagenet',
                    help='Pretrained weights for encoder (imagenet or None)')

args = parser.parse_args()

def inference(args, model, test_save_path=None):
    if args.is_savenii:
        db_test = args.Dataset(base_dir=args.volume_path, split=args.data_split, list_dir=args.list_dir)
    else:
        db_test = args.Dataset(base_dir=args.volume_path, list_dir=args.list_dir, split=args.data_split,
                               transform=args.test_transform)

    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    logging.info(f"Test dataset has {len(db_test)} samples")

    model.eval()
    iou_thresholds = [0.5, 0.75, 0.9]
    metrics_agg = {
        "dice": [], 
        "hd95": [], 
        "iou_scores": [], 
        "ap_scores": {thr: [] for thr in iou_thresholds}
    }
    
    # Track success and error counts
    successful_cases = 0
    error_cases = 0

    for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader)):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
         
        # try:
        # Handle UNet++ output specifically - it might return a list in some configurations
        if not args.is_savenii and args.vit_name == "UnetPlusPlus":
            with torch.no_grad():
                outputs = model(image.cuda())
                
                # Check if model output is a list (can happen with UNet++)
                if isinstance(outputs, list):
                    outputs = outputs[-1]  # Use the final output in the list
        
        metrics, iou_scores, aps = test_single_volume(
            image, label, model, classes=args.num_classes, 
            patch_size=[args.img_size, args.img_size],
            test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing,
            iou_thresholds=iou_thresholds
        )
        
        # Aggregate Dice and HD95
        metrics_agg["dice"].append(metrics[0])
        metrics_agg["hd95"].append(metrics[1])
        
        # Collect raw IoU value for each case
        # Add the IoU score (same for all thresholds, since it's the raw IoU value)
        if iou_scores[iou_thresholds[0]]:  # Use first threshold as they all have same IoU value
            metrics_agg["iou_scores"].append(iou_scores[iou_thresholds[0]][0])
        
        # Collect AP scores (0 or 1) for each threshold
        for thr in iou_thresholds:
            if aps[thr]:
                metrics_agg["ap_scores"][thr].extend(aps[thr])
                
        successful_cases += 1
                
        # except Exception as e:
        #     error_cases += 1
        #     logging.error(f'Error processing case {case_name}: {str(e)}')
        #     import traceback
        #     logging.error(traceback.format_exc())
        #     continue

    # Log test summary statistics
    logging.info(f"\nTest Summary: {successful_cases} successful cases, {error_cases} error cases")

    # Calculate mean IoU (raw score)
    if metrics_agg["iou_scores"]:
        mean_iou = np.mean(metrics_agg["iou_scores"])
        logging.info(f"Mean IoU (raw): {mean_iou:.4f}")
    
    # Calculate mean AP for each threshold (percentage of cases where IoU > threshold)
    for thr in iou_thresholds:
        if metrics_agg["ap_scores"][thr]:
            ap_mean = np.mean(metrics_agg["ap_scores"][thr])
            logging.info(f"AP@{thr}: {ap_mean:.4f} ({int(ap_mean * 100)}% of cases have IoU > {thr})")
    
    # Calculate success rate at each threshold
    for thr in iou_thresholds:
        if metrics_agg["iou_scores"]:
            # Count cases where IoU > threshold
            above_threshold = sum(1 for iou in metrics_agg["iou_scores"] if iou > thr)
            pct_above = (above_threshold / len(metrics_agg["iou_scores"])) * 100
            logging.info(f"IoU > {thr}: {above_threshold}/{len(metrics_agg['iou_scores'])} cases ({pct_above:.1f}%)")

    # Print mean Dice and HD95 for each class
    for i in range(1, args.num_classes):
        class_dices = [metrics_agg["dice"][j] for j in range(len(metrics_agg["dice"])) if (j % (args.num_classes - 1)) == (i - 1)]
        class_hd95s = [metrics_agg["hd95"][j] for j in range(len(metrics_agg["hd95"])) if (j % (args.num_classes - 1)) == (i - 1)]
        
        if class_dices:  # Check if not empty
            mean_dice = np.mean(class_dices)
            mean_hd95 = np.mean(class_hd95s)
            logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, mean_dice, mean_hd95))

    if metrics_agg["dice"]:  # Check if not empty
        performance = np.mean(metrics_agg["dice"])
        mean_hd95 = np.mean(metrics_agg["hd95"])
        logging.info('Testing performance: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    
        # Calculate mean AP across all thresholds
        ap_means = []
        for thr in iou_thresholds:
            if metrics_agg["ap_scores"][thr]:
                ap_means.append(np.mean(metrics_agg["ap_scores"][thr]))
        
        if ap_means:
            mean_ap = np.mean(ap_means)
            logging.info('Mean AP across all thresholds: %f' % mean_ap)
        
        # Report IoU and AP for standardized format
        mean_iou = np.mean(metrics_agg["iou_scores"]) if metrics_agg["iou_scores"] else 0
        ap_values = [np.mean(metrics_agg["ap_scores"][thr]) if metrics_agg["ap_scores"][thr] else 0 for thr in iou_thresholds]
        
        logging.info('Raw IoU score: %.4f' % mean_iou)
        logging.info('AP scores for thresholds 0.5, 0.75, 0.9: %.4f, %.4f, %.4f' % 
                    (ap_values[0], ap_values[1], ap_values[2]))

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

    # Dataset configuration
    dataset_config = {
        'CellSeg': {
            'Dataset': CellSeg_dataset,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
            'z_spacing': 1,
        },
        'BloodCell': {
            'Dataset': BloodCell_dataset,
            'volume_path': '/mnt/parscratch/users/coq20tz/TransUNet/data/BCCD',
            'list_dir': '/mnt/parscratch/users/coq20tz/TransUNet/lists/bloodcell',
            'num_classes': 2,  # Adjust based on your classes
            'z_spacing': 1,
        },
        'LiveCell': {
            'Dataset': LiveCell_dataset,
            'volume_path': '/mnt/parscratch/users/coq20tz/TransUNet/data/livecell',
            'list_dir': '/mnt/parscratch/users/coq20tz/TransUNet/lists/livecell',
            'num_classes': 2,
            'z_spacing': 1,
        },
    }
     
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True
    
    # Add proper transform to ensure test images are resized to the correct input size
    # Create a custom transform that will properly resize images to 224x224
    from datasets.dataset_cellseg import RandomGenerator
    args.test_transform = RandomGenerator(output_size=[args.img_size, args.img_size])

    # name the same snapshot defined in train script!
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "./model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    
    # Only add skip and vitpatch info if using TransUNet (not other models)
    if args.vit_name not in ['ScaleFormer', 'SwinUnet']:
        snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
        snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    
    snapshot_path = snapshot_path + str(args.dataset)
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
        snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
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

    # Get the snapshot name for use in log filename
    snapshot_name = os.path.basename(snapshot_path)
    
    # Setup logging using specific folder and filename format
    log_folder = f'/mnt/parscratch/users/coq20tz/TransUNet/test_log/test_log_TU_{dataset_name}{args.img_size}'
    os.makedirs(log_folder, exist_ok=True)
    # Use snapshot name directly as the log filename as specified
    log_filename = os.path.join(log_folder, f'{snapshot_name}.txt')
    
    # Setup file logging without console output
    logging.basicConfig(
        level=logging.INFO, 
        format='[%(asctime)s.%(msecs)03d] %(message)s', 
        datefmt='%H:%M:%S',
        handlers=[logging.FileHandler(log_filename)]
    )
    
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
        print("Initialized ScaleFormer with Kaiming initialization for testing")
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
        swin_args.zip = args.zip if hasattr(args, 'zip') else False
        config = get_swin_config(swin_args)
        
        # Make config mutable before updating
        config.defrost()
        config.MODEL.NUM_CLASSES = args.num_classes
        config.freeze()
        
        # Create the SwinUnet model
        net = SwinUnet(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
        logging.info(f"Created SwinUnet with img_size={args.img_size}, num_classes={args.num_classes}")
    elif args.vit_name == 'UnetPlusPlus':
        # Create a UNet++ model from segmentation_models_pytorch
        net = smp.UnetPlusPlus(
            encoder_name=args.encoder_name,
            encoder_weights=args.encoder_weights,
            in_channels=1,  # Using 1 for grayscale medical images
            classes=args.num_classes,
            activation=None  # No activation, we'll use softmax in loss function
        ).cuda()
        
        logging.info(f"Initialized UNet++ with {args.encoder_name} backbone, pretrained: {args.encoder_weights}, input channels: 1")
    else:
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = args.n_skip
        config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
        if args.vit_name.find('R50') !=-1:
            config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): 
        snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
        
    # Load trained model
    if os.path.exists(snapshot):
        print(f"Loading trained model from {snapshot}")
        checkpoint = torch.load(snapshot)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            net.load_state_dict(checkpoint['state_dict'])
        else:
            net.load_state_dict(checkpoint)
        print(f"Successfully loaded trained model")
    else:
        print(f"ERROR: Could not find trained model at {snapshot}")
        sys.exit(1)  # Exit if model not found

    logging.info(f"Testing model: {args.vit_name}")
    logging.info(f"Snapshot path: {snapshot}")
    logging.info(f"Snapshot name: {snapshot_name}")

    # Add model summary info
    logging.info(f"Model parameters: {sum(p.numel() for p in net.parameters())}")
    logging.info(f"Model trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}")

    print("Starting inference on {} dataset...".format(dataset_name))

    if args.is_savenii:
        args.test_save_dir = './predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name + '_debug_vis')
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    # Add device info
    logging.info(f"Testing on device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # Modified inference call to pass the visualization debug path
    result = inference(args, net, test_save_path)
    
        
    # Only print the finished message
    print("Testing Finished!")


