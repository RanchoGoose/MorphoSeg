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
parser.add_argument('--vit_name', type=str, default='ViT-B_16', help='select one vit model')

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
args = parser.parse_args()

# def inference(args, model, test_save_path=None):   
#     db_test = args.Dataset(base_dir=args.volume_path, split=args.data_split, list_dir=args.list_dir)
#     testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
#     logging.info("{} test iterations per epoch".format(len(testloader)))
#     model.eval()
#     metric_list = 0.0   
#     for i_batch, sampled_batch in tqdm(enumerate(testloader)):
#         # h, w = sampled_batch["image"].size()[2:]
#         c, h, w = sampled_batch["image"].size()
#         image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
#         metric_i= test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
#                                       test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
#         metric_list += np.array(metric_i)
#         logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
#     metric_list = metric_list / len(db_test)
    
#     for i in range(1, args.num_classes):
#         logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
#     performance = np.mean(metric_list, axis=0)[0]
#     mean_hd95 = np.mean(metric_list, axis=0)[1]
#     logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
#     return "Testing Finished!"

def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split=args.data_split, list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))

    model.eval()
    iou_thresholds = [0.5, 0.75, 0.9]
    metrics_agg = {
        "dice": [], 
        "hd95": [], 
        "iou_scores": {thr: [] for thr in iou_thresholds}, 
        "ap": {thr: [] for thr in iou_thresholds}
    }

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metrics, iou_scores, aps = test_single_volume(
            image, label, model, classes=args.num_classes, 
            patch_size=[args.img_size, args.img_size],
            test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing,
            iou_thresholds=iou_thresholds
        )
        
        # Aggregate Dice and HD95
        for metric in metrics:
            metrics_agg["dice"].append(metric[0])
            metrics_agg["hd95"].append(metric[1])
        
        # Aggregate IoU scores and AP for each threshold
        for thr in iou_thresholds:
            metrics_agg["iou_scores"][thr].extend(iou_scores[thr])
            metrics_agg["ap"][thr].extend(aps[thr])

    # Calculate mean IoU scores and AP for each threshold
    for thr in iou_thresholds:
        metrics_agg["iou_scores"][thr] = np.mean(metrics_agg["iou_scores"][thr])
        metrics_agg["ap"][thr] = np.mean(metrics_agg["ap"][thr])

    # Print mean Dice and HD95 for each class
    for i in range(1, args.num_classes):
        class_dices = [metrics_agg["dice"][j] for j in range(len(metrics_agg["dice"])) if (j % (args.num_classes - 1)) == (i - 1)]
        class_hd95s = [metrics_agg["hd95"][j] for j in range(len(metrics_agg["hd95"])) if (j % (args.num_classes - 1)) == (i - 1)]
        mean_dice = np.mean(class_dices)
        mean_hd95 = np.mean(class_hd95s)
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, mean_dice, mean_hd95))

    performance = np.mean(metrics_agg["dice"])
    mean_hd95 = np.mean(metrics_agg["hd95"])
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    
    mean_ap = np.mean([metrics_agg["ap"][thr] for thr in iou_thresholds])
    # Log the IoU scores for the specified thresholds and mean AP
    logging.info('IoU scores for thresholds 0.5, 0.75, 0.9: {}, {}, {}'.format(
        metrics_agg["iou_scores"][0.5], metrics_agg["iou_scores"][0.75], metrics_agg["iou_scores"][0.9]
    ))
    logging.info('Mean AP: %f' % mean_ap)

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

    dataset_config = {
        'CellSeg': {
            'Dataset': CellSeg_dataset,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
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

    # name the same snapshot defined in train script!
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "./model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
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
    
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # print the logging info
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = './predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)


