import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, convert_to_uint8
from torchvision import transforms
from skimage.transform import warp, AffineTransform
import SimpleITK as sitk
# from datasets.dataset_vos import generate_synthetic_outliers

def trainer_cellseg(args, model, snapshot_path):
    from datasets.dataset_cellseg import CellSeg_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = CellSeg_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    # def worker_init_fn(worker_id):
    #     random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    epsilon = 1e-8
    
    # # Read the training filenames from train.txt
    # with open('/mnt/parscratch/users/coq20tz/TransUNet/lists/cellseg/train_full.txt', 'r') as file:
    #     full_filenames = file.read().splitlines()
    
    # data_dict = {class_index: torch.tensor([]).cuda() for class_index in range(num_classes)}
    for epoch_num in iterator:
        ood_samples = None  # Initialize outside the inner loop
        # pareto_loss_handler = None
        # if args.loss_type == 'pareto':
        #     pareto_loss_handler = ParetoMultiTaskLoss(device='cuda')
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            # for i, feature_tensor in enumerate(outputs):
            #     print(f"Shape of x {i}: {feature_tensor.shape}")
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss_ce_vos = None
            loss_dice_vos = None
            loss_outputs = [loss_ce, loss_dice] 

            if epoch_num >= args.start_epoch and args.use_vos:               
                syn_outputs = generate_synthetic_segmentation_maps(outputs, args)                                                                 
                loss_ce_vos = ce_loss(syn_outputs, label_batch[:].long())
                loss_dice_vos = dice_loss(syn_outputs, label_batch, softmax=True)     
                loss_outputs.extend([loss_ce_vos, loss_dice_vos])
                
                # Loop through all case names in the batch
                # for idx, current_filename in enumerate(sampled_batch['case_name']):
                #     # if current_filename in full_filenames and (epoch_num + 10) % 10 == 0:
                #     # print(current_filename)
                #     # Compute the prediction map (e.g., argmax over the output channels)
                #     syn_map = torch.softmax(syn_outputs[idx], dim=0).argmax(dim=0).squeeze().cpu().numpy()
                #     prediction_map = torch.softmax(outputs[idx], dim=0).argmax(dim=0).squeeze().cpu().numpy()
                    # save_training_sample(image_batch[idx], label_batch[idx], syn_map, prediction_map, i_batch, epoch_num, current_filename, save_dir='/mnt/parscratch/users/coq20tz/TransUNet/figs')
                    
            if args.loss_type == 'norm':
                loss = sum(loss / (loss.detach() + epsilon) for loss in loss_outputs)
            elif args.loss_type == 'pareto':
                pareto_loss_handler = ParetoMultiTaskLoss(device='cuda')
                loss = pareto_loss_handler.compute_loss(torch.stack(loss_outputs))
            elif args.loss_type == 'orig':
                weights = [0.5, 0.5] if loss_ce_vos is None else [0.3, 0.3, 0.2, 0.2]
                loss = sum(w * l for w, l in zip(weights, loss_outputs))
            optimizer.zero_grad()           
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/loss_ce_vos', loss_ce_vos if loss_ce_vos is not None else 0, iter_num)
            writer.add_scalar('info/loss_dice_vos', loss_dice_vos if loss_dice_vos is not None else 0, iter_num)

            # logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, loss_ce_vos: %f, loss_dice_vos: %f', % (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), loss_ce_vos.item(), loss_dice_vos.item()))
            
            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, loss_ce_vos: %f, loss_dice_vos: %f', iter_num, loss.item(), loss_ce.item(), loss_dice.item(), loss_ce_vos.item() if loss_ce_vos is not None else 0, loss_dice_vos.item() if loss_dice_vos is not None else 0)

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"

def is_positive_definite(matrix):
    try:
        torch.linalg.cholesky(matrix)
        return True
    except RuntimeError:
        return False
    
def generate_synthetic_segmentation_maps(outputs, args):
    B, C, H, W = outputs.shape
    device = outputs.device
    # print(B, C, H, W)

    # Assuming class_means and covariance have been calculated as described previously
    class_means = torch.zeros(C, device=device)
    flattened_outputs = outputs.permute(1, 0, 2, 3).reshape(C, -1)
    for c in range(C):
        class_means[c] = flattened_outputs[c].mean()
    centered_outputs = flattened_outputs - class_means[:, None]
    cov = (centered_outputs @ centered_outputs.T) / (flattened_outputs.shape[1] - 1)
    cov += 0.01 * torch.eye(C, device=device)

    if not is_positive_definite(cov):
        raise ValueError("Covariance matrix is not positive definite.")

    # Initialize the Multivariate Normal distribution
    mvn = torch.distributions.MultivariateNormal(class_means, covariance_matrix=cov)
    # Sample from the distribution
    samples = mvn.rsample(sample_shape=(args.sample_from,))
    # Calculate log probabilities for the sampled logits to identify low-density areas
    log_probs = mvn.log_prob(samples)
    # Select indices of samples in low-density areas
    _, low_density_indices = torch.topk(-log_probs, k=args.select, largest=True)
    # Prepare the tensor for selected synthetic logits based on low-density samples
    selected_synthetic_logits = samples[low_density_indices]
    # Initialize a tensor to hold updated logits, starting with original outputs
    updated_logits = outputs.reshape(B * H * W, C).clone()  # Flatten logits for easier indexing
    # Since low_density_indices are selected from a flat array of samples,
    # randomly choose positions in the logits tensor to replace with synthetic samples.
    # Here, args.select determines how many replacements you make, ensuring they match the low-density count.
    indices_to_replace = torch.randperm(B * H * W)[:args.select].to(device)
    updated_logits[indices_to_replace] = selected_synthetic_logits

    # Reshape the updated logits back to match the original outputs' shape
    updated_logits = updated_logits.reshape(B, C, H, W)

    return updated_logits


class ParetoMultiTaskLoss:
    def __init__(self, weights=None, device='cpu'):
        """
        Initialize the multi-task loss handler with Pareto optimization.
        :param weights: Initial weights for each loss, if None, initialize with equal weights.
        :param device: Device to perform computations on.
        """
        self.weights = weights  # Initialize weights, or they will be set when compute_loss is called
        self.device = device

    def update_weights(self, losses):
        """
        Update weights inversely proportional to the losses to implement a form of Pareto optimization.
        """
        with torch.no_grad():
            normalized_losses = losses / losses.sum()
            self.weights = 1.0 / (normalized_losses + 1e-6)
            self.weights /= self.weights.sum()

    def compute_loss(self, losses):
        """
        Compute the combined loss using the current weights and update weights based on current losses.
        """
        if self.weights is None or len(self.weights) != len(losses):
            # Initialize or update weights if they are not set or if their length is not correct
            self.weights = torch.ones(len(losses), dtype=torch.float32, device=self.device)
        
        # Now `losses` and `self.weights` should have the same dimension
        combined_loss = torch.sum(losses * self.weights)
        self.update_weights(losses)
        return combined_loss
    
def save_training_sample(image, original_mask, synthetic_map, prediction_map, batch_idx, epoch_num, filename, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    # image = image.cpu().numpy().transpose(1, 2, 0)
    image = image.cpu().numpy()
    original_mask = original_mask.cpu().numpy()
    
    base_filename = os.path.basename(filename).replace('.png', '') 
    
    # Convert images to uint8
    img_uint8 = convert_to_uint8(image)
    prd_uint8 = convert_to_uint8(prediction_map)
    syn_uint8 = convert_to_uint8(synthetic_map)
    lab_uint8 = convert_to_uint8(original_mask)

    # Convert numpy arrays to SimpleITK images
    img_itk = sitk.GetImageFromArray(img_uint8)
    prd_itk = sitk.GetImageFromArray(prd_uint8)
    syn_itk = sitk.GetImageFromArray(syn_uint8)
    lab_itk = sitk.GetImageFromArray(lab_uint8)

    img_itk.SetSpacing((1, 1, 1))
    prd_itk.SetSpacing((1, 1, 1))
    syn_itk.SetSpacing((1, 1, 1))
    lab_itk.SetSpacing((1, 1, 1))

    # Save images
    sitk.WriteImage(prd_itk, os.path.join(save_dir, f'{base_filename}_epoch_{epoch_num}_batch_{batch_idx}_pred.png'))
    sitk.WriteImage(img_itk, os.path.join(save_dir, f'{base_filename}_epoch_{epoch_num}_batch_{batch_idx}_img.png'))
    sitk.WriteImage(syn_itk, os.path.join(save_dir, f'{base_filename}_epoch_{epoch_num}_batch_{batch_idx}_synthetic.png'))
    sitk.WriteImage(lab_itk, os.path.join(save_dir, f'{base_filename}_epoch_{epoch_num}_batch_{batch_idx}_gt.png'))
    
    # plt.figure()
    # plt.imshow(image)
    # plt.title('Original Image')
    # plt.axis('off')
    # plt.savefig(os.path.join(save_dir, f'{base_filename}_epoch_{epoch_num}_batch_{batch_idx}_original_image.png'))
    # plt.close()

    # plt.figure()
    # plt.imshow(original_mask, cmap='gray')
    # plt.title('GT Mask')
    # plt.axis('off')
    # plt.savefig(os.path.join(save_dir, f'{base_filename}_epoch_{epoch_num}_batch_{batch_idx}_original_mask.png'))
    # plt.close()

    # plt.figure()
    # plt.imshow(synthetic_map, cmap='gray')
    # plt.title('Synthetic Map')
    # plt.axis('off')
    # plt.savefig(os.path.join(save_dir, f'{base_filename}_epoch_{epoch_num}_batch_{batch_idx}_synthetic_map.png'))
    # plt.close()

    # plt.figure()
    # plt.imshow(prediction_map, cmap='gray')
    # plt.title('Prediction Map')
    # plt.axis('off')
    # plt.savefig(os.path.join(save_dir, f'{base_filename}_epoch_{epoch_num}_batch_{batch_idx}_prediction_map.png'))
    # plt.close()