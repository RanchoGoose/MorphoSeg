import os
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from torchvision.transforms.functional import to_tensor, to_pil_image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
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
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0

SMOOTH = 1e-6
iou_thresholds = [0.5, 0.75, 0.9]

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Intersection over Union (IoU) for segmentation predictions in PyTorch.
    
    Parameters:
    - outputs: A torch.Tensor of predicted segmentation maps.
    - labels: A torch.Tensor of ground truth segmentation maps.
    
    Returns:
    - A torch.Tensor of IoU scores.
    """
    outputs = outputs.squeeze(1)  # Convert BATCH x 1 x H x W => BATCH x H x W if necessary
    outputs = (outputs > 0).float()  # Ensure binary format
    labels = (labels > 0).float()  # Ensure binary format
    
    intersection = (outputs * labels).sum((1, 2))
    union = outputs.sum((1, 2)) + labels.sum((1, 2)) - intersection
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou

def iou_numpy(outputs: np.array, labels: np.array) -> np.array:
    """
    Calculate the Intersection over Union (IoU) for segmentation predictions in NumPy.
    
    Parameters:
    - outputs: A np.array of predicted segmentation maps.
    - labels: A np.array of ground truth segmentation maps.
    
    Returns:
    - A np.array of IoU scores.
    """
    outputs = np.expand_dims(outputs, axis=0) if outputs.ndim == 2 else outputs
    labels = np.expand_dims(labels, axis=0) if labels.ndim == 2 else labels
    outputs = (outputs > 0).astype(np.float32)
    labels = (labels > 0).astype(np.float32)
    
    intersection = (outputs * labels).sum((1, 2))
    union = outputs.sum((1, 2)) + labels.sum((1, 2)) - intersection
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou

def calculate_iou_ap_per_class(prediction, label, iou_thresholds):
    """
    Calculate IoU for different thresholds and AP for a single class.
    
    Parameters:
    - prediction: Predicted mask for a single class (binary).
    - label: Ground truth mask for a single class (binary).
    - iou_thresholds: List of IoU thresholds.
    
    Returns:
    - iou_scores: IoU scores for each threshold.
    - ap: Average precision across the IoU thresholds.
    """
    iou_score = iou_numpy(prediction, label)
    iou_scores = [iou_score >= threshold for threshold in iou_thresholds]
    ap = np.mean(iou_scores)
    return iou_scores, ap

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    
    prediction = np.zeros_like(label) if len(image.shape) == 3 else None  # Adjust for 2D or 3D
    
    if image.shape[-2] > patch_size[0] or image.shape[-1] > patch_size[1]:
        prediction = infer_large_image_in_patches(image, net, patch_size=[224, 224], overlap=56, device='cuda')
    else:
        if len(image.shape) == 3:
            prediction = np.zeros_like(label)
            for ind in range(image.shape[0]):
                slice = image[ind, :, :]
                x, y = slice.shape[0], slice.shape[1]
                if x != patch_size[0] or y != patch_size[1]:
                    slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
                input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
                net.eval()
                with torch.no_grad():
                    outputs = net(input)
                    out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                    out = out.cpu().detach().numpy()
                    if x != patch_size[0] or y != patch_size[1]:
                        pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                    else:
                        pred = out
                    prediction[ind] = pred
        else:
            input = torch.from_numpy(image).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
                prediction = out.cpu().detach().numpy()
                
    metric_list = []
    ap_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))       
        # pred_mask = (prediction == i).astype(int)
        # true_mask = (label == i).astype(int)
        # iou_scores, ap = calculate_iou_ap_per_class(pred_mask, true_mask, iou_thresholds)
        # metric_list.append(iou_scores)
        # ap_list.append(ap)

    if test_save_path is not None:
        # img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        # prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        # lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        
        # Convert images to uint8
        img_uint8 = convert_to_uint8(image)
        prd_uint8 = convert_to_uint8(prediction)
        lab_uint8 = convert_to_uint8(label)
        # Convert numpy arrays to SimpleITK images
        img_itk = sitk.GetImageFromArray(img_uint8)
        prd_itk = sitk.GetImageFromArray(prd_uint8)
        lab_itk = sitk.GetImageFromArray(lab_uint8)        
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.png")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.png")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.png")
        
        overlay_mask_on_image_and_save(img_uint8, prd_uint8, save_path=test_save_path + '/' + case + "_vis.png", alpha=0.3, mask_color='red', dpi=300, threshold=0.1)

    return metric_list

def convert_to_uint8(image):
    """
    Convert image to uint8 by normalizing its range to [0, 255] and clipping.
    """
    image_min, image_max = np.min(image), np.max(image)
    if image_max > image_min:
        # Normalize to [0.0, 1.0]
        image_normalized = (image - image_min) / (image_max - image_min)
        # Scale to [0, 255] and convert to uint8
        image_uint8 = (image_normalized * 255).astype(np.uint8)
    else:
        # Avoid division by zero if image is constant
        image_uint8 = np.zeros_like(image, dtype=np.uint8)
    return image_uint8

def infer_large_image_in_patches(image, net, patch_size=(224, 224), overlap=56, device='cuda'):
    """
    Infer a large grayscale image by dividing it into patches with overlap, performing
    inference on each patch, and then stitching the patches back together with handling
    for edge patches that might be smaller than the defined patch_size.

    Args:
    - image (numpy.ndarray): Input grayscale image, shape should be (H, W).
    - net (torch.nn.Module): PyTorch model for inference.
    - patch_size (tuple of int): Size of the patches (height, width).
    - overlap (int): Overlap between patches.
    - device (str): Computation device ('cuda' or 'cpu').

    Returns:
    - reconstructed_image (numpy.ndarray): Reconstructed image after patch-wise inference.
    """
    net.eval()
    net.to(device)
    
    H, W = image.shape
    reconstructed_image = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)  # For averaging overlaps

    step = max(patch_size[0] - overlap, 1)
    
    for i in range(0, H, step):
        for j in range(0, W, step):
            # Extract the patch with padding if needed
            end_i = min(i + patch_size[0], H)
            end_j = min(j + patch_size[1], W)
            patch = image[i:end_i, j:end_j]

            # Pad the patch if it's smaller than patch_size
            if patch.shape[0] < patch_size[0] or patch.shape[1] < patch_size[1]:
                pad_height = patch_size[0] - patch.shape[0]
                pad_width = patch_size[1] - patch.shape[1]
                patch = np.pad(patch, ((0, pad_height), (0, pad_width)), 'constant', constant_values=0)

            # Convert patch to tensor and infer
            patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device).float()
            with torch.no_grad():
                output = net(patch_tensor)
                output = torch.softmax(output, dim=1).argmax(dim=1).squeeze().cpu().numpy()

            # Resize output if it was padded
            output = output[:end_i - i, :end_j - j]

            # Add output back to reconstructed image, handling edges correctly
            reconstructed_image[i:end_i, j:end_j] += output
            count_map[i:end_i, j:end_j] += 1

    # Normalize to handle overlaps
    valid_mask = count_map > 0
    reconstructed_image[valid_mask] /= count_map[valid_mask]

    return reconstructed_image
        
def overlay_mask_on_image_and_save(original_image, mask, save_path, alpha=0.5, mask_color='red', dpi=100, threshold=0.5):
    """
    Overlay a mask on an original image with a specified color and save the result,
    with an option to threshold mask values.

    Parameters:
    - original_image: The original image as a numpy array (H, W) or (H, W, C).
    - mask: The mask as a numpy array (H, W), not necessarily binary.
    - save_path: Full path to save the overlay image.
    - alpha: Transparency of the mask overlay.
    - mask_color: Color of the mask overlay.
    - dpi: Dots per inch (resolution) for the saved image.
    - threshold: A value to threshold the mask; values above this are considered mask.
    """
    # Ensure the original image is in uint8
    if original_image.dtype != np.uint8:
        original_image = convert_to_uint8(original_image)

    # Threshold the mask
    binary_mask = np.where(mask > threshold, 1, 0)

    # Normalize the original image for display if needed
    normalized_image = original_image / 255.0 if np.max(original_image) > 1 else original_image

    # Create a color map for the mask: 0s will be transparent, 1s will be colored
    cmap = ListedColormap(['none', mask_color])

    fig, ax = plt.subplots()
    # Display the original image
    ax.imshow(normalized_image, cmap='gray', interpolation='none')
    # Overlay the binary mask with transparency
    ax.imshow(binary_mask, cmap=cmap, alpha=alpha, extent=(0, mask.shape[1], mask.shape[0], 0), interpolation='none')

    plt.axis('off')  # Remove the axis for a cleaner look
    # Calculate figure size to maintain original resolution
    fig_width = original_image.shape[1] / dpi
    fig_height = original_image.shape[0] / dpi
    fig.set_size_inches(fig_width, fig_height)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
   
    
