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
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import disk, binary_dilation, binary_erosion, opening, closing

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
    - A float IoU score (scalar value, not an array)
    """
    outputs = np.expand_dims(outputs, axis=0) if outputs.ndim == 2 else outputs
    labels = np.expand_dims(labels, axis=0) if labels.ndim == 2 else labels
    outputs = (outputs > 0).astype(np.float32)
    labels = (labels > 0).astype(np.float32)
    
    intersection = (outputs * labels).sum((1, 2))
    union = outputs.sum((1, 2)) + labels.sum((1, 2)) - intersection
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    # Return the first element as a scalar float value, not an array
    return float(iou[0])

def calculate_iou_ap_per_class(prediction, label, iou_thresholds):
    """
    Calculate IoU for different thresholds and AP for a single class.
    
    Parameters:
    - prediction: Predicted mask for a single class (binary).
    - label: Ground truth mask for a single class (binary).
    - iou_thresholds: List of IoU thresholds.
    
    Returns:
    - iou_scores: List of boolean values indicating if IoU exceeds each threshold.
    - ap: Average precision across the IoU thresholds.
    """
    # Calculate the single IoU score between prediction and label
    iou_score = iou_numpy(prediction, label)
    
    # For each threshold, check if the IoU score exceeds it
    iou_scores = [iou_score >= threshold for threshold in iou_thresholds]
    
    # AP is simply the average of these boolean values (1.0 for each threshold that's met)
    ap = float(np.mean(iou_scores))
    
    # Return a list of python scalar values, not numpy arrays
    return [float(iou_score) for _ in iou_thresholds], ap

def test_single_volume(image, label, net, classes, patch_size=[224, 224], test_save_path=None, case=None, z_spacing=1, 
                       iou_thresholds=[0.5, 0.75, 0.9]):
    """
    Process a single volume or image for testing.
    Uses patch-based inference for large images and preserves original image dimensions.
    For images that match the patch size (224x224), it uses direct inference.
    """
    # Convert to numpy arrays
    if isinstance(image, torch.Tensor):
        image_np = image.squeeze(0).cpu().detach().numpy()
    else:
        image_np = image
        
    if isinstance(label, torch.Tensor):
        label_np = label.squeeze(0).cpu().detach().numpy()
    else:
        label_np = label
    
    # Handle different input dimensions
    if len(image_np.shape) == 3:
        # If the first dimension is 1, it's likely a 2D image with a channel dimension
        if image_np.shape[0] == 1:
            # Get the 2D image by removing the channel dimension
            slice_data = image_np[0]
            
            # Check if we need patch-based inference
            if slice_data.shape[0] > patch_size[0] or slice_data.shape[1] > patch_size[1]:
                prediction = infer_large_image_in_patches(slice_data, net, patch_size=patch_size, 
                                                         overlap=56, device='cuda')
            else:
                # Direct inference for images that fit in memory
                input_tensor = torch.from_numpy(slice_data).unsqueeze(0).unsqueeze(0).float().cuda()
                with torch.no_grad():
                    outputs = net(input_tensor)
                    prediction = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze().cpu().numpy()
        else:
            # True 3D volume with multiple slices
            prediction = np.zeros_like(label_np)
            
            # Process each slice
            for ind in range(image_np.shape[0]):
                slice_data = image_np[ind]
                
                # Check if slice needs patch-based inference
                if slice_data.shape[0] > patch_size[0] or slice_data.shape[1] > patch_size[1]:
                    pred_slice = infer_large_image_in_patches(slice_data, net, patch_size=patch_size,
                                                            overlap=56, device='cuda')
                else:
                    # Direct inference
                    input_tensor = torch.from_numpy(slice_data).unsqueeze(0).unsqueeze(0).float().cuda()
                    with torch.no_grad():
                        outputs = net(input_tensor)
                        pred_slice = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze().cpu().numpy()
                
                # Safely assign to prediction - make sure dimensions match
                if len(prediction.shape) == 3:
                    if ind < prediction.shape[0]:
                        prediction[ind] = pred_slice
                else:
                    # If prediction is 2D but we're processing multiple slices, 
                    # we'll just use the last processed slice
                    prediction = pred_slice
    else:
        # 2D image
        # Check if image needs patch-based inference
        if image_np.shape[0] > patch_size[0] or image_np.shape[1] > patch_size[1]:
            prediction = infer_large_image_in_patches(image_np, net, patch_size=patch_size,
                                                   overlap=56, device='cuda')
        else:
            # Direct inference
            input_tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).float().cuda()
            with torch.no_grad():
                outputs = net(input_tensor)
                prediction = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze().cpu().numpy()
    
    # IMPORTANT: metric_list is a FLAT list of metrics, not a list of tuples
    # Each class will contribute two values: dice and hd95, in order
    metric_list = []
    iou_scores = {thr: [] for thr in iou_thresholds}
    aps = {thr: [] for thr in iou_thresholds}
    
    # Calculate metrics for each class
    for i in range(1, classes):
        try:
            # Calculate Dice and HD95
            if prediction is not None and label_np is not None:
                # Create binary masks for class i
                pred_mask = (prediction == i)
                gt_mask = (label_np == i)
                
                # Calculate metrics
                dice, hd95 = calculate_metric_percase(pred_mask, gt_mask)
                
                # IMPORTANT: Append each metric individually (not as a tuple)
                # This matches the expected format in test.py
                metric_list.append(dice)
                metric_list.append(hd95)
                
                # Calculate IoU for this class
                pred_mask_int = pred_mask.astype(int)
                gt_mask_int = gt_mask.astype(int)
                
                # Calculate IoU scores and AP
                iou_scores_list, ap = calculate_iou_ap_per_class(pred_mask_int, gt_mask_int, iou_thresholds)
                
                # Store results - ensure the values are Python scalar floats, not numpy arrays
                for idx, thr in enumerate(iou_thresholds):
                    iou_scores[thr].append(float(iou_scores_list[idx]))
                for thr in iou_thresholds:
                    aps[thr].append(float(ap))
            else:
                # If prediction or label is None, add default values
                metric_list.append(0.0)  # dice
                metric_list.append(0.0)  # hd95
                for thr in iou_thresholds:
                    iou_scores[thr].append(0.0)
                    aps[thr].append(0.0)
                
        except Exception as e:
            print(f"Error calculating metrics for class {i}: {e}")
            # Add default values if an error occurs
            metric_list.append(0.0)  # dice
            metric_list.append(0.0)  # hd95
            for thr in iou_thresholds:
                iou_scores[thr].append(0.0)
                aps[thr].append(0.0)
    
    # If no metrics were calculated, add default values
    if len(metric_list) == 0:
        for i in range(1, classes):
            metric_list.append(0.0)  # dice
            metric_list.append(0.0)  # hd95
            for thr in iou_thresholds:
                iou_scores[thr].append(0.0)
                aps[thr].append(0.0)
    
    # Save visualizations if requested
    if test_save_path is not None:
        try:
            # Convert images to uint8
            img_uint8 = convert_to_uint8(image_np)
            prd_uint8 = convert_to_uint8(prediction)
            lab_uint8 = convert_to_uint8(label_np)
            
            # For 3D volumes, take the middle slice for visualization
            if len(img_uint8.shape) == 3 and img_uint8.shape[0] > 1:
                middle_slice = img_uint8.shape[0] // 2
                img_for_vis = img_uint8[middle_slice]
                prd_for_vis = prd_uint8[middle_slice]
                lab_for_vis = lab_uint8[middle_slice]
            else:
                img_for_vis = img_uint8
                prd_for_vis = prd_uint8
                lab_for_vis = lab_uint8
            
            # Convert numpy arrays to SimpleITK images for saving
            img_itk = sitk.GetImageFromArray(img_uint8)
            prd_itk = sitk.GetImageFromArray(prd_uint8)
            lab_itk = sitk.GetImageFromArray(lab_uint8)        
            img_itk.SetSpacing((1, 1, z_spacing))
            prd_itk.SetSpacing((1, 1, z_spacing))
            lab_itk.SetSpacing((1, 1, z_spacing))
            
            # Save the images
            sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.png")
            sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.png")
            sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.png")
            
            # Create overlay visualization
            overlay_mask_on_image_and_save(img_for_vis, prd_for_vis, 
                                          save_path=test_save_path + '/' + case + "_vis.png", 
                                          alpha=0.5, dpi=300, threshold=0.5, 
                                          use_different_colors=True)
        except Exception as e:
            print(f"Error saving visualization: {e}")
            import traceback
            traceback.print_exc()

    return metric_list, iou_scores, aps

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
    - image (numpy.ndarray): Input grayscale image, shape can be (C, H, W) or (H, W).
    - net (torch.nn.Module): PyTorch model for inference.
    - patch_size (tuple of int): Size of the patches (height, width).
    - overlap (int): Overlap between patches.
    - device (str): Computation device ('cuda' or 'cpu').

    Returns:
    - reconstructed_image (numpy.ndarray): Reconstructed image after patch-wise inference.
    """
    net.eval()
    net.to(device)
    
    # Get the original image shape to ensure we return the same shape
    original_shape = image.shape
    
    
    # Handle image dimensions - Check if we have a 3D image (C, H, W) or 2D image (H, W)
    if len(original_shape) == 3:
        # For multi-channel or multi-slice images (C, H, W)
        C, H, W = original_shape
        # We'll process the first channel/slice - typically for medical images
        # this is what we want. Alternatively, we could process each channel.
        image_to_process = image[0]  # Take the first channel/slice
        reconstructed_image = np.zeros((H, W), dtype=np.float32)
    elif len(original_shape) == 2:
        # For 2D grayscale images (H, W)
        H, W = original_shape
        image_to_process = image
        reconstructed_image = np.zeros((H, W), dtype=np.float32)
    elif len(original_shape) == 1:
        H = 1
        W = original_shape[0]
        image_to_process = image.reshape(H, W)
        reconstructed_image = np.zeros((H, W), dtype=np.float32)
    else:
        raise ValueError(f"Unexpected image shape: {original_shape}. Expected (C, H, W), (H, W), or (W,)")
    
    count_map = np.zeros((H, W), dtype=np.float32)  # For averaging overlaps

    step = max(patch_size[0] - overlap, 1)
    
    for i in range(0, H, step):
        for j in range(0, W, step):
            # Extract the patch with padding if needed
            end_i = min(i + patch_size[0], H)
            end_j = min(j + patch_size[1], W)
            patch = image_to_process[i:end_i, j:end_j]

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

    # Reshape the output to match the original dimensions
    if len(original_shape) == 1:
        # For 1D input, return a 1D output
        reconstructed_image = reconstructed_image.reshape(original_shape)
    elif len(original_shape) == 2:
        # For 2D input, ensure the shape matches
        if reconstructed_image.shape != original_shape:
            reconstructed_image = zoom(reconstructed_image, 
                                      (original_shape[0] / reconstructed_image.shape[0], 
                                       original_shape[1] / reconstructed_image.shape[1]), 
                                      order=0)
    
    return reconstructed_image
        
def overlay_mask_on_image_and_save(original_image, mask, save_path=None, alpha=0.7, mask_color='red', dpi=100, threshold=0.5, use_different_colors=True):
    """
    Overlay a mask on an original image with colors to distinguish different cell instances.
    Uses connected component labeling to ensure all segmented cells are properly visualized.
    
    Parameters:
    - original_image: The original image as a numpy array (H, W) or (H, W, C) or (C, H, W).
    - mask: The mask as a numpy array (H, W), where positive values indicate cells.
    - save_path: Full path to save the overlay image. If None, the image will be shown instead.
    - alpha: Transparency of the mask overlay.
    - mask_color: Color of the mask overlay (used only if use_different_colors=False).
    - dpi: Dots per inch (resolution) for the saved image.
    - threshold: A value to threshold the mask (between 0 and 1); values above threshold*max_value are considered cells.
    - use_different_colors: If True, assign different colors to different cell instances.
    """
    from scipy import ndimage as ndi
    
    # Handle 3D images (C, H, W) by taking the first channel
    if len(original_image.shape) == 3:
        # Check if it's (C, H, W) or (H, W, C)
        if original_image.shape[0] < original_image.shape[1] and original_image.shape[0] < original_image.shape[2]:
            # Likely (C, H, W) format, take the first channel
            original_image = original_image[0]
        else:
            # Could be (H, W, C) format for RGB images
            pass
    
    # Similarly handle 3D masks
    if len(mask.shape) == 3:
        if mask.shape[0] < mask.shape[1] and mask.shape[0] < mask.shape[2]:
            mask = mask[0]
    
    # Ensure the original image is in uint8
    if original_image.dtype != np.uint8:
        original_image = convert_to_uint8(original_image)
    
    # Get min and max values of mask for proper scaling
    mask_min = np.min(mask)
    mask_max = np.max(mask)
    unique_values = np.unique(mask)

    # Handle thresholding based on the mask properties
    if len(unique_values) <= 10 and np.issubdtype(mask.dtype, np.integer):
        # For masks with few distinct integer values (likely class labels)
        # Scale the threshold between 0 and max_value
        scaled_threshold = int(threshold * mask_max)
        binary_mask = (mask >= scaled_threshold).astype(np.uint8)

    else:
        # For continuous-valued masks or probability masks
        if np.issubdtype(mask.dtype, np.floating):
            # Handle floating point masks (0.0-1.0 range)
            binary_mask = (mask > threshold).astype(np.uint8)

        else:
            # For other numeric types, scale the threshold to the data range
            scaled_threshold = mask_min + threshold * (mask_max - mask_min)
            binary_mask = (mask >= scaled_threshold).astype(np.uint8)

    # Normalize the original image for display
    normalized_image = original_image / 255.0 if np.max(original_image) > 1 else original_image
    
    fig, ax = plt.subplots()
    # Display the original image
    ax.imshow(normalized_image, cmap='gray', interpolation='none')
    
    if use_different_colors:
        # Create cellpose-like colormap - vibrant distinct colors
        def create_cellpose_cmap(n_colors=256):
            # Generate a colormap similar to cellpose with vibrant, distinguishable colors
            import colorsys
            
            # Use HSV color space for more vibrant colors
            hsv_colors = []
            for i in range(n_colors):
                # Distribute hues evenly around the color wheel
                h = i / n_colors
                # Full saturation for vibrant colors
                s = 0.9
                # Value (brightness) - keep high for visibility
                v = 0.9
                hsv_colors.append((h, s, v))
            
            # Convert HSV to RGB
            rgb_colors = [colorsys.hsv_to_rgb(*hsv) for hsv in hsv_colors]
            
            # Create a ListedColormap
            return plt.cm.colors.ListedColormap(rgb_colors)
            
        # Simply use connected components to label each distinct region
        labeled_mask, num_cells = ndi.label(binary_mask)
        print(f"Connected components analysis found {num_cells} distinct regions")
        
        # Create a cellpose-like colormap
        cellpose_cmap = create_cellpose_cmap(n_colors=256)
        
        # Create a colored overlay for all cells
        overlay = np.zeros((*labeled_mask.shape, 4))  # RGBA
        
        # For each cell, assign a color with good spacing for distinctiveness
        for i in range(1, num_cells + 1):
            # Get color for this cell - use a good spacing to avoid similar colors next to each other
            color_idx = int((i * 67) % 256)  # Use prime number multiplication for better distribution
            cell_color = np.array(cellpose_cmap(color_idx))
            
            # Apply the color to this cell in the overlay
            cell_mask = (labeled_mask == i)
            overlay[cell_mask] = cell_color
            
            # Set alpha value for this cell
            overlay[cell_mask, 3] = alpha
        
        # Areas with no mask should be transparent
        overlay[labeled_mask == 0, 3] = 0
        
        # Display the colored overlay on top of the image
        ax.imshow(overlay, interpolation='none')
        
    else:
        # Use a single color for all mask values above threshold
        cmap = ListedColormap(['none', mask_color])
        ax.imshow(binary_mask, cmap=cmap, alpha=alpha, 
                 extent=(0, binary_mask.shape[1], binary_mask.shape[0], 0), 
                 interpolation='none')

    plt.axis('off')  # Remove the axis for a cleaner look
    
    # Calculate figure size to maintain original resolution
    fig_width = normalized_image.shape[1] / dpi
    fig_height = normalized_image.shape[0] / dpi
    fig.set_size_inches(fig_width, fig_height)

    # Add title with number of cells if using different colors
    if use_different_colors:
        plt.title(f"Found {num_cells} cells", fontsize=10)

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()
    else:
        plt.show()
   
    
