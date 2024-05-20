import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt, gaussian_filter

def generate_flows(masks):
    """
    Generate flow fields from masks.
    Args:
        masks (numpy.ndarray): Binary mask image where each cell is marked with a unique integer.
    Returns:
        numpy.ndarray: The flow fields derived from the masks.
    """
    flows = np.zeros_like(masks, dtype=np.float32)
    
    for label in np.unique(masks):
        if label == 0:
            continue  # Skip background
        mask = (masks == label)
        dist = distance_transform_edt(mask)
        grad_x, grad_y = np.gradient(dist)
        flow = np.stack((grad_x, grad_y), axis=0)
        flows += flow

    return flows

def generate_centers(masks, sigma=3):
    """
    Generate center maps from masks using Gaussian peaks.
    Args:
        masks (numpy.ndarray): Binary mask image where each cell is marked with a unique integer.
        sigma (float): Standard deviation for Gaussian filter.
    Returns:
        numpy.ndarray: The center maps derived from the masks.
    """
    centers = np.zeros_like(masks, dtype=np.float32)
    
    for label in np.unique(masks):
        if label == 0:
            continue  # Skip background
        mask = (masks == label)
        # Find centroid
        M = cv2.moments(mask.astype(np.uint8))
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers[cy, cx] = 1  # Place a peak at the centroid
    
    # Apply Gaussian filter to create a smooth peak
    centers = gaussian_filter(centers, sigma=sigma)
    return centers
