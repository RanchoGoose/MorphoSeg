from PIL import Image
import os
import shutil
import random
import numpy as np

def has_annotation(mask):
    mask_array = np.array(mask)
    return np.any(mask_array > 0)

def divide_images_and_masks(image_dir, mask_dir, output_dir, patch_sizes=[224], overlap=0.1, output_size=224):
    os.makedirs(output_dir, exist_ok=True)
    full_image_list = []
    full_mask_list = []

    for filename in os.listdir(image_dir):
        if not filename.endswith('.tif'): continue  # Adjust the extension according to your files
        image_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename.replace('.tif', '_mask.tif'))  # Adjust naming convention as needed

        image = Image.open(image_path)
        mask = Image.open(mask_path)
        width, height = image.size
        
        # Convert images to a supported mode before saving
        image = image.convert('L')
        mask = mask.convert('L')

        # Save the full-size image and mask
        full_image_path = os.path.join(output_dir, filename.replace('.tif', '.png'))
        full_mask_path = os.path.join(output_dir, filename.replace('.tif', '_mask.png'))
        image.save(full_image_path)
        mask.save(full_mask_path)
        full_image_list.append(full_image_path)
        full_mask_list.append(full_mask_path)

        # Proceed with dividing the image into patches
        for patch_size in patch_sizes:
            # Ensure the patch size is smaller than the image dimensions
            if patch_size < min(width, height):
                step_size = int(patch_size * (1 - overlap))  # Calculate step size based on overlap
                for i in range(0, height - patch_size + 1, step_size):
                    for j in range(0, width - patch_size + 1, step_size):
                        image_patch = image.crop((j, i, j + patch_size, i + patch_size)).resize((output_size, output_size))
                        mask_patch = mask.crop((j, i, j + patch_size, i + patch_size)).resize((output_size, output_size))

                        # Prune patches without annotations
                        if has_annotation(mask_patch):
                            image_patch = image_patch.convert('L')  # Convert to grayscale if needed
                            mask_patch = mask_patch.convert('L')  # Convert to grayscale if needed

                            patch_filename = f'{filename[:-4]}_ps{patch_size}_{i}_{j}.png'  # Include patch size in filename
                            image_patch.save(os.path.join(output_dir, patch_filename))
                            mask_patch.save(os.path.join(output_dir, patch_filename.replace('.png', '_mask.png')))

    return full_image_list, full_mask_list

image_dir = '/mnt/parscratch/users/coq20tz/cellpose/data_raw/img'
mask_dir = '/mnt/parscratch/users/coq20tz/cellpose/data_raw/binary_mask'
output_dir = '/mnt/parscratch/users/coq20tz/TransUNet/data/cell_arg'

# Divide images and masks and get lists of undivided images and masks
full_image_list, full_mask_list = divide_images_and_masks(image_dir, mask_dir, output_dir, patch_sizes=[224, 448, 1000, 1500, 2000], overlap=0.35, output_size=224)

# divide_images_and_masks(image_dir, mask_dir, output_dir, patch_sizes=[224, 448, 1000, 1500, 2500], overlap=0.35, output_size=224)


# Specify your dataset directory and the desired split ratio
dataset_dir = output_dir
lists_dir = '/mnt/parscratch/users/coq20tz/TransUNet/lists/cellseg'
train_ratio = 0.8  # 80% for training, 20% for testing
 
# List all data files, assuming they are stored in 'images' subdirectory
# Adjust this line if your data files are located differently
data_files = [f for f in os.listdir(dataset_dir) if f.endswith('.png') and not f.endswith('_mask.png')]

# Separate full images from patches
full_images = [f for f in data_files if not any(sub in f for sub in ['_ps224_', '_ps448_', '_ps1000_', '_ps1500_', '_ps2000_'])]
patches = [f for f in data_files if any(sub in f for sub in ['_ps224_', '_ps448_', '_ps1000_', '_ps1500_', '_ps2000_'])]

# Shuffle data files to ensure random distribution
random.shuffle(full_images)
random.shuffle(patches)

# Split full images based on the specified train_ratio
split_index_full = int(len(full_images) * train_ratio)
train_files_full = full_images[:split_index_full]
test_files_full = full_images[split_index_full:]

# Split patches based on the specified train_ratio
split_index_patches = int(len(patches) * train_ratio)
train_files_patches = patches[:split_index_patches]
test_files_patches = patches[split_index_patches:]

# Combine full images and patches for training and testing
train_files = train_files_full + train_files_patches
test_files = test_files_full + test_files_patches

# Function to write file paths to a text file
def write_paths(file_list, file_path):
    with open(file_path, 'w') as file:
        for item in file_list:
            file.write("%s\n" % item)

# Write train and test file lists to 'train.txt' and 'test.txt'
write_paths(train_files_full, os.path.join(lists_dir, 'train_full.txt'))
write_paths(test_files_full, os.path.join(lists_dir,  'test_full.txt'))
write_paths(train_files, os.path.join(lists_dir, 'train.txt'))
write_paths(test_files, os.path.join(lists_dir,  'test.txt'))


def generate_or_load_splits(data_dir, list_dir):
    eval_list_path = os.path.join(list_dir, 'eval.txt')

    if os.path.exists(eval_list_path):
        return
    
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.tif') and not f.endswith('_mask.tif')]

    with open(eval_list_path, 'w') as f:
        for item in all_files:
            f.write("%s\n" % item)

data_dir = '/mnt/parscratch/users/coq20tz/cellpose/data_raw/full_img'
generate_or_load_splits(data_dir, list_dir)