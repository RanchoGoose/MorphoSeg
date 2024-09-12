# MorphoSeg
This repository holds code for stem cell segmentation. The codes are built based on the original TransUNet repo: [TransUNet](https://github.com/Beckschen/TransUNet).

Our dataset is available at https://orda.shef.ac.uk/account/projects/201540/articles/25604421 

## Visualization Example
![Example Image](example/example.png)

## Usage

### 1. Prepare the environment

Please prepare an environment with python=3.10, We recommand using Anaconda3. 

After installing python(Anaconda), please run the following command for installing the dependencies.

```bash
pip install -r requirements.txt
```

### 2. Prepare pretrained model

To train your own model from zero please download the Google ViT pre-trained models
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16, ViT-B_16, ViT-L_16...
or run the following commands replacing the {MODEL_NAME} to one of the R50-ViT-B_16, ViT-B_16, ViT-L_16..
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz &&
mkdir ../model/vit_checkpoint/imagenet21k &&
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

*For inference only, download the provided [checkpoint](https://drive.google.com/drive/folders/1A2fYP5uPjevKxKek0pneYLQzUPSQISua?usp=sharing), you will need to download the whole folder and put it under the ./model/ path. Create model folder if you dont have one.

### 3. Prepare data

Prepare the data and generate Train and Test Lists:

Download the data first, then run the following script to spilit the data into training and testing 224x224 patches, you will need to change the data path in the script:

```bash
python data_pre.py --mode train_test --divide --image_dir /path/to/images --mask_dir /path/to/masks --dataset_dir /path/to/training_and_testing_patches --lists_dir /path/to/lists
```

For multi images inference only:
Generate Eval List by running:

```bash
python data_pre.py --mode eval --image_dir /path/to/images --lists_dir /path/to/lists
```

The whole data and the lists should be under a directory structure like this:
```bash
Cellseg_UOS/
├── data/
│   └── Cell/
│       ├── image1.png
│       ├── image1_mask.png
│       ├── image1_ps224_0_0.png
│       ├── image1_ps224_0_0_mask.png
│       ├── image1_ps448_0_0.png
│       ├── image1_ps448_0_0_mask.png
│       ├── ...
│       ├── image2.png
│       ├── image2_mask.png
│       ├── image2_ps224_0_0.png
│       ├── image2_ps224_0_0_mask.png
│       ├── image2_ps448_0_0.png
│       ├── image2_ps448_0_0_mask.png
│       ├── ...
├── model/
│   ├── vit_checkpoint/
│   │   └── imagenet21k/
│   │       ├── R50+ViT-B_16.npz       # Pretrained model checkpoint
│   │       └── *.npz                  # Other model checkpoints
│   └── TU_CellSeg224/
│       └── TU_pretrain_R50-ViT-B_16_skip0_epo200_bs128_224_St150_SN100_SEL10000_SF100000_LTpareto_VOS/
│           ├── epoch_199.pth          # Model weights
│           └── ...                        
└── lists/
    ├── train.txt
    ├── test.txt
    ├── eval.txt
├── datasets/
├── vis/
├── data_pre.py
├── train.py
├── test.py
├── ...
```

### 4. Train/Test

Example to train the model with outlier argument:

```bash
python train.py --batch_size 256 --max_epochs 200 --vit_name ViT-B_16 --start_epoch 150 --select 10000 --sample_from 100000 --loss_type norm --use_vos
```

- Run the test with the same configs on the test set:

```bash
python test.py --batch_size 128 --max_epochs 200 --vit_name R50-ViT-B_16 --start_epoch 150 --select 10000 --sample_from 100000 --loss_type $loss_type --use_vos
```

Visualize the high resolution raw images using the model trained by:
```bash
python test.py --volume_path 'your path to images' --max_epochs 200 --batch_size 256 --vit_name R50-ViT-B_16 --is_savenii --data_split eval --start_epoch 150 --select 10000 --sample_from 100000 --loss_type norm --use_vos
```

### 5. Single image visualization demo

For single image visualization, please open the inference.ipynb in jupyter notebook, put your testing images under the folder ./vis/, and then run the notebook for results. Note that you still need to install the dependencies first (Step 1). 

Please note that if you are not using the provided checkpoint, you will need to change the snapshot_path and other configs in the notebook.


## Citations

<!-- ```bibtex

``` -->
