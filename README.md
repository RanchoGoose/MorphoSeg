# Cellseg_UoS
This repo holds code for stem cell segmentation. The codes are built based on the original TransUNet repo: [TransUNet](https://github.com/Beckschen/TransUNet).

## Usage

### 1. Download pre-trained models
To train the model please down the Google ViT pre-trained models
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16, ViT-B_16, ViT-L_16...
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz &&
mkdir ../model/vit_checkpoint/imagenet21k &&
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

*For inference only, download the provided [checkpoint](https://drive.google.com/drive/folders/1A2fYP5uPjevKxKek0pneYLQzUPSQISua?usp=sharing)

### 2. Prepare data
Generate Train and Test Lists:

Download the data dirst, then run the following script to spilit the data into training and testing 224x224 patches, you will need to change the data path in the script:

```bash
python data_pre.py --mode train_test --dataset_dir /path/to/dataset --lists_dir /path/to/lists --train_ratio 0.8

```

For inference:
Generate Eval List:

```bash
python data_pre.py --mode eval --image_dir /path/to/images --mask_dir /path/to/masks --output_dir /path/to/output --lists_dir /path/to/lists


```

### 3. Environment

Please prepare an environment with python=3.10, and then use the command "pip install -r requirements.txt" for the dependencies.

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

## Reference
[TransUNet](https://arxiv.org/pdf/2102.04306.pdf)

## Citations

<!-- ```bibtex

``` -->
