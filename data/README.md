# Data Preparation 

Before running our codes, the datasets for training and evaluation should be prepared for audio reconstruction, image reconstruction, and novel view synthesis. Since we assume that all datasets are located in `data`, please download each dataset following the instructions below.  

## Audio Reconstruction
We use the [LibriSpeech](https://ieeexplore.ieee.org/document/7178964) dataset for the task of audio reconstruction, while we trimmed each audio sample into a fixed length of audio (1s and 3d). Our code will automatically download the `train-clean-100` split for training and `test-clean` split for evaluation. If you have already downloaded the datasets, please make a symbolic link. The dataset has to be located in `data/LIBRISPEECH` . 


## Image Reconstruction  
We use three datasets, FFHQ, CelebA, and ImageNette, for the task of image reconstruction.   

### FFHQ
Please refer to codes in [the original repository of FFHQ](https://github.com/NVlabs/ffhq-dataset) and download 70,000 images of human faces. The dataset should be located in `data/FFHQ`. Then, our codes automatically split training and evaluation splits based on the indexes in `src/datasets/assets/ffhqtrain.txt` and `src/datasets/assets/ffhqvalidation.txt` .

### CelebA
Download the CelebA dataset in [Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) and extract. The dataset should be located in `data/CelebA_aligned`. 

### ImageNette
Download the dataset from [this link](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz) and extract it. The dataset should be located in `data/imagenette`. 

## Novel View Synthesis
For the task of novel view synthesis, we use ShapeNet-cars, ShapeNet-chairs, and ShapeNet-lamps. Please download the datasets from [the google drive link](https://drive.google.com/drive/folders/1lRfg-Ov1dd3ldke9Gv9dyzGGTxiFOhIs) and unzip. The dataset should be located in `/data/learnit_shapenet/cars`, `/data/learnit_shapenet/chairs`, and `/data/learnit_shapenet/lamps`, while JSON files are located in `/data/learnit_shapenet/`. 


## Acknowledgement
We appreciate the authors of [TransINR](https://github.com/yinboc/trans-inr) for preparing the instruction for CelebA and ImageNette. In addition, we thank the authors of [Learned Init](https://github.com/tancik/learnit) for preparing the ShapeNet datasets.