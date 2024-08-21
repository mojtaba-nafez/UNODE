# Universal Novelty Detection Through Adaptive Contrastive Learning

Official PyTorch implementation of
["**Universal Novelty Detection Through Adaptive Contrastive Learning**"](https://openaccess.thecvf.com/content/CVPR2024/html/Mirzaei_Universal_Novelty_Detection_Through_Adaptive_Contrastive_Learning_CVPR_2024_paper.html) (CVPR 2024) by
[Hossein Mirzaei](),
[Mojtaba Nafez](),
[Mohammad Jafari](),
[Mohammad Bagher Soltani](),
[Jafar Habibi](),
[Mohammad Sabokrou](),
and [MohammadHossein Rohban]().

<p align="center">
    <img src=figures/method.png width="500"> 
</p>

## Requirements
### Environments
- [torchlars](https://github.com/kakaobrain/torchlars) == 0.1.2 

### Datasets 

Dataset Download Link:

* [MVTecAD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
* [ISIC2018](https://www.kaggle.com/datasets/maxjen/isic-task3-dataset)
* [ImageNet-30-train](https://drive.google.com/file/d/1B5c39Fc3haOPzlehzmpTLz6xLtGyKEy4/view),
[ImageNet-30-test](https://drive.google.com/file/d/13xzVuQMEhSnBRZr-YaaO08coLU2dxAUq/view)


## Training

To train our models, run this command:

```train
python train.py --model $model --epochs $epochs --eval_steps $eval_steps --normal_class $normal_class --normal_data_count $normal_data_count --image_size $image_size --dataset $dataset --batch_size $batch_size --outlier_dataset $outlier_dataset
```

* The option `--normal_class` denotes the in-distribution for one-class training.

* For multi-class training, set `--outlier_dataset` as the OOD target dataset, and --dataset will be the determined ID dataset.

> Note: The `config.json` file specifies the probability of each negative transformation used during training. This probability distribution is determined by the AutoAugOOD module in  `AutoAugOOD.ipynb` individually for each dataset and each normal class.

## Evaluation

We provide the checkpoint of the Unode pre-trained model. Download the checkpoint from the following link:
- One-class CIFAR-10: [Wide-Res](https://drive.google.com/drive/folders/1-vmaK398GWxdyNJbXObeVyHYWzszT7GY?usp=sharing)
- One-class MVtecAD: [ResNet-18](https://drive.google.com/drive/folders/1--lOGcKV0LGbI_qV9DIt-ifUr-KYZOe6?usp=sharing)

To evaluate our model, run this command:

```eval
python ./eval.py --normal_class $normal_class --image_size $image_size --dataset $dataset --model $model --print_score --ood_samples 10 --resize_factor 0.54 --resize_fix --load_path $load_path 
```


* The option `--normal_class` denotes the in-distribution for one-class training.

* For multi-class training, set `--outlier_dataset` as the OOD target dataset, and --dataset will be the determined ID dataset.

* The resize_factor & resize fix option fix the cropping size of RandomResizedCrop().

## Notebook

[Google Colab notebook](https://colab.research.google.com/drive/1p2fLr6dR8A5VobiNm6tviKoWFCoP4cvN?usp=sharing) as an example of training and evaluation.


## Citation
```
@InProceedings{Mirzaei_2024_CVPR,
    author    = {Mirzaei, Hossein and Nafez, Mojtaba and Jafari, Mohammad and Soltani, Mohammad Bagher and Azizmalayeri, Mohammad and Habibi, Jafar and Sabokrou, Mohammad and Rohban, Mohammad Hossein},
    title     = {Universal Novelty Detection Through Adaptive Contrastive Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {22914-22923}
}
```
