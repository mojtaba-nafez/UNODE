import os

import numpy as np
import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from utils.utils import set_random_seed
from PIL import Image
from glob import glob
import random

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from datasets.custom_datasets import *
from torch.utils.data import ConcatDataset

DATA_PATH = './data/'
IMAGENET_PATH = './data/ImageNet'


CIFAR10_SUPERCLASS = list(range(10))  # one class
IMAGENET_SUPERCLASS = list(range(30))  # one class
MNIST_SUPERCLASS = list(range(10))
SVHN_SUPERCLASS = list(range(10))
FashionMNIST_SUPERCLASS = list(range(10))
HEAD_CT_SUPERCLASS = list(range(2))
MVTEC_HV_SUPERCLASS = list(range(2))
CIFAR100_SUPERCLASS = list(range(20))
CIFAR10_CORRUPTION_SUPERCLASS = list(range(10))
MNIST_CORRUPTION_SUPERCLASS = list(range(10))
CIFAR10_VER_CIFAR100_SUPERCLASS = list(range(2))
ISIC2018_SUPERCLASS = list(range(2))

def sparse2coarse(targets):
    coarse_labels = np.array(
        [4,1,14, 8, 0, 6, 7, 7, 18, 3, 3,
         14, 9, 18, 7, 11, 3, 9, 7, 11, 6, 11, 5,
         10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10,
         12, 14, 16, 9, 11, 5, 5, 19, 8, 8, 15,
         13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0,
         17, 4, 18, 17, 10, 3, 2, 12, 12, 16, 12,
         1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13,
         15, 13, 16, 19, 2, 4, 6, 19, 5, 5, 8,
         19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13,])
    return coarse_labels[targets]

CLASS_NAMES = ['toothbrush', 'zipper', 'transistor', 'tile', 'grid', 'wood', 'pill', 'bottle', 'capsule', 'metal_nut', 'hazelnut', 'screw', 'carpet', 'leather', 'cable']

def get_transform(image_size=None):
    if image_size:  # use pre-specified image size

        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.ToTensor()

    return train_transform, test_transform


def get_subset_with_len(dataset, length, shuffle=False):
    set_random_seed(0)
    dataset_size = len(dataset)

    index = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(index)

    index = torch.from_numpy(index[0:length])
    subset = Subset(dataset, index)

    assert len(subset) == length

    return subset


def get_transform_imagenet():

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_transform = MultiDataTransform(train_transform)

    return train_transform, test_transform



def get_dataset(P, dataset, image_size=(32, 32, 3), download=False, eval=False, labels=None):
    if dataset in ['imagenet']:
        if eval:
            train_transform, test_transform = get_simclr_eval_transform_imagenet(P.ood_samples, P.resize_factor, P.resize_fix)
        else:
            train_transform, test_transform = get_transform_imagenet()
    else:
        train_transform, test_transform = get_transform(image_size=image_size)

    if dataset == 'cifar10':
        n_classes = 10
        
        train_set = datasets.CIFAR10(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR10(DATA_PATH, train=False, download=download, transform=test_transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    
    elif dataset == 'cifar10-versus-100':
        n_classes = 2
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_set = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)

        for i in range(len(train_set)):
            train_set.targets[i] = 0
        
        anomaly_testset = datasets.CIFAR100('./data', train=False, download=True, transform=transform)
        for i in range(len(anomaly_testset)):
            anomaly_testset.targets[i] = 1
        normal_testset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        for i in range(len(normal_testset)):
            normal_testset.targets[i] = 0
        test_set = torch.utils.data.ConcatDataset([anomaly_testset, normal_testset]) 
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    
    elif dataset == 'cifar100-versus-10':
        n_classes = 2
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_set = datasets.CIFAR100('./data', train=True, download=True, transform=train_transform)

        for i in range(len(train_set)):
            train_set.targets[i] = 0
        
        anomaly_testset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        for i in range(len(anomaly_testset)):
            anomaly_testset.targets[i] = 1
        normal_testset = datasets.CIFAR100('./data', train=False, download=True, transform=transform)
        for i in range(len(normal_testset)):
            normal_testset.targets[i] = 0
        test_set = torch.utils.data.ConcatDataset([anomaly_testset, normal_testset]) 
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
        
    elif dataset == 'head-ct':
        n_classes = 2
        d_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=1),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        import pandas as pd
        labels_df = pd.read_csv('./head-ct/labels.csv')
        labels_ = np.array(labels_df[' hemorrhage'].tolist())
        images = np.array(sorted(glob('./head-ct/head_ct/head_ct/*.png')))
        np.random.seed  (1225)
        indicies = np.random.permutation(100)
        train_true_idx, test_true_idx = indicies[:75]+ 100, indicies[75:]+ 100
        train_false_idx, test_false_idx = indicies[:75], indicies[75:]
        train_idx, test_idx = train_true_idx, np.concatenate((test_true_idx, test_false_idx, train_false_idx))

        train_image, train_label = images[train_idx], labels_[train_idx]
        test_image, test_label = images[test_idx], labels_[test_idx]

        print("train_image.shape, test_image.shape: ", train_image.shape, test_image.shape)
        print("train_label.shape, test_label.shape: ", train_label.shape, test_label.shape)
        
        train_set = Custome_Dataset(image_path=train_image, labels=train_label, transform=d_transform)
        test_set = Custome_Dataset(image_path=test_image, labels=test_label, transform=d_transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
   
    elif dataset == 'mvtecad':
        n_classes = 2
        train_dataset = []
        test_dataset = []
        root = "./mvtec_anomaly_detection"
        train_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((image_size[0], image_size[1])),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        
        test_transform = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
            ])
        for class_idx in labels:
            train_dataset.append(MVTecDataset(root=root, train=True, category=CLASS_NAMES[class_idx], transform=train_transform, count=-1))
            test_dataset.append(MVTecDataset(root=root, train=False, category=CLASS_NAMES[class_idx], transform=test_transform, count=-1))

        train_set = ConcatDataset(train_dataset)
        test_set = ConcatDataset(test_dataset)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
        
        print("len(test_dataset), len(train_dataset)", len(test_set), len(train_set))
    
    elif dataset == 'fashion-mnist':
        # image_size = (32, 32, 3)
        n_classes = 10
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        train_set = datasets.FashionMNIST(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.FashionMNIST(DATA_PATH, train=False, download=download, transform=test_transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    
    elif dataset == 'cifar100':
        n_classes = 100
    
        train_set = datasets.CIFAR100(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR100(DATA_PATH, train=False, download=download, transform=test_transform)
        test_set.targets = sparse2coarse(test_set.targets)
        train_set.targets = sparse2coarse(train_set.targets)

        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    elif dataset == 'mnist':
        n_classes = 10
        d_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        
        train_set = datasets.MNIST(DATA_PATH, train=True, download=download, transform=d_transform)
        test_set = datasets.MNIST(DATA_PATH, train=False, download=download, transform=d_transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    elif dataset=='cifar10-corruption':
        n_classes = 10
        transform = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
        ])
        test_set = CIFAR_CORRUCPION(transform=transform, cifar_corruption_data=P.cifar_corruption_data)
        train_set = datasets.CIFAR10(DATA_PATH, train=True, download=download, transform=transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    elif dataset=='cifar100-corruption':
        n_classes = 100
        transform = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
        ])
        test_set = CIFAR_CORRUCPION(transform=transform, cifar_corruption_label='CIFAR-100-C/labels.npy', cifar_corruption_data=P.cifar_corruption_data)
        train_set = datasets.CIFAR100(DATA_PATH, train=True, download=download, transform=transform)
        
        train_set.targets = sparse2coarse(train_set.targets)

        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    
    elif dataset=='mnist-corruption':
        n_classes = 10
        transform = transforms.Compose([
                transforms.Resize(32),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
        ])
        test_set = MNIST_CORRUPTION(root_dir=P.mnist_corruption_folder, corruption_type=P.mnist_corruption_type, transform=transform, train=False)
        train_set = datasets.MNIST(DATA_PATH, train=True, download=True, transform=transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
        
    elif dataset == 'svhn-10':
        n_classes = 10
        transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
        train_set = datasets.SVHN(DATA_PATH, split='train', download=download, transform=transform)
        test_set = datasets.SVHN(DATA_PATH, split='test', download=download, transform=transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
   
    
    elif dataset == 'svhn-10-corruption':

        def gaussian_noise(image, mean=P.noise_mean, std = P.noise_std, noise_scale = P.noise_scale):
            image = image + (torch.randn(image.size()) * std + mean)*noise_scale
            return image

        n_classes = 10
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Lambda(gaussian_noise)
        ])

        train_set = datasets.SVHN(DATA_PATH, split='train', download=download, transform=train_transform)
        test_set = datasets.SVHN(DATA_PATH, split='test', download=download, transform=test_transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    
    elif dataset == 'ISIC2018':
        n_classes = 2
        train_path = glob('./ISIC_DATASET/dataset/train/NORMAL/*')
        train_label = [0]*len(train_path)

        test_anomaly_path = glob('./ISIC_DATASET/dataset/test/ABNORMAL/*')
        test_anomaly_label = [1]*len(test_anomaly_path)
        test_normal_path = glob('./ISIC_DATASET/dataset/test/NORMAL/*')
        test_normal_label = [0]*len(test_normal_path)

        test_label = test_anomaly_label + test_normal_label
        test_path = test_anomaly_path + test_normal_path

        transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
        train_set = Custome_Dataset(image_path=train_path, labels=train_label, transform=transform)
        test_set = Custome_Dataset(image_path=test_path, labels=test_label, transform=transform)

        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
        print("len(test_set), len(train_set): ", len(test_set), len(train_set))
    elif dataset == 'cifar100-vs-x':
        n_classes = 2
        cifar_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])
        if P.outlier_dataset == 'mnist' or P.outlier_dataset == 'fashion-mnist':
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])

        train_set = datasets.CIFAR100('./data', train=True, download=True, transform=cifar_transform)
        train_set.targets = sparse2coarse(train_set.targets)
        if P.outlier_dataset == 'svhn':
            anomaly_testset = datasets.SVHN('./data', split='test', download=True, transform=transform)
            for i in range(len(anomaly_testset)):
                anomaly_testset.labels[i] = 1
        elif P.outlier_dataset == 'mnist':
            anomaly_testset = datasets.MNIST('./data', train=False, download=True, transform=transform)
            for i in range(len(anomaly_testset)):
                anomaly_testset.targets[i] = 1
        elif P.outlier_dataset == 'fashion-mnist':
            anomaly_testset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
            for i in range(len(anomaly_testset)):
                anomaly_testset.targets[i] = 1
        elif P.outlier_dataset == 'imagenet30':
            n_classes = 2
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])
            image_path = glob('./one_class_test/*/*/*')
            anomaly_testset = ImageNet30_Dataset(image_path=image_path, labels=[1]*len(image_path), transform=transform)


        normal_testset = datasets.CIFAR100('./data', train=False, download=True, transform=cifar_transform)
        for i in range(len(normal_testset)):
            normal_testset.targets[i] = 0
        test_set = torch.utils.data.ConcatDataset([anomaly_testset, normal_testset]) 
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)


    elif dataset == 'cifar10-vs-x':
        n_classes = 2
        cifar_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])
        if P.outlier_dataset == 'mnist' or P.outlier_dataset == 'fashion-mnist':
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])
        train_set = datasets.CIFAR10('./data', train=True, download=True, transform=cifar_transform)

        if P.outlier_dataset == 'svhn':
            anomaly_testset = datasets.SVHN('./data', split='test', download=True, transform=transform)
            for i in range(len(anomaly_testset)):
                anomaly_testset.labels[i] = 1
        elif P.outlier_dataset == 'mnist':
            anomaly_testset = datasets.MNIST('./data', train=False, download=True, transform=transform)
            for i in range(len(anomaly_testset)):
                anomaly_testset.targets[i] = 1
        elif P.outlier_dataset == 'fashion-mnist':
            anomaly_testset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
            for i in range(len(anomaly_testset)):
                anomaly_testset.targets[i] = 1
        elif P.outlier_dataset == 'imagenet30':
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])
            image_path = glob('./one_class_test/*/*/*')
            anomaly_testset = ImageNet30_Dataset(image_path=image_path, labels=[1]*len(image_path), transform=transform)



        normal_testset = datasets.CIFAR10('./data', train=False, download=True, transform=cifar_transform)
        for i in range(len(normal_testset)):
            normal_testset.targets[i] = 0
        
        test_set = torch.utils.data.ConcatDataset([anomaly_testset, normal_testset]) 
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
   
    elif dataset == 'imagenet':
        image_size = (224, 224, 3)
        n_classes = 30
        train_dir = os.path.join(IMAGENET_PATH, 'one_class_train')
        test_dir = os.path.join(IMAGENET_PATH, 'one_class_test')
        train_set = datasets.ImageFolder(train_dir, transform=train_transform)
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)

    else:
        raise NotImplementedError()

    return train_set, test_set, image_size, n_classes


def get_superclass_list(dataset):
    if dataset=='svhn-10' or  dataset=='svhn-10-corruption':
        return SVHN_SUPERCLASS
    elif dataset == 'cifar10-corruption':
        return CIFAR10_CORRUPTION_SUPERCLASS
    elif dataset == 'mnist-corruption':
        return MNIST_CORRUPTION_SUPERCLASS
    elif dataset == 'cifar10-versus-100':
        return CIFAR10_VER_CIFAR100_SUPERCLASS
    elif dataset == 'cifar100-versus-10':
        return CIFAR10_VER_CIFAR100_SUPERCLASS
        return ART_BENCH_SUPERCLASS
    elif dataset=='head-ct' or dataset=='cifar100-vs-x' or dataset=='cifar10-vs-x':
        return HEAD_CT_SUPERCLASS
    elif dataset == 'mvtecad':
        return MVTEC_HV_SUPERCLASS
    elif dataset == 'cifar10':
        return CIFAR10_SUPERCLASS
    elif dataset == 'fashion-mnist':
        return FashionMNIST_SUPERCLASS
    elif dataset == 'mnist':
        return MNIST_SUPERCLASS
    elif dataset == 'cifar100' or dataset=='cifar100-corruption':
        return CIFAR100_SUPERCLASS
    elif dataset == 'ISIC2018':
        return ISIC2018_SUPERCLASS
    elif dataset == 'imagenet':
        return IMAGENET_SUPERCLASS
    else:
        raise NotImplementedError()


def get_subclass_dataset(P, dataset, classes, count=-1):
    if not isinstance(classes, list):
        classes = [classes]
    indices = []
    try:        
        for idx, tgt in enumerate(dataset.targets):
            if tgt in classes:
                indices.append(idx)
    except:
        for idx, (_, tgt) in enumerate(dataset):
            if tgt in classes:
                indices.append(idx)
   
    dataset = Subset(dataset, indices)
    if count==-1:
        pass
    elif len(dataset)>count:
        unique_numbers = []
        while len(unique_numbers) < count:
            number = random.randint(0, len(dataset)-1)
            if number not in unique_numbers:
                unique_numbers.append(number)
        dataset = Subset(dataset, unique_numbers)
    else:
        num = int(count / len(dataset))
        remainding = (count - num*len(dataset))
        trnsets = [dataset for i in range(num)]
        unique_numbers = []
        while len(unique_numbers) < remainding:
            number = random.randint(0, len(dataset)-1)
            if number not in unique_numbers:
                unique_numbers.append(number)
        dataset = Subset(dataset, unique_numbers)
        trnsets = trnsets + [dataset]
        dataset = torch.utils.data.ConcatDataset(trnsets)
    return dataset

def set_dataset_count(dataset, count=-1):
    if count==-1:
        pass
    elif len(dataset)>count:
        unique_numbers = []
        while len(unique_numbers) < count:
            number = random.randint(0, len(dataset)-1)
            if number not in unique_numbers:
                unique_numbers.append(number)
        dataset = Subset(dataset, unique_numbers)
    else:
        num = int(count / len(dataset))
        remainding = (count - num*len(dataset))
        trnsets = [dataset for i in range(num)]
        unique_numbers = []
        while len(unique_numbers) < remainding:
            number = random.randint(0, len(dataset)-1)
            if number not in unique_numbers:
                unique_numbers.append(number)
        dataset = Subset(dataset, unique_numbers)
        trnsets = trnsets + [dataset]
        dataset = torch.utils.data.ConcatDataset(trnsets)

    return dataset

def get_simclr_eval_transform_imagenet(sample_num, resize_factor, resize_fix):

    resize_scale = (resize_factor, 1.0)
    if resize_fix:
        resize_scale = (resize_factor, resize_factor)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=resize_scale),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    clean_trasform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    transform = MultiDataTransformList(transform, clean_trasform, sample_num)

    return transform, transform


