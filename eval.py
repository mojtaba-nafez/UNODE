import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy

from args import parse_args
import models.classifier as C
from datasets import set_dataset_count, get_dataset, get_superclass_list, get_subclass_dataset
from utils.utils import get_loader_unique_label
from evals.evaluation import eval_ood_detection

def initialize():
    P = parse_args()
    cls_list = get_superclass_list(P.dataset)
    anomaly_labels = [elem for elem in cls_list if elem not in [P.normal_class]]
    
    P.n_gpus = torch.cuda.device_count()
    assert P.n_gpus <= 1  # no multi GPU
    P.multi_gpu = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return P, anomaly_labels, device

def prepare_datasets(P):
    train_set, test_set, image_size, n_classes = get_dataset(
        P, dataset=P.dataset, eval=True, download=True, image_size=(P.image_size, P.image_size, 3), labels=[P.normal_class]
    )
    P.image_size = image_size
    P.n_classes = n_classes

    full_test_set = deepcopy(test_set)
    if P.dataset in ['cifar10-vs-x', 'cifar100-vs-x', 'ISIC2018', 'mvtecad', 'cifar10-versus-100', 'cifar100-versus-10']:
        train_set = set_dataset_count(train_set, count=P.normal_data_count)
        test_set = get_subclass_dataset(P, test_set, classes=[0])
    else:
        train_set = get_subclass_dataset(P, train_set, classes=[P.normal_class], count=P.normal_data_count)
        test_set = get_subclass_dataset(P, test_set, classes=[P.normal_class])

    return train_set, test_set, full_test_set

def prepare_dataloaders(train_set, test_set, P):
    kwargs = {'pin_memory': False, 'num_workers': 4}
    train_loader = DataLoader(train_set, shuffle=True, batch_size=P.batch_size, **kwargs)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)
    return train_loader, test_loader

def prepare_ood_loaders(P, full_test_set, kwargs):
    P.ood_dataset = P.ood_dataset if P.dataset in ['cifar10-vs-x', 'cifar100-vs-x', 'ISIC2018', 'mvtecad', 'cifar10-versus-100', 'cifar100-versus-10'] else [1]
    print("P.ood_dataset", P.ood_dataset)
    ood_test_loader = {}
    for ood in P.ood_dataset:
        ood_test_set = get_subclass_dataset(P, full_test_set, classes=ood)
        ood_label = f'one_class_{ood}'
        print(f"testset anomaly(class {ood_label}):", len(ood_test_set))
        ood_test_loader[ood_label] = DataLoader(ood_test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)
        print("Unique labels(ood_test_loader):", get_loader_unique_label(ood_test_loader[ood_label]))
    return ood_test_loader

def initialize_model(P, device):
    simclr_aug = C.get_simclr_augmentation(P, image_size=P.image_size).to(device)
    model = C.get_classifier(P.model, n_classes=P.n_classes, activation=P.activation_function).to(device)
    model = C.get_shift_classifer(model, 2).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    if P.load_path is not None:
        checkpoint = torch.load(P.load_path)
        model.load_state_dict(checkpoint, strict=not P.no_strict)

    model.eval()
    return model, simclr_aug, criterion

def main():
    P, anomaly_labels, device = initialize()
    P.ood_dataset = anomaly_labels
    train_set, test_set, full_test_set = prepare_datasets(P)
    train_loader, test_loader = prepare_dataloaders(train_set, test_set, P)
    
    print("len train_set", len(train_set))
    print("len test_set", len(test_set))
    print("Unique labels(test_loader):", get_loader_unique_label(test_loader))
    print("Unique labels(train_loader):", get_loader_unique_label(train_loader))
    
    kwargs = {'pin_memory': False, 'num_workers': 4}
    ood_test_loader = prepare_ood_loaders(P, full_test_set, kwargs)

    model, simclr_aug, criterion = initialize_model(P, device)
    
    print(P)

    with torch.no_grad():
        auroc_dict = eval_ood_detection(P, model, test_loader, ood_test_loader,
                                        train_loader=train_loader, simclr_aug=simclr_aug)
    
    if P.normal_class is not None:
        mean = sum(auroc_dict.values()) / len(auroc_dict)
        auroc_dict['one_class_mean'] = mean

    for ood, auroc in auroc_dict.items():
        message = f"[{ood} {auroc:.4f}]"
        if P.print_score:
            print(message)

if __name__ == '__main__':
    main()
