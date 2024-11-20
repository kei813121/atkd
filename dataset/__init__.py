from .cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample, get_cifar100_dataloaders_trainval, get_cifar100_dataloaders_val_only, get_cifar100_dataloaders_train_only, get_cifar100_dataloaders_strong
from .imagenet import get_imagenet_dataloaders, get_imagenet_dataloaders_sample, get_imagenet_dataloaders_strong


def get_dataset(cfg):
    if cfg.dataset == 'cifar100':
        if cfg.distill == 'crd':
            train_loader, val_loader, num_data = get_cifar100_dataloaders_sample(
                batch_size=cfg.batch_size,
                val_batch_size=int(cfg.batch_size/2),
                num_workers=cfg.num_workers,
                k=cfg.nce_k,
                mode=cfg.mode
            )
        else:
            train_loader, val_loader, num_data = get_cifar100_dataloaders(
                batch_size=cfg.batch_size,
                val_batch_size=int(cfg.batch_size/2),
                num_workers=cfg.num_workers
            )
        num_classes = 100
    elif cfg.dataset == "imagenet":
        if cfg.distill == 'crd':
            train_loader, val_loader, num_data = get_imagenet_dataloaders_sample(
                batch_size=cfg.batch_size,
                val_batch_size=int(cfg.batch_size/2),
                num_workers=cfg.num_workers,
                k=cfg.nce_k,
                mode=cfg.mode
            )
        else:
            train_loader, val_loader, num_data = get_imagenet_dataloaders(
                batch_size=cfg.batch_size,
                val_batch_size=int(cfg.batch_size/2),
                num_workers=cfg.num_workers
            )
        num_classes = 1000
    else:
        raise NotImplementedError(cfg.DATASET.TYPE)

    return train_loader, val_loader, num_data, num_classes


def get_dataset_strong(cfg):
    if cfg.dataset == "cifar100":
        if cfg.distill == 'crd':
            train_loader, val_loader, num_data = get_cifar100_dataloaders_sample(
                batch_size=cfg.batch_size,
                val_batch_size=int(cfg.batch_size/2),
                num_workers=cfg.num_workers,
                k=cfg.nce_k,
                mode=cfg.mode
            )
        else:
            train_loader, val_loader, num_data = get_cifar100_dataloaders_strong(
                batch_size=cfg.batch_size,
                val_batch_size=int(cfg.batch_size/2),
                num_workers=cfg.num_workers,
            )
        num_classes = 100
    elif cfg.dataset == "imagenet":
        if cfg.distill == 'crd':
            train_loader, val_loader, num_data = get_imagenet_dataloaders_sample(
                batch_size=cfg.batch_size,
                val_batch_size=int(cfg.batch_size/2),
                num_workers=cfg.num_workers,
                k=cfg.nce_k,
            )
        else:
            train_loader, val_loader, num_data = get_imagenet_dataloaders_strong(
                batch_size=cfg.batch_size,
                val_batch_size=int(cfg.batch_size/2),
                num_workers=cfg.num_workers,
            )
        num_classes = 1000
    else:
        raise NotImplementedError(cfg.dataset)

    return train_loader, val_loader, num_data, num_classes