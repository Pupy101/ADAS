from typing import Union, List

from torch import nn, optim, Tensor
from catalyst import dl
from catalyst.contrib import nn as catalyst_nn

from segmentation import U2Net, BDD100K, InferenceDataset, train_transform, valid_transforms


class SegmentationLoss(nn.Module):
    def __init__(
        self,
        loss1: nn.Module = catalyst_nn.DiceLoss(),
        loss2: nn.Module = catalyst_nn.TrevskyLoss(alpha=0.7),
        koeff: Union[List, Tensor] = [0.45, 0.55]
    ):
        super().__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        assert isinstance(koeff, (list, tuple)) and len(koeff) == 2
        self.koeff = koeff

    def forward(self, inputs, targets):
        return self.koeff[0]*self.loss1(inputs, targets) + self.koeff[1]*self.loss2(inputs, targets)


class config:
    # This attribute for activate train or evaluation segmentation model
    need_segmentation = True
    # url and name to checpoint or None (example for url is url_datasets). It been save in dir ./checkpoint
    url_checkpoint = None
    dir_for_save_checkpoint = './checkpoints'
    # path to checpoint or None   | need saved *_full.pth weights from catalyst
    checkpoint_path = None # or path to model
    # Model
    in_channels = 3
    out_channels = 3
    model = U2Net(
        in_channels=in_channels,
        out_channels=out_channels
        )
    # Type of using model train = True - training | False - inference
    train = True
    # dir for datasets
    dir_for_train = {
        'train': {
            'image': './datasets/bdd100k/images/100k/train',
            'mask': './datasets/bdd100k/drivable_maps/color_labels/train'
        },
        'valid': {
            'image': './datasets/bdd100k/images/100k/val',
            'mask': './datasets/bdd100k/drivable_maps/color_labels/val'
        }
    }
    dir_for_inference = '/content/bdd100k/images/100k/val'
    # creating datasets
    if train:
        # url with name and url for linux gdown or None
        url_datasets = [
            {
                'name': 'bdd100k.zip',
                'url': 'https://drive.google.com/uc?id=1pc9KR0mGJtgsZDpSmlcIa-Zau_Qk3yQk'
            },
            {
                'name': 'bdd100k_drivable_maps.zip',
                'url': 'https://drive.google.com/uc?id=1lgSGH_ifIzTSn9aeeQ1f0wYA42ctR9hB'
            }
        ]
        dir_for_save_datasets = './datasets'
        # training datasets
        dataset = {
            'train': BDD100K(
                dir_for_train['train']['image'],
                dir_for_train['train']['mask'],
                check=True,
                transforms=train_transform
            ),
            'valid': BDD100K(
                dir_for_train['valid']['image'],
                dir_for_train['valid']['mask'],
                check=True,
                transforms=valid_transforms
            )
        }
    else:
        # Same is url_datasets 
        url_datasets = None
        # inference dataset and directory for save result
        dataset = {
            'valid': InferenceDataset(dir_for_inference, transforms=valid_transforms)
        }
        dir_for_save = './targets'

    # Dataloader parameters
    batch_train = 16
    batch_valid = 64
    num_workers = 2

    # Training parameters
    n_epochs = 10
    criterion = SegmentationLoss()
    LR = 3e-4
    optimizer = optim.AdamW
    sheduler_params = {
        'max_lr': LR,
        'epochs': n_epochs
    }
    scheduler = optim.lr_scheduler.OneCycleLR

    # Parameters catalyst runner
    seed = 1234
    logdir = './training'
    valid_loader = 'valid'
    valid_metric = 'loss'
    callbacks = [
        dl.IOUCallback(input_key="logits", target_key="targets"),
        dl.DiceCallback(input_key="logits", target_key="targets"),
        dl.TrevskyCallback(input_key="logits", target_key="targets", alpha=0.7),
    ]
    fp16 = True
    verbose = True
