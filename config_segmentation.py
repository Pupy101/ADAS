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

    checkpoint = None # or path to model
    model = U2Net(3, 3)
    # if True - dataset for train if False model in inference mode
    train = True
    dir_for_train = {
        'train': {
            'image': '/content/bdd100k/images/100k/train',
            'mask': '/content/bdd100k/drivable_maps/color_labels/train'
        },
        'valid': {
            'image': '/content/bdd100k/images/100k/val',
            'mask': '/content/bdd100k/drivable_maps/color_labels/val'
        }
    }
    dir_for_inference = '/content/bdd100k/images/100k/val'
    if train:
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
