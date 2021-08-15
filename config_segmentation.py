from typing import List

from torch import nn, optim, Tensor, tensor
from catalyst import dl
from catalyst.contrib import nn as catalyst_nn

from segmentation import U2Net, BDD100K, InferenceDataset, train_transform, valid_transforms


class SegmentationLoss(nn.Module):
    def __init__(
        self,
        base_loss: nn.Module = catalyst_nn.DiceLoss(),
        n_outputs: int = 7
    ):
        super().__init__()
        self.loss = base_loss
        self._n_outputs = n_outputs

    def forward(self, inputs: List[Tensor], targets: Tensor):
        overall_loss = self.loss(inputs[0], targets)
        for i in range(1, self._n_outputs):
            overall_loss = self.loss(inputs[i], targets)
        return overall_loss


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
    is_big_net = False
    model = U2Net(
        in_channels=in_channels,
        out_channels=out_channels,
        is_big_net=is_big_net
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
    batch_train = 64
    batch_valid = 128
    num_workers = 2

    # Training parameters
    n_epochs = 10
    criterion = SegmentationLoss(
        base_loss=catalyst_nn.TrevskyLoss(
            alpha=0.7,
            weights=tensor([1, 20, 1])
        )
    )
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
        dl.IOUCallback(input_key='overall_logits', target_key='targets'),
        dl.DiceCallback(input_key='overall_logits', target_key='targets')
    ]
    fp16 = True
    verbose = True
