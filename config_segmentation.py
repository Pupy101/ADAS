from typing import List, Dict, Union

import segmentation_models_pytorch as smp

from torch import nn, optim, Tensor, tensor
from catalyst import dl
from catalyst.contrib import nn as catalyst_nn

from segmentation import U2Net, BDD100K, InferenceDataset, train_transform, valid_transforms


class SegmentationLossU2Net(nn.Module):

    def __init__(
            self,
            base_loss: nn.Module = catalyst_nn.DiceLoss(),
            n_outputs: int = 7,
            weights: List[float] = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    ):
        super().__init__()
        assert len(weights) == n_outputs, 'Length of weight must equal number of outputs'
        self.loss = base_loss
        self._n_outputs = n_outputs
        self.weights = weights

    def forward(self, prediction: List[Tensor], target: Tensor):
        overall_loss = sum(
            self.loss(prediction[i], target) * self.weights[i]
            for i in range(self._n_outputs)
        )
        return overall_loss


class SegmentationMultipleLoss(nn.Module):

    def __init__(
            self,
            losses: List[nn.Module] = [catalyst_nn.DiceLoss()],
            weights: List[float] = [1]
    ):
        super().__init__()
        assert len(losses) == len(weights), 'Count losses must equal count weights'
        self.losses = losses
        self.weights = weights

    def forward(self, prediction, target):
        overall_loss = sum(
            self.losses[i](prediction, target) * self.weights[i]
            for i in range(len(self.weights))
        )
        return overall_loss


class Config:
    """
    Config of using Segmentation net
    """
    # Type of using segmentation net
    TYPE_SEGMENTATION_USING = 'train'  # or 'eval' or None if you don't want use net
    # Urls for checkpoint and datasets
    DIR_FOR_SAVE_CHECKPOINT = './checkpoints'
    URL_CHECKPOINT: Union[None, Dict[str, Union[str]]] = None  # {'name': 'checkpoint.pth', 'url': 'url/to/checkpoint'}

    DIR_FOR_SAVE_DATASETS = './datasets'
    if TYPE_SEGMENTATION_USING in ['train', 'eval']:
        URL_DATASETS: Union[None, List[Dict[str, str]]] = [
            {
                'name': 'bdd100k.zip',
                'url': 'https://drive.google.com/uc?id=1pc9KR0mGJtgsZDpSmlcIa-Zau_Qk3yQk'
            },
            {
                'name': 'bdd100k_drivable_maps.zip',
                'url': 'https://drive.google.com/uc?id=1lgSGH_ifIzTSn9aeeQ1f0wYA42ctR9hB'
            }
        ]

    # Paths for creating dataset and checkpoint
    CHECKPOINT_PATH: Union[str, None] = None  # or path to model checkpoint

    # Path for dataset
    if TYPE_SEGMENTATION_USING in 'train':
        DIR_PATH_DATASET: Dict[str, Dict[str, str]] = {
            'train': {
                'image': './datasets/bdd100k/images/100k/train',
                'mask': './datasets/bdd100k/drivable_maps/color_labels/train'
            },
            'valid': {
                'image': './datasets/bdd100k/images/100k/val',
                'mask': './datasets/bdd100k/drivable_maps/color_labels/val'
            }
        }
    elif TYPE_SEGMENTATION_USING == 'eval':
        DIR_PATH_DATASET: str = '/content/bdd100k/images/100k/val'
        DIR_PATH_SAVE_RESULT = './result'

    LOADERS_PARAMS: Dict[str, Dict[str, Union[str, bool, float]]] = {
        'train': {'batch_size': 8, 'num_workers': 2, 'shuffle': True},
        'valid': {'batch_size': 8, 'num_workers': 2, 'shuffle': False}
    }

    # Model
    IS_USE_U2NET = True
    MODEL_PARAMS: Dict[str, Union[float, str]] = {
        'in_channels': 3, 'out_channels': 3, 'is_big_net': False
    }

    # Train parameters
    TRAIN_PARAMS: Dict[str, Union[bool, str, float]] = {
        'num_epochs': 10, 'callbacks': [
            dl.IOUCallback(input_key='overall_logits', target_key='targets'),
            dl.DiceCallback(input_key='overall_logits', target_key='targets')
        ],
        'seed': 1234, 'logdir': './training_results', 'valid_loader': 'valid',
        'valid_metric': 'loss', 'fp16': None, 'verbose': True
    }

    # Optimizer
    OPTIMIZER_PARAMS: Dict[str, Union[str, float]] = {
        'lr': 1e-4
    }
    OPTIMIZER = optim.AdamW

    # Scheduler
    SCHEDULER_PARAMS: Dict[str, Union[str, float]] = {
        'max_lr': OPTIMIZER_PARAMS['lr'], 'epochs': TRAIN_PARAMS['num_epochs']
    }
    SCHEDULER = optim.lr_scheduler.OneCycleLR

    # Loss function and model
    if IS_USE_U2NET:
        MODEL = U2Net(**MODEL_PARAMS)
        CRITERION = SegmentationLossU2Net()
    else:
        MODEL = smp.Unet(**MODEL_PARAMS)
        CRITERION = SegmentationMultipleLoss()


    # Creating datasets
    if TYPE_SEGMENTATION_USING == 'train':
        DATASET = {
            'train': BDD100K(
                DIR_PATH_DATASET['train']['image'], DIR_PATH_DATASET['train']['mask'],
                check=True, transforms=train_transform
            ),
            'valid': BDD100K(
                DIR_PATH_DATASET['valid']['image'], DIR_PATH_DATASET['valid']['mask'],
                check=True, transforms=valid_transforms
            )
        }
    elif TYPE_SEGMENTATION_USING == 'eval':
        DATASET = {
            'valid': InferenceDataset(DIR_PATH_DATASET, transforms=valid_transforms)
        }
