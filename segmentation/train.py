import torch

from typing import Mapping, Any

from torch import nn
from torch.utils.data import DataLoader, Dataset
from catalyst import dl, utils


class RunnerU2Net(dl.SupervisedRunner):
    def handle_batch(self, batch: Mapping[str, Any]):
        all_logits = self.forward(batch)['logits']
        self.batch = {
            **batch,
            'logits': all_logits,
            'overall_logits': all_logits[-1]
        }


def train_segmentation(config) -> None:
    """
    Function training segmentation model
    ------------------------------
    Input parametrs:
        config  - python class with parameters for training
    """
    loaders = {
        'train': DataLoader(
            config.DATASET['train'],
            batch_size=config.batch_train,
            num_workers=config.num_workers,
            shuffle=True
        ),
        'valid': DataLoader(
            config.DATASET['valid'],
            batch_size=config.batch_valid,
            num_workers=config.num_workers
        )
    }
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = config.MODEL.to(DEVICE)
    criterion = config.CRITERION
    optimizer = config.OPTIMIZER(model.parameters(), **config.OPTIMIZER_PARAMS)
    if config.SCHEDULER is not None:
        scheduler = config.SCHEDULER(
            optimizer,
            **config.SCHEDULER_PARAMS,
            steps_per_epoch=len(loaders['train']) + 5
        )
    else:
        scheduler = None
    if config.checkpoint_path:
        checkpoint = utils.load_checkpoint(config.checkpoint_path)
        utils.unpack_checkpoint(
            checkpoint=checkpoint,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler
        )
    config.TRAIN_PARAMS.update(
        {
            'loaders': loaders, 'model': model, 'criterion': criterion, 'optimizer': optimizer,
            'scheduler': scheduler
        }
    )
    if config.IS_USE_U2NET:
        runner = RunnerU2Net()
    else:
        runner = dl.SupervisedRunner()
    runner.train(**config.TRAIN_PARAMS)
