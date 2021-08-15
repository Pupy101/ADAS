import torch

from typing import Mapping, Any

from torch import nn
from torch.utils.data import DataLoader, Dataset
from catalyst import dl, utils


class SegmentationRunner(dl.SupervisedRunner):
    def handle_batch(self, batch: Mapping[str, Any]):
        all_logits = self.forward(batch)['logits']
        self.batch = {
            **batch,
            'logits': all_logits,
            'overall_logits': nn.Sigmoid()(all_logits[6])
        }


def train_segmentation(config):
    """
    Function training segmentation model
    ------------------------------
    Input parametrs:
        config  - python class with parameters for training
    """
    loaders = {
        'train': DataLoader(
            config.dataset['train'],
            batch_size=config.batch_train,
            num_workers=config.num_workers,
            shuffle=True
        ),
        'valid': DataLoader(
            config.dataset['valid'],
            batch_size=config.batch_valid,
            num_workers=config.num_workers
        )
    }
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = config.model.to(DEVICE)

    criterion = config.criterion
    optimizer = config.optimizer(model.parameters(), lr=config.LR)
    scheduler = config.scheduler(
        optimizer,
        **config.sheduler_params,
        steps_per_epoch=len(loaders['train']) + 5
    )
    if config.checkpoint_path:
        checkpoint = utils.load_checkpoint(config.checkpoint_path)
        utils.unpack_checkpoint(
            checkpoint=checkpoint,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler
        )
    runner = SegmentationRunner()
    runner.train(
        num_epochs=config.n_epochs,
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        callbacks=config.callbacks,
        seed=config.seed,
        logdir=config.logdir,
        valid_loader=config.valid_loader,
        valid_metric=config.valid_metric,
        fp16=config.fp16,
        verbose=config.verbose
    )
