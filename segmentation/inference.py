import os

import cv2
import torch

from os.path import join as path_join

from catalyst import dl, utils
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .utils.functions import transform_tensor_to_numpy


def inference_segmentation(config):
    """
    Function for inference segmentation model
    """
    assert not config.train, 'For inference set train to False'
    assert not config.dir_with_image is None and not config.dir_with_image.strip() == '', 'Needed dir with images for segmentations'
    assert not config.dir_for_save is None and not config.dir_for_save.strip() == '', 'Needed dir with images for segmentations'
    # prepare DataLoader
    dataloader = DataLoader(
        config.dataset['valid'],
        batch_size=config.batch_valid,
        num_workers=config.num_workers
    )
    loaders = {
        'train': dataloader,
        'valid': dataloader
    }
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = config.model.to(DEVICE)
    criterion = config.criterion
    optimizer = config.optimizer(model.parameters(), lr=config.LR)
    # load weights
    if config.checkpoint_path:
        checkpoint = utils.load_checkpoint(config.checkpoint_path)
        utils.unpack_checkpoint(
            checkpoint=checkpoint,
            model=config.model,
            criterion=criterion,
            optimizer=optimizer
        )
    runner = dl.SupervisedRunner()
    runner.train(
        model=config.model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        logdir=config.logdir,
        valid_loader=config.valid_loader,
        valid_metric=config.valid_metric,
        fp16=config.fp16,
        verbose=config.verbose
    )
    os.makedirs(config.dir_for_save, exist_ok=True)
    for i, data in enumerate(dataloader):
        output = F.sigmoid(runner.predict_batch(data)[6])
        images = transform_tensor_to_numpy(output)
        for j in images.shape[0]:
            img = images[j]
            name = dataloader.dataset.files[i*config.batch_valid + j]
            cv2.imwrite(
                path_join(config.dir_for_save, name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
