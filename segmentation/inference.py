import os

import cv2
import torch

from os.path import join as path_join

from catalyst import dl, utils
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .utils.functions import transform_tensor_to_numpy
from .train import RunnerU2Net


def inference_segmentation(config):
    """
    Function for inference segmentation model
    """
    # prepare DataLoader
    dataloader = DataLoader(
        config.DATASET['valid'],
        **config.LOADERS_PARAMS['valid']
    )
    loaders = {
        'train': dataloader,
    }
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = config.MODEL.to(DEVICE)
    criterion = config.CRITERION
    optimizer = config.OPTIMIZER(model.parameters(), lr=config.LR)
    config.TRAIN_PARAMS.update(
        {
            'loaders': loaders, 'model': model, 'criterion': criterion, 'optimizer': optimizer,
        }
    )
    # load weights
    if config.CHECKPOINT_PATH:
        checkpoint = utils.load_checkpoint(config.CHECKPOINT_PATH)
        utils.unpack_checkpoint(
            checkpoint=checkpoint,
            model=config.model,
            criterion=criterion,
            optimizer=optimizer
        )
    if config.IS_USE_U2NET:
        runner = RunnerU2Net()
    else:
        runner = dl.SupervisedRunner()
    runner.train(**config.TRAIN_PARAMS)
    os.makedirs(config.DIR_PATH_SAVE_RESULT, exist_ok=True)
    for i, data in enumerate(dataloader):
        if config.IS_USE_U2NET:
            output = runner.predict_batch(data)[-1]
        else:
            output = runner.predict_batch(data)
        images = transform_tensor_to_numpy(output)
        for j in images.shape[0]:
            img = images[j]
            name = dataloader.dataset.files[i*config.batch_valid + j]
            cv2.imwrite(
                path_join(config.dir_for_save, name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
