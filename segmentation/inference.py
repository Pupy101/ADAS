import os

import cv2
import torch

from os.path import join as path_join

from tqdm import tqdm
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
    optimizer = config.OPTIMIZER(model.parameters(), **config.OPTIMIZER_PARAMS)
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
    os.makedirs(config.DIR_PATH_SAVE_RESULT, exist_ok=True)
    for data in tqdm(dataloader):
        batch, naming = data['features'].to(DEVICE), data['name'][0]
        if config.IS_USE_U2NET:
            output = model(batch)[-1]
        else:
            output = model(batch)
        images = transform_tensor_to_numpy(output)
        for j in range(images.shape[0]):
            img, name = images[j], naming[j].rsplit('/', maxsplit=1)[1]
            cv2.imwrite(path_join(config.DIR_PATH_SAVE_RESULT, name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
