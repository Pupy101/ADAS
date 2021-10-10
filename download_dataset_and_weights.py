import os
import zipfile

import gdown

from config_segmentation import Config as segmentation_config

if segmentation_config.TYPE_SEGMENTATION_USING in ['train', 'eval']:
    if segmentation_config.URL_DATASETS:
        cur_dir = os.getcwd()
        os.makedirs(segmentation_config.DIR_FOR_SAVE_DATASETS, exist_ok=True)
        os.chdir(segmentation_config.DIR_FOR_SAVE_DATASETS)
        for data in segmentation_config.URL_DATASETS:
            gdown.download(
                url=data['url'],
                output=data['name']
            )
            with zipfile.ZipFile(data['name']) as zip_ref:
                zip_ref.extractall('./')
        os.chdir(cur_dir)
    if segmentation_config.URL_CHECKPOINT:
        cur_dir = os.getcwd()
        os.makedirs(segmentation_config.DIR_FOR_SAVE_CHECKPOINT, exist_ok=True)
        os.chdir(segmentation_config.DIR_FOR_SAVE_CHECKPOINT)
        gdown.download(
            url=segmentation_config.URL_CHECKPOINT['url'],
            output=segmentation_config.URL_CHECKPOINT['name']
        )
        os.chdir(cur_dir)
            