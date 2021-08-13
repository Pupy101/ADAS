import os
import zipfile

import gdown

from config_segmentation import config as segmentation_config

if segmentation_config.need_segmentation:
    if segmentation_config.url_datasets:
        cur_dir = os.getcwd()
        os.makedirs(segmentation_config.dir_for_save_datasets, exist_ok=True)
        os.chdir(segmentation_config.dir_for_save_datasets)
        for data in segmentation_config.url_datasets:
            gdown.download(
                url=data['url'],
                output=data['name']
            )
            with zipfile.ZipFile(data['name']) as zip_ref:
                zip_ref.extractall('./')
        os.chdir(cur_dir)
    if segmentation_config.url_checkpoint:
        cur_dir = os.getcwd()
        os.makedirs(segmentation_config.dir_for_save_checkpoint, exist_ok=True)
        os.chdir(segmentation_config.dir_for_save_checkpoint)
        for data in segmentation_config.url_datasets:
            gdown.download(
                url=data['url'],
                output=data['name']
            )    
        os.chdir(cur_dir)
            