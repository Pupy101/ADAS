import argparse

from config_segmentation import config as segmentation_config 
from segmentation import train_segmentation, inference_segmentation

if __name__ == '__main__':
    if segmentation_config.need_segmentation:
        if segmentation_config.train:
            train_segmentation(segmentation_config)
        else:
            inference_segmentation(segmentation_config)
 