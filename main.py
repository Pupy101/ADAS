import argparse

from config_segmentation import Config as segmentation_config
from segmentation import train_segmentation, inference_segmentation

if __name__ == '__main__':
    if segmentation_config.TYPE_SEGMENTATION_USING in ['train', 'eval']:
        if segmentation_config.TYPE_SEGMENTATION_USING == 'train':
            train_segmentation(segmentation_config)
        elif segmentation_config.TYPE_SEGMENTATION_USING == 'eval':
            inference_segmentation(segmentation_config)
 