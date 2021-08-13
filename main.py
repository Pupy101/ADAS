import argparse

from config_segmentation import config
from segmentation import train_segmentation, inference_segmentation

if __name__ == '__main__':
    if config.train:
        train_segmentation(config)
    else:
        inference_segmentation(config)
 