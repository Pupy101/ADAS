from adas.segmentation.run import run_segmentation
from adas.segmentation.utils.parser import parse_train_segmentation_args


def main():
    """Train segmentation model"""
    config = parse_train_segmentation_args()
    run_segmentation(config)


if __name__ == "__main__":
    main()
