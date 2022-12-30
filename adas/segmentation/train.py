from adas.segmentation.run import run
from adas.segmentation.utils.parser import parse_train_args


def main():
    """Train segmentation model"""
    config = parse_train_args()
    run(config)


if __name__ == "__main__":
    main()
