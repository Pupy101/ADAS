from adas.segmentation.run import run_classification
from adas.segmentation.utils.parser import parse_train_encoder_args


def main():
    """Train classification model"""
    config = parse_train_encoder_args()
    run_classification(config)


if __name__ == "__main__":
    main()
