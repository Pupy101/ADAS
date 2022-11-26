from adas.segmentation.run import run_classification
from adas.segmentation.utils.parser import parse_evaluation_encoder_args


def main():
    """Evaluation classification model"""
    config = parse_evaluation_encoder_args()
    run_classification(config)


if __name__ == "__main__":
    main()
