from adas.segmentation.run import run_segmentation
from adas.segmentation.utils.parser import parse_evaluation_segmentation_args


def main():
    """Evaluation segmentation model"""
    config = parse_evaluation_segmentation_args()
    run_segmentation(config)


if __name__ == "__main__":
    main()
