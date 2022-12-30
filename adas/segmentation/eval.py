from adas.segmentation.run import run
from adas.segmentation.utils.parser import parse_eval_args


def main():
    """Evaluation segmentation model"""
    config = parse_eval_args()
    run(config)


if __name__ == "__main__":
    main()
