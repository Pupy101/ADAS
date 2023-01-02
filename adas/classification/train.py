from adas.classification.run import run
from adas.classification.utils.parser import parse_train_args


def main():
    """Train classification model"""
    config = parse_train_args()
    run(config)


if __name__ == "__main__":
    main()
