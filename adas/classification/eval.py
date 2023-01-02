from adas.classification.run import run
from adas.classification.utils.parser import parse_eval_args


def main():
    """Evaluation classification model"""
    config = parse_eval_args()
    run(config)


if __name__ == "__main__":
    main()
