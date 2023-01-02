from argparse import ArgumentParser


def add_hyper_params(parser: ArgumentParser) -> ArgumentParser:
    """Add to parser training parameters"""
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=1e-4,
        help="Optimizer learning rate (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Train seed (default: %(default)s)")
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Count training epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--num_batch_steps",
        type=int,
        default=None,
        help="Run only n batches from loaders (default: %(default)s)",
    )
    parser.add_argument(
        "--profile",
        "-p",
        action="store_true",
        default=False,
        help="Profile first validation batch (default: %(default)s)",
    )
    return parser


def add_logging_params(parser: ArgumentParser) -> ArgumentParser:
    """Add to parser logging parameters"""
    parser.add_argument(
        "--logdir",
        type=str,
        default="./logs",
        help="Directory for logs and checkpoints (default: %(default)s)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Verbose training (default: %(default)s)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Logging in initialized wandb (default: %(default)s)",
    )
    parser.add_argument("--name_run", "-name", type=str, help="Name of experiment in wandb")
    return parser


def add_engine_params(parser: ArgumentParser) -> ArgumentParser:
    """Add to parser engine parameters"""
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="Train on cpu (default: %(default)s)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Train in half precision (default: %(default)s)",
    )
    return parser
