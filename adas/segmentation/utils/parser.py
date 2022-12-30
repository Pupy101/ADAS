from argparse import ArgumentParser, Namespace
from typing import Any, Dict

from adas.core.models.types import DownsampleMode, UpsampleMode
from adas.utils.misc import find_enum

from ..config import EvalCfg, TrainCfg
from ..models.types import ModelSize, ModelType


def _add_model_params(parser: ArgumentParser) -> ArgumentParser:
    """Add to parser model parameters"""
    parser.add_argument(
        "--model",
        choices=[_.value for _ in ModelType],
        type=str,
        default=ModelType.UNET.value,
        help="Model type (default: %(default)s)",
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=3,
        help="Count model input channels (default: %(default)s)",
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=3,
        help="Count model output channels (default: %(default)s)",
    )
    parser.add_argument(
        "--size",
        choices=[_.value for _ in ModelSize],
        type=str,
        default=ModelSize.SMALL.value,
        help="Model size (default: %(default)s)",
    )
    parser.add_argument(
        "--downsample",
        choices=[_.value for _ in DownsampleMode],
        type=str,
        default=DownsampleMode.MAX_POOL.value,
        help="Downsample mode (default: %(default)s)",
    )
    parser.add_argument(
        "--upsample",
        choices=[_.value for _ in UpsampleMode],
        type=str,
        default=UpsampleMode.BILINEAR.value,
        help="Upsample mode (default: %(default)s)",
    )
    parser.add_argument(
        "--count_features",
        type=int,
        help="Number of predict features maps from model",
        required=True,
    )
    return parser


def _add_data_params(parser: ArgumentParser, is_train: bool) -> ArgumentParser:
    """Add to parser data parameters"""
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Path to directory with images",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        required=True,
        help="Path to directory with masks",
    )
    if is_train:
        parser.add_argument(
            "--train_batch_size",
            type=int,
            default=40,
            help="Train loader batch size (default: %(default)s)",
        )
    parser.add_argument(
        "--valid_batch_size",
        type=int,
        default=120,
        help="Validation loader batch size (default: %(default)s)",
    )
    if is_train:
        parser.add_argument(
            "--valid_size",
            type=float,
            default=0.3,
            help="Size of valid dataset (default: %(default)s)",
        )
    return parser


def _add_training_params(parser: ArgumentParser, is_train: bool) -> ArgumentParser:
    """Add to parser training parameters"""
    if is_train:
        parser.add_argument(
            "--learning_rate",
            "-lr",
            type=float,
            default=1e-4,
            help="Optimizer learning rate (default: %(default)s)",
        )
    parser.add_argument(
        "--predicts_coeffs",
        type=float,
        nargs="*",
        help="Coefficients for aggregate model predict masks in loss",
        required=True,
    )
    if is_train:
        parser.add_argument(
            "--seed", type=int, default=1234, help="Train seed (default: %(default)s)"
        )
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


def _add_weight_params(parser: ArgumentParser, is_train: bool) -> ArgumentParser:
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint (default: %(default)s)",
    )
    if is_train:
        parser.add_argument(
            "--resume_encoder",
            type=str,
            default=None,
            help="Resume from checkpoint segmentation encoder (default: %(default)s)",
        )
    return parser


def _add_engine_params(parser: ArgumentParser) -> ArgumentParser:
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


def _add_logging_params(parser: ArgumentParser) -> ArgumentParser:
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


def _parse_args(args: Namespace) -> Dict[str, Any]:
    """
    Create parser and composite all functions for adding parameters to parser,
        change model type and size and return parameters as dict
    """

    args.model = find_enum(value=args.model, enum_type=ModelType)
    args.size = find_enum(value=args.size, enum_type=ModelSize)
    args.downsample = find_enum(value=args.downsample, enum_type=DownsampleMode)
    args.upsample = find_enum(value=args.upsample, enum_type=UpsampleMode)
    return vars(args)


def create_parser(is_train: bool) -> ArgumentParser:
    parser = ArgumentParser()
    parser = _add_model_params(parser)
    parser = _add_data_params(parser, is_train=is_train)
    parser = _add_training_params(parser, is_train=is_train)
    parser = _add_weight_params(parser, is_train=is_train)
    parser = _add_engine_params(parser)
    parser = _add_logging_params(parser)
    return parser


def parse_train_args() -> TrainCfg:
    """Parse train arguments"""

    parser = create_parser(is_train=True)
    args = parser.parse_args()
    return TrainCfg(**_parse_args(args))


def parse_eval_args() -> EvalCfg:
    """Parse evaluation arguments"""

    parser = create_parser(is_train=False)
    args = parser.parse_args()
    return EvalCfg(**_parse_args(args))
