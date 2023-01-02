from argparse import ArgumentParser, Namespace
from typing import Any, Dict

from adas.core.models.types import DownsampleMode
from adas.core.utils.misc import find_enum
from adas.core.utils.parser import add_engine_params, add_hyper_params, add_logging_params

from ..config import EvalCfg, TrainCfg
from ..models.types import ModelSize, ModelType


def add_model_params(parser: ArgumentParser) -> ArgumentParser:
    """Add to parser model parameters"""

    parser.add_argument(
        "--model",
        choices=[_.value for _ in ModelType],
        type=str,
        default=ModelType.UNET_ENCODER.value,
        help="Model type (default: %(default)s)",
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=3,
        help="Count model input channels (default: %(default)s)",
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
        "--classes",
        type=int,
        default=1000,
        help="Count of predictable classes (default: %(default)s)",
    )
    return parser


def add_data_params(parser: ArgumentParser, is_train: bool) -> ArgumentParser:
    """Add to parser data parameters"""

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to directory with images",
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


def add_weight_params(parser: ArgumentParser) -> ArgumentParser:
    """Add to parser weights parameters"""

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint (default: %(default)s)",
    )
    return parser


def create_parser(is_train: bool) -> ArgumentParser:
    """Create argument parser"""

    parser = ArgumentParser()
    parser = add_model_params(parser)
    parser = add_data_params(parser, is_train=is_train)
    if is_train:
        parser = add_hyper_params(parser)
    parser = add_weight_params(parser)
    parser = add_engine_params(parser)
    parser = add_logging_params(parser)
    return parser


def parse_args(args: Namespace) -> Dict[str, Any]:
    """
    Create parser and composite all functions for adding parameters to parser,
        change model type and size and return parameters as dict
    """

    args.model = find_enum(value=args.model, enum_type=ModelType)
    args.size = find_enum(value=args.size, enum_type=ModelSize)
    args.downsample = find_enum(value=args.downsample, enum_type=DownsampleMode)
    return vars(args)


def parse_train_args() -> TrainCfg:
    """Parse train arguments"""

    parser = create_parser(is_train=True)
    args = parser.parse_args()
    return TrainCfg(**parse_args(args))


def parse_eval_args() -> EvalCfg:
    """Parse evaluation arguments"""

    parser = create_parser(is_train=False)
    args = parser.parse_args()
    return EvalCfg(**parse_args(args))
