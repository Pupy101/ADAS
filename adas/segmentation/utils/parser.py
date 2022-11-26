from argparse import ArgumentParser
from typing import Any, Dict

from adas.utils.misc import find_enum

from ..configs import (
    EvaluationEncoderConfig,
    EvaluationSegmentationConfig,
    ModelType,
    TrainEncoderConfig,
    TrainSegmentationConfig,
)
from ..models import ModelSize


def _add_model_params(parser: ArgumentParser, is_pretraining: bool) -> ArgumentParser:
    """Add to parser model parameters"""
    parser.add_argument(
        "--model",
        choices=[_.value for _ in ModelType],
        type=str,
        default=ModelType.UNET.value,
        help="Model type (default: %(default)s)",
    )
    parser.add_argument(
        "--size",
        choices=[_.value for _ in ModelSize],
        type=str,
        default=ModelSize.MEDIUM.value,
        help="Model size (default: %(default)s)",
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=3,
        help="Count model input channels (default: %(default)s)",
    )
    if not is_pretraining:
        parser.add_argument(
            "--out_channels",
            type=int,
            default=3,
            help="Count model output channels (default: %(default)s)",
        )
    parser.add_argument(
        "--max_pool",
        action="store_true",
        default=False,
        help="Use for downsample max pooling, without \
flag convolution with stride=2 (default: %(default)s)",
    )
    if not is_pretraining:
        parser.add_argument(
            "--bilinear",
            action="store_true",
            default=False,
            help="Use for upsample bilinear, without flag transpose\
 convolution (default: %(default)s)",
        )
        parser.add_argument(
            "--count_predict_masks",
            type=int,
            help="Number of predict masks from different model's layers for training",
            required=True,
        )
    else:
        parser.add_argument(
            "--count_classes",
            "-classes",
            type=int,
            help="Count classes in classification dataset",
            required=True,
        )
        parser.add_argument(
            "--dropout",
            type=float,
            default=0.1,
            help="Dropout probability in classificator (default: %(default)s)",
        )
    return parser


def _add_data_params(
    parser: ArgumentParser, is_pretraining: bool, is_train: bool
) -> ArgumentParser:
    """Add to parser data parameters"""
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Path to directory with images",
    )
    if not is_pretraining:
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


def _add_training_params(parser: ArgumentParser, is_pretraining: bool) -> ArgumentParser:
    """Add to parser training parameters"""
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=1e-4,
        help="Optimizer learning rate (default: %(default)s)",
    )
    if not is_pretraining:
        parser.add_argument(
            "--predicts_coeffs",
            type=float,
            nargs="*",
            help="Coefficients for aggregate model predict masks in loss",
            required=True,
        )
    parser.add_argument("--seed", type=int, default=1234, help="Train seed (default: %(default)s)")
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Count training epochs (default: %(default)s)"
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


def _add_resume_params(
    parser: ArgumentParser, is_pretraining: bool, is_train: bool
) -> ArgumentParser:
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint (default: %(default)s)"
    )
    if not is_pretraining and is_train:
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
        "--cpu", action="store_true", default=False, help="Train on cpu (default: %(default)s)"
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
        "--logging",
        "-l",
        action="store_true",
        default=False,
        help="Logging in initialized wandb (default: %(default)s)",
    )
    parser.add_argument("--name_run", "-name", type=str, help="Name of experiment in wandb")
    return parser


def _parse_args(is_pretraining: bool, is_train: bool) -> Dict[str, Any]:
    """
    Create parser and composite all functions for adding parameters to parser,
        change model type and size and return parameters as dict
    """
    parser = ArgumentParser()
    parser = _add_model_params(parser, is_pretraining=is_pretraining)
    parser = _add_data_params(parser, is_pretraining=is_pretraining, is_train=is_train)
    if is_train:
        parser = _add_training_params(parser, is_pretraining=is_pretraining)
    parser = _add_resume_params(parser, is_pretraining=is_pretraining, is_train=is_train)
    parser = _add_logging_params(_add_engine_params(parser))

    args = parser.parse_args()

    args.model = find_enum(value=args.model, enum_type=ModelType)
    args.size = find_enum(value=args.size, enum_type=ModelSize)

    if args.logging and args.name_run is None:
        raise ValueError("Please set parameter --name_run for logging to wandb")

    return vars(args)


def parse_train_segmentation_args() -> TrainSegmentationConfig:
    """Parse train arguments"""

    return TrainSegmentationConfig(**_parse_args(is_pretraining=False, is_train=True))


def parse_evaluation_segmentation_args() -> EvaluationSegmentationConfig:
    """Parse evaluation arguments"""

    return EvaluationSegmentationConfig(**_parse_args(is_pretraining=False, is_train=False))


def parse_train_encoder_args() -> TrainEncoderConfig:
    """Parse pretraining arguments"""

    return TrainEncoderConfig(**_parse_args(is_pretraining=True, is_train=True))


def parse_evaluation_encoder_args() -> EvaluationEncoderConfig:
    """Parse pretraining arguments"""

    return EvaluationEncoderConfig(**_parse_args(is_pretraining=True, is_train=False))
