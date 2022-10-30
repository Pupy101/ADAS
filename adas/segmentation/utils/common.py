from argparse import ArgumentParser
from typing import Any, Dict, List, Mapping, Optional, Union

from catalyst import dl
from catalyst.core.callback import Callback
from catalyst.core.engine import Engine
from catalyst.core.runner import RunnerModel
from catalyst.loggers.wandb import WandbLogger

from adas.segmentation.models import U2net, Unet
from adas.segmentation.utils.configs import (
    CLASS_NAMES,
    INPUTS_KEYS,
    PREFIXES,
    TARGETS_KEYS,
    EvaluationConfig,
    InferenceConfig,
    ModelType,
    TrainConfig,
)


class MultipleOutputModelRunner(dl.SupervisedRunner):
    """Multi output model catalyst runner."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model: RunnerModel = None,
        engine: Engine = None,
        input_key: Any = "features",
        output_key: Any = "probas",
        target_key: str = "targets",
        loss_key: str = "loss",
    ):
        super().__init__(model, engine, input_key, output_key, target_key, loss_key)

    def handle_batch(self, batch: Mapping[str, Any]):
        probas = self.forward(batch)["probas"]
        self.batch = {
            **batch,
            "probas": probas,
            "last_probas": probas[-2],
            "agg_probas": probas[-1],
        }


def create_callbacks(  # pylint: disable=too-many-arguments
    logdir: str,
    metric_key: str = "loss",
    resume: Optional[str] = None,
    compute_iou: bool = True,
    compute_dice: bool = True,
    profiling: bool = True,
    num_batch_steps: Optional[int] = None,
) -> List[Callback]:
    """Create callback for catalyst runner"""
    callbacks: List[Callback]
    callbacks = [
        dl.CheckpointCallback(
            logdir=logdir, loader_key="valid", metric_key=metric_key, topk=3, resume_model=resume
        ),
        dl.EarlyStoppingCallback(
            loader_key="valid", metric_key="loss", minimize=True, patience=3, min_delta=1e-2
        ),
    ]
    if compute_iou:
        for input_key, targets_key, prefix in zip(INPUTS_KEYS, TARGETS_KEYS, PREFIXES):
            callbacks.append(
                dl.IOUCallback(
                    input_key=input_key,
                    target_key=targets_key,
                    prefix=prefix,
                    class_names=CLASS_NAMES,
                )
            )
    if compute_dice:
        for input_key, targets_key, prefix in zip(INPUTS_KEYS, TARGETS_KEYS, PREFIXES):
            callbacks.append(
                dl.DiceCallback(
                    input_key=input_key,
                    target_key=targets_key,
                    prefix=prefix,
                    class_names=CLASS_NAMES,
                )
            )
    if profiling:
        callbacks.append(dl.ProfilerCallback(loader_key="valid"))
    if num_batch_steps is not None:
        callbacks.append(dl.CheckRunCallback(num_batch_steps=num_batch_steps, num_epoch_steps=10))
    return callbacks


def parse_train_args() -> TrainConfig:
    """Parse train arguments"""
    parser = ArgumentParser()
    # Model parameters
    parser.add_argument(
        "--model_type",
        "-model",
        choices=["Unet", "U2net"],
        type=str,
        default="Unet",
        help="Model type for train (default: %(default)s)",
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
        default=2,
        help="Count model output channels (default: %(default)s)",
    )
    parser.add_argument(
        "--big",
        action="store_true",
        default=False,
        help="Initialize big model (default: %(default)s)",
    )
    parser.add_argument(
        "--max_pool",
        action="store_true",
        default=False,
        help="Use for downsample max pooling, without \
flag convolution with stride=2 (default: %(default)s)",
    )
    parser.add_argument(
        "--bilinear",
        action="store_true",
        default=False,
        help="Use for upsample bilinear, without flag transpose convolution (default: %(default)s)",
    )
    # dataset parameters
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
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=40,
        help="Train loader batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--valid_batch_size",
        type=int,
        default=80,
        help="Validation loader batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--valid_size",
        type=float,
        default=0.3,
        help="Size of valid dataset (default: %(default)s)",
    )
    # training parameters
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=1e-4,
        help="Optimizer learning rate (default: %(default)s)",
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
        "--cpu", action="store_true", default=False, help="Train on cpu (default: %(default)s)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Train in half precision (default: %(default)s)",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint (default: %(default)s)"
    )
    # logging parameters
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
    args = parser.parse_args()
    if args.model_type in ["Unet", "U2net"]:
        model_type = ModelType.UNET if args.model_type == "Unet" else ModelType.U2NET
    else:
        raise ValueError(f"Strange model type: '{args.model_type}'")
    return TrainConfig(
        model=model_type,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        big=args.big,
        max_pool=args.max_pool,
        bilinear=args.bilinear,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        train_batch_size=args.train_batch_size,
        valid_batch_size=args.valid_batch_size,
        valid_size=args.valid_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        num_epochs=args.num_epochs,
        num_batch_steps=args.num_batch_steps,
        cpu=args.cpu,
        fp16=args.fp16,
        resume=args.resume,
        logging=args.logging,
        logdir=args.logdir,
        verbose=args.verbose,
    )


def create_model(
    config: Union[EvaluationConfig, TrainConfig, InferenceConfig]
) -> Union[U2net, Unet]:
    """Create model from config"""
    kwargs: Mapping[str, Any] = {
        "in_channels": config.in_channels,
        "out_channels": config.out_channels,
        "big": config.big,
        "max_pool": config.max_pool,
        "bilinear": config.bilinear,
    }
    return Unet(**kwargs) if config.model is ModelType.UNET else U2net(**kwargs)


def create_logger(config: Union[EvaluationConfig, TrainConfig]) -> Optional[Dict[str, Any]]:
    """Create wandb logger or return None"""
    prefix = "TRAIN_" if isinstance(config, TrainConfig) else "EVAL_"
    if config.logging:
        return {
            "wandb": WandbLogger(
                project="ADAS", name=prefix + config.model.value, log_batch_metrics=True
            )
        }
    return None


def parse_eval_args() -> EvaluationConfig:
    """Parse train arguments"""
    parser = ArgumentParser()
    # Model parameters
    parser.add_argument(
        "--model_type",
        "-model",
        choices=["Unet", "U2net"],
        type=str,
        default="Unet",
        help="Model type for train (default: %(default)s)",
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
        default=2,
        help="Count model output channels (default: %(default)s)",
    )
    parser.add_argument(
        "--big",
        action="store_true",
        default=False,
        help="Initialize big model (default: %(default)s)",
    )
    parser.add_argument(
        "--max_pool",
        action="store_true",
        default=False,
        help="Use for downsample max pooling, without \
flag convolution with stride=2 (default: %(default)s)",
    )
    parser.add_argument(
        "--bilinear",
        action="store_true",
        default=False,
        help="Use for upsample bilinear, without flag transpose convolution (default: %(default)s)",
    )
    # dataset parameters
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
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=80,
        help="Validation loader batch size (default: %(default)s)",
    )
    # training parameters
    parser.add_argument(
        "--cpu", action="store_true", default=False, help="Train on cpu (default: %(default)s)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Train in half precision (default: %(default)s)",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint (default: %(default)s)"
    )
    # logging parameters
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
    args = parser.parse_args()
    if args.model_type in ["Unet", "U2net"]:
        model_type = ModelType.UNET if args.model_type == "Unet" else ModelType.U2NET
    else:
        raise ValueError(f"Strange model type: '{args.model_type}'")
    return EvaluationConfig(
        model=model_type,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        big=args.big,
        max_pool=args.max_pool,
        bilinear=args.bilinear,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        eval_batch_size=args.eval_batch_size,
        cpu=args.cpu,
        fp16=args.fp16,
        resume=args.resume,
        logging=args.logging,
        logdir=args.logdir,
        verbose=args.verbose,
    )
