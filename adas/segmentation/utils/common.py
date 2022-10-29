from argparse import ArgumentParser
from typing import Tuple

from adas.segmentation.utils.configs import ModelType, TrainConfig


def parse_train_args() -> Tuple[TrainConfig, int, int]:
    parser = ArgumentParser()
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
        help="Use for downsample max pooling, without flag convolution with stride=2 (default: %(default)s)",
    )
    parser.add_argument(
        "--bilinear",
        action="store_true",
        default=False,
        help="Use for upsample bilinear, without flag transpose convolution (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Train seed (default: %(default)s)")
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Count training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--run_n_batches",
        "-n_batches",
        type=int,
        default=None,
        help="Run only n batches from loaders (default: %(default)s)",
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=1e-4,
        help="Optimizer learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="./logs",
        help="Directory for logs and checkpoints (default: %(default)s)",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint (default: %(default)s)"
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
        "--ddp", action="store_true", default=False, help="DDP train (default: %(default)s)"
    )
    args = parser.parse_args()
    if args.model_type in ["Unet", "U2net"]:
        model_type = ModelType.UNET if args.model_type == "Unet" else ModelType.U2NET
    else:
        raise ValueError(f"Strange model type: '{args.model_type}'")
    return (
        TrainConfig(
            model=model_type,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            big=args.big,
            max_pool=args.max_pool,
            bilinear=args.bilinear,
            learning_rate=args.learning_rate,
            seed=args.seed,
            logging=args.logging,
            num_epochs=args.num_epochs,
            count_batches=args.run_n_batches,
            logdir=args.logdir,
            resume=args.resume,
            verbose=args.verbose,
            cpu=args.cpu,
            fp16=args.fp16,
            ddp=args.ddp,
        ),
        args.train_batch_size,
        args.valid_batch_size,
    )
