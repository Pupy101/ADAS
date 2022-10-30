from argparse import ArgumentParser

from adas.segmentation.utils.configs import EvaluationConfig, ModelType, TrainConfig


def parse_train_args() -> TrainConfig:
    """Parse train arguments"""
    parser = ArgumentParser()
    # Model parameters
    parser.add_argument(
        "--model_type",
        "-model",
        choices=["UNET", "U2NET"],
        type=str,
        default="UNET",
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
        default=120,
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
    parser.add_argument(
        "--name_run",
        "-name",
        type=str,
        default="",
        help="Name of experiment in wandb \
(default: TRAIN_<model_type>)",
    )
    parser.add_argument(
        "--profile",
        "-p",
        action="store_true",
        default=False,
        help="Profile first validation batch (default: %(default)s)",
    )
    args = parser.parse_args()
    if args.model_type in ["UNET", "U2NET"]:
        model_type = ModelType.UNET if args.model_type == "UNET" else ModelType.U2NET
    else:
        raise ValueError(f"Strange model type: '{args.model_type}'")
    if not args.name_run:
        name = "TRAIN_" + args.model_type
    else:
        name = args.name_run
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
        name=name,
        profile=args.profile,
    )


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
    parser.add_argument(
        "--name_run",
        "-name",
        type=str,
        default="",
        help="Name of experiment in wandb \
(default: TRAIN_<model_type>)",
    )
    parser.add_argument(
        "--profile",
        "-p",
        action="store_true",
        default=False,
        help="Profile first validation batch (default: %(default)s)",
    )
    args = parser.parse_args()
    if args.model_type in ["Unet", "U2net"]:
        model_type = ModelType.UNET if args.model_type == "Unet" else ModelType.U2NET
    else:
        raise ValueError(f"Strange model type: '{args.model_type}'")
    if not args.name_run:
        name = "TRAIN_" + args.model_type
    else:
        name = args.name_run
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
        name=name,
        profile=args.profile,
    )
