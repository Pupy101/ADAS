from argparse import ArgumentParser
from typing import Tuple

from adas.segmentation.utils.configs import ModelType, TrainConfig


def parse_train_args() -> Tuple[TrainConfig, int, int]:
    parser = ArgumentParser()
    parser.add_argument("--train_batch_size", type=int, default=40)
    parser.add_argument("--valid_batch_size", type=int, default=80)
    parser.add_argument("--model_type", choices=["Unet", "U2net"], type=str, default="Unet")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--out_channels", type=int, default=2)
    parser.add_argument("--big", action="store_true", default=False)
    parser.add_argument("--max_pool", action="store_true", default=False)
    parser.add_argument("--bilinear", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    parser.add_argument("--logging", "-l", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--ddp", action="store_true", default=False)
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
