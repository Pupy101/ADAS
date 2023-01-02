from catalyst.contrib.losses import DiceLoss, IoULoss
from catalyst.contrib.optimizers import AdamP
from torch.utils.data import DataLoader

from adas.core.data.augmentation import create_image_augmentation
from adas.core.data.types import DatasetType
from adas.core.utils.misc import create_logger, train_test_split

from .config import Config, TrainCfg
from .data.dataset import BDD100KDataset
from .utils.loss import ManyOutputsLoss
from .utils.misc import SegmentationRunner, create_callbacks, create_model, load_encoder_weights


def run(config: Config) -> None:
    """Run segmentation model"""
    model = create_model(config)
    if isinstance(config, TrainCfg) and config.resume_encoder:
        load_encoder_weights(str(config.resume_encoder), model)
    optimizer = (
        AdamP(model.parameters(), lr=config.learning_rate) if isinstance(config, TrainCfg) else None
    )
    criterion = ManyOutputsLoss(
        losses=(IoULoss(class_dim=config.out_channels), DiceLoss(class_dim=config.out_channels)),
        coefficients=tuple(config.predicts_coeffs),
        losses_coefficients=(0.6, 0.4),
    )
    callbacks = create_callbacks(config)
    logger = create_logger(config)
    if isinstance(config, TrainCfg):
        train_data, valid_data = train_test_split(
            data=BDD100KDataset.found_dataset_data(
                image_dir=config.image_dir, mask_dir=config.mask_dir
            ),
            test_size=config.valid_size,
            seed=config.seed,
        )
        train_dataset = BDD100KDataset(
            data=train_data,
            transforms=create_image_augmentation(dataset_type=DatasetType.TRAIN.value),
        )
        valid_dataset = BDD100KDataset(
            data=valid_data,
            transforms=create_image_augmentation(dataset_type=DatasetType.VALID.value),
        )
        loaders = {
            "train": DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True),
            "valid": DataLoader(valid_dataset, batch_size=config.valid_batch_size, shuffle=False),
        }
        seed = config.seed
        num_epochs = config.num_epochs
    else:
        eval_data = BDD100KDataset.found_dataset_data(
            image_dir=config.image_dir, mask_dir=config.mask_dir
        )

        eval_dataset = BDD100KDataset(
            data=eval_data,
            transforms=create_image_augmentation(dataset_type=DatasetType.VALID.value),
        )
        loaders = {
            "valid": DataLoader(eval_dataset, batch_size=config.valid_batch_size, shuffle=False)
        }
        seed = 1
        num_epochs = 1

    SegmentationRunner().train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        seed=seed,
        loaders=loaders,
        callbacks=callbacks,
        cpu=config.cpu,
        fp16=config.fp16,
        num_epochs=num_epochs,
        loggers=logger,
        logdir=config.logdir,
        verbose=config.verbose,
        hparams=config.asdict(),
    )
