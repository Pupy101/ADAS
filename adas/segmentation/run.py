from catalyst import dl
from catalyst.contrib.losses import DiceLoss, IoULoss
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from adas.segmentation.configs import (
    Config,
    EvaluationEncoderConfig,
    EvaluationSegmentationConfig,
    TrainEncoderConfig,
    TrainSegmentationConfig,
)
from adas.segmentation.data import BDD100KDataset, ImageClassificationDataset, create_augmentation
from adas.segmentation.utils.loss import ManyOutputsLoss
from adas.segmentation.utils.misc import (
    SegmentationRunner,
    create_callbacks,
    create_logger,
    create_model,
)
from adas.utils.misc import train_test_split


def run_classification(config: Config) -> None:
    """Run classification model"""
    assert isinstance(config, (EvaluationEncoderConfig, TrainEncoderConfig))

    model = create_model(config)
    optimizer = (
        AdamW(model.parameters(), lr=config.learning_rate)
        if isinstance(config, TrainEncoderConfig)
        else None
    )
    criterion = nn.CrossEntropyLoss() if isinstance(config, TrainEncoderConfig) else None
    callbacks = create_callbacks(config)
    logger = create_logger(config)
    if isinstance(config, TrainEncoderConfig):
        assert isinstance(config, TrainEncoderConfig)
        train_data, valid_data = train_test_split(
            data=ImageClassificationDataset.found_dataset_data(image_dir=config.image_dir),
            test_size=config.valid_size,
            seed=config.seed,
        )
        train_dataset = ImageClassificationDataset(
            data=train_data, transforms=create_augmentation(is_train=True)
        )
        valid_dataset = ImageClassificationDataset(
            data=valid_data, transforms=create_augmentation(is_train=False)
        )
        loaders = {
            "train": DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=False),
            "valid": DataLoader(valid_dataset, batch_size=config.valid_batch_size, shuffle=False),
        }
        seed = config.seed
        num_epochs = config.num_epochs
    else:
        eval_data = ImageClassificationDataset.found_dataset_data(image_dir=config.image_dir)

        eval_dataset = ImageClassificationDataset(
            data=eval_data, transforms=create_augmentation(is_train=False)
        )
        loaders = {
            "valid": DataLoader(eval_dataset, batch_size=config.valid_batch_size, shuffle=False)
        }
        seed = None
        num_epochs = 1
    dl.SupervisedRunner().train(
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


def run_segmentation(config: Config) -> None:
    """Run segmentation model"""
    assert isinstance(config, (EvaluationSegmentationConfig, TrainSegmentationConfig))
    model = create_model(config)
    optimizer = (
        AdamW(model.parameters(), lr=config.learning_rate)
        if isinstance(config, TrainSegmentationConfig)
        else None
    )
    criterion = (
        ManyOutputsLoss(
            losses=(
                IoULoss(class_dim=config.out_channels),
                DiceLoss(class_dim=config.out_channels),
            ),
            coefficients=tuple(config.predicts_coeffs),
            losses_coefficients=(0.6, 0.4),
        )
        if isinstance(config, TrainSegmentationConfig)
        else None
    )
    callbacks = create_callbacks(config)
    logger = create_logger(config)
    if isinstance(config, TrainSegmentationConfig):
        assert isinstance(config, TrainSegmentationConfig)
        train_data, valid_data = train_test_split(
            data=BDD100KDataset.found_dataset_data(
                image_dir=config.image_dir, mask_dir=config.mask_dir
            ),
            test_size=config.valid_size,
            seed=config.seed,
        )
        train_dataset = BDD100KDataset(
            data=train_data, transforms=create_augmentation(is_train=True)
        )
        valid_dataset = BDD100KDataset(
            data=valid_data, transforms=create_augmentation(is_train=False)
        )
        loaders = {
            "train": DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=False),
            "valid": DataLoader(valid_dataset, batch_size=config.valid_batch_size, shuffle=False),
        }
        seed = config.seed
        num_epochs = config.num_epochs
    else:
        eval_data = BDD100KDataset.found_dataset_data(
            image_dir=config.image_dir, mask_dir=config.mask_dir
        )

        eval_dataset = BDD100KDataset(
            data=eval_data, transforms=create_augmentation(is_train=False)
        )
        loaders = {
            "valid": DataLoader(eval_dataset, batch_size=config.valid_batch_size, shuffle=False)
        }
        seed = None
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
