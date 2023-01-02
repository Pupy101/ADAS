from catalyst import dl
from catalyst.contrib.losses import FocalLossMultiClass
from catalyst.contrib.optimizers import AdamP
from torch.utils.data import DataLoader

from adas.core.data.augmentation import create_image_augmentation
from adas.core.data.types import DatasetType
from adas.core.utils.misc import create_logger
from adas.utils.misc import train_test_split

from .config import Config, TrainCfg
from .data.dataset import ImageClassificationDataset
from .utils.misc import create_callbacks, create_model


def run(config: Config) -> None:
    """Run segmentation model"""
    model = create_model(config)
    optimizer = (
        AdamP(model.parameters(), lr=config.learning_rate) if isinstance(config, TrainCfg) else None
    )
    criterion = FocalLossMultiClass()
    callbacks = create_callbacks(config)
    logger = create_logger(config)
    if isinstance(config, TrainCfg):
        train_data, valid_data = train_test_split(
            data=ImageClassificationDataset.found_dataset_data_from_dir(data_dir=config.data_dir),
            test_size=config.valid_size,
            seed=config.seed,
        )
        print("-" * 100)
        print(len(train_data), len(valid_data))
        print("-" * 100)
        train_dataset = ImageClassificationDataset(
            data=train_data,
            transforms=create_image_augmentation(dataset_type=DatasetType.TRAIN.value),
        )
        valid_dataset = ImageClassificationDataset(
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
        eval_data = ImageClassificationDataset.found_dataset_data_from_dir(data_dir=config.data_dir)

        eval_dataset = ImageClassificationDataset(
            data=eval_data,
            transforms=create_image_augmentation(dataset_type=DatasetType.VALID.value),
        )
        loaders = {
            "valid": DataLoader(eval_dataset, batch_size=config.valid_batch_size, shuffle=False)
        }
        seed = 1
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
