from catalyst.contrib.losses import DiceLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader

from adas.common.loss import AggregatorManyOutputsLoss
from adas.common.utils import train_test_split
from adas.segmentation.data import BDD100KDataset, create_augmentation
from adas.segmentation.utils.common import (
    MultipleOutputModelRunner,
    create_callbacks,
    create_logger,
    create_model,
    parse_train_args,
)
from adas.segmentation.utils.configs import COEFFICIENTS

if __name__ == "__main__":

    CONFIG = parse_train_args()

    MODEL = create_model(CONFIG)

    OPTIMIZER = AdamW(MODEL.parameters(), lr=CONFIG.learning_rate)

    CRITERION = AggregatorManyOutputsLoss(
        losses=DiceLoss(class_dim=CONFIG.out_channels), coefficients=COEFFICIENTS[CONFIG.model]
    )

    LOGGER = create_logger(CONFIG)

    CALLBACKS = create_callbacks(
        resume=CONFIG.resume, logdir=CONFIG.logdir, num_batch_steps=CONFIG.num_batch_steps
    )

    TRAIN_DATA, VALID_DATA = train_test_split(
        data=BDD100KDataset.found_images(image_dir=CONFIG.image_dir, mask_dir=CONFIG.mask_dir),
        test_size=CONFIG.valid_size,
        seed=CONFIG.seed,
    )
    TRAIN_DATASET = BDD100KDataset(data=TRAIN_DATA, transforms=create_augmentation(is_train=True))
    VALID_DATASET = BDD100KDataset(data=VALID_DATA, transforms=create_augmentation(is_train=False))
    LOADERS = {
        "train": DataLoader(TRAIN_DATASET, batch_size=CONFIG.train_batch_size, shuffle=True),
        "valid": DataLoader(VALID_DATASET, batch_size=CONFIG.valid_batch_size, shuffle=False),
    }

    HPARAMS = CONFIG.asdict(exclude=["model"])
    HPARAMS["model"] = CONFIG.model.value
    if isinstance(CRITERION, AggregatorManyOutputsLoss):
        HPARAMS["loss_coefs"] = CRITERION.coeffs

    MultipleOutputModelRunner().train(
        model=MODEL,
        optimizer=OPTIMIZER,
        criterion=CRITERION,
        seed=CONFIG.seed,
        loaders=LOADERS,
        callbacks=CALLBACKS,
        cpu=CONFIG.cpu,
        fp16=CONFIG.fp16,
        num_epochs=CONFIG.num_epochs,
        loggers=LOGGER,
        logdir=CONFIG.logdir,
        verbose=CONFIG.verbose,
        hparams=HPARAMS,
    )
