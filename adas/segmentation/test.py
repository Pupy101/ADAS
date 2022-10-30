from torch.utils.data import DataLoader

from adas.segmentation.data import BDD100KDataset, create_augmentation
from adas.segmentation.utils.common import (
    MultipleOutputModelRunner,
    create_callbacks,
    create_logger,
    create_model,
)
from adas.segmentation.utils.parser import parse_eval_args

if __name__ == "__main__":

    CONFIG = parse_eval_args()

    MODEL = create_model(CONFIG)

    LOGGER = create_logger(CONFIG)

    CALLBACKS = create_callbacks(logdir=CONFIG.logdir, resume=CONFIG.resume, profile=CONFIG.profile)

    EVAL_DATASET = BDD100KDataset(
        data=BDD100KDataset.found_images(image_dir=CONFIG.image_dir, mask_dir=CONFIG.mask_dir),
        transforms=create_augmentation(is_train=False),
    )
    LOADERS = {
        "valid": DataLoader(EVAL_DATASET, batch_size=CONFIG.eval_batch_size, shuffle=False),
    }

    HPARAMS = CONFIG.asdict(exclude=["model"])
    HPARAMS["model"] = CONFIG.model.value

    MultipleOutputModelRunner().train(
        model=MODEL,
        loaders=LOADERS,
        callbacks=CALLBACKS,
        cpu=CONFIG.cpu,
        fp16=CONFIG.fp16,
        loggers=LOGGER,
        logdir=CONFIG.logdir,
        verbose=CONFIG.verbose,
        hparams=HPARAMS,
    )
