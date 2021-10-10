import pathlib

import cv2
import numpy as np
import albumentations as A

from tqdm import tqdm
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2


class BDD100K(Dataset):
    """
    Dataset for train segmentation net based on BD100K https://bair.berkeley.edu/blog/2018/05/30/bdd/'
    """
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        ext_images: str = 'jpg',
        prefix_with_ext_mask: str = '_drivable_color.png',
        transforms=None,
        check: bool = False
    ):
        self.image_dir = pathlib.Path(image_dir)
        self.mask_dir = pathlib.Path(mask_dir)
        self.ext_images = ext_images
        self.prefix = prefix_with_ext_mask
        self.images = sorted(self.image_dir.rglob(f'*.{self.ext_images}'))
        self.transforms = transforms
        if check:
            self._check()

    @staticmethod
    def load_image_and_mask(image_path, mask_path):
        img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        msk = cv2.cvtColor(cv2.imread(str(mask_path)), cv2.COLOR_BGR2RGB)
        return img, msk, mask_path

    def _check(self):
        bad_images = []
        self.masks = []
        for image_path in tqdm(self.images):
            mask_path = self.mask_dir / f'{image_path.stem}{self.prefix}'
            if mask_path.exists():
                self.masks.append(mask_path)
            else:
                bad_images.append(image_path)
        for bad_image in bad_images:
            self.images.remove(bad_image)

    def __getitem__(self, ind):
        image_path = self.images[ind]
        mask_path = self.masks[ind]
        img, msk, _ = BDD100K.load_image_and_mask(image_path, mask_path)

        data = {'image': img, 'mask': msk}

        if self.transforms:
            data = self.transforms(**data)
            img, msk = data['image'], data['mask']
        msk = np.transpose(msk, (2, 0, 1))
        msk[1] = -(msk[0] + msk[2] - 1)
        return {'features': img, 'targets': msk}

    def __len__(self):
        return len(self.images)


# Dataset for inference segmentation net
class InferenceDataset(Dataset):
    def __init__(self, dir_images, transforms):
        self.dir_images = pathlib.Path(dir_images)
        self.transforms = transforms
        self.files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.files.extend(self.dir_images.rglob(ext))
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, ind):
        image_path = self.files[ind]
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        if self.transforms:
            image = self.transforms(image=image)['image']
        return {'features': image, 'name': [self.files[ind]]}


# transforms for train stage net
train_transform = A.Compose(
    [
        A.Flip(),
        A.Resize(300, 300, p=1),
        A.CoarseDropout(),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.3
        ),
        A.GridDistortion(p=0.3),
        A.HueSaturationValue(p=0.3),
        A.Normalize(),
        ToTensorV2()
    ]
)

# transforms for validation and inference stages net
valid_transforms = A.Compose(
    [
        A.Resize(300, 300, p=1),
        A.Normalize(),
        ToTensorV2()
    ]
)
