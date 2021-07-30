import pathlib

import cv2

from torch.utils.data import Dataset


class BDD100K(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, transforms=None, check=False):
        self.images, self.masks = BDD100K.__load_image(image_dir, mask_dir)
        self.transforms = transforms
        if check:
            self._check()

    @staticmethod
    def _load_image(image_dir, mask_dir):
        image_dir = pathlib.Path(image_dir)
        mask_dir = pathlib.Path(mask_dir)
        images = sorted(image_dir.rglob('*/*.jpg'))
        masks = sorted(mask_dir.rglob('*/*.png'))
        return images, masks

    def _check(self):
        bad_images = []
        bad_masks = []
        for image, mask in zip(self.images, self.masks):
            img = cv2.imread(image)
            msk = cv2.imread(mask)
            if img.shape[2] != 3 or msk.shape[2] != 3:
                bad_images.append(image)
                bad_masks.append(mask)
        for bad_image, bad_mask in zip(bad_images, bad_masks):
            self.images.remove(bad_image)
            self.masks.remove(bad_mask)

    def __getitem__(self, ind):

        image_path = self.images[ind]
        mask_path = self.masks[ind]
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        msk = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)

        data = {'image': img, 'mask': msk}

        if self.transforms:
            data = self.transforms(**data)
            img, msk = data['image'], data['mask']

        msk = msk.squeeze(0).permute(2, 0, 1)
        msk[1] = -(msk[0] + msk[2] - 1)
        return {'features': img, 'targets': msk}

    def __len__(self):
        return len(self.images)


