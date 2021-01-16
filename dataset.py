import albumentations
import torch
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class OCR_data:
    def __init__(self, image_path, target, resize=None):
        self.image_paths = image_path
        self.target = target
        self.resize = resize
        self.aug = albumentations.Compose([albumentations.Normalize(always_apply=True)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item]).convert("RGB")
        target = self.target[item]
        if self.resize:
            image = image.resize((self.resize[1], self.resize[0]), resample=Image.BILINEAR)

        image = np.array(image)
        aug = self.aug(image=image)
        image = aug['image']
        # to make channel first
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        '''return {'images': torch.tensor(image, dtype=torch.float),
                'targets': torch.tensor(target, dtype=torch.long)}'''
        return {'images': torch.tensor(image, dtype=torch.float),
                'targets': target}
