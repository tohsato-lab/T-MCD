from module.torch.segmentation_transforms import *


class ImageTransform:
    def __init__(self):
        self.data_transform = {
            'dbscreen_train': Compose([
                ToTensor(),
            ]),
            'wddd_train': Compose([
                ToTensor(),
            ]),
            'dbscreen_test': Compose([
                ToTensor(),
            ]),
            'wddd_test': Compose([
                ToTensor(),
            ]),
        }

    def __call__(self, phase='train', **kwargs):
        return self.data_transform[phase](kwargs)
