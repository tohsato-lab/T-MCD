from module.torch.segmentation_transforms import *


class ImageTransform:
    def __init__(self, size, augment=False, identical=True):
        self.identical = identical
        if augment:
            self.data_transform = {
                'dbscreen_train': Compose([
                    Resize((int(size * 1.5), size)),
                    RandomCrop(size),
                    RandomRot90(),
                    RandomGammaCorrection(),
                    ToTensor(),
                ]),
                'wddd_train': Compose([
                    Resize(int(size * 1.5)),
                    RandomCrop(size),
                    RandomRot90(),
                    RandomGammaCorrection(),
                    ToTensor(),
                ]),
            }
        else:
            self.data_transform = {
                'dbscreen_train': Compose([
                    Resize((int(size * 1.5), size)),
                    CenterCrop(size),
                    ToTensor(),
                ]),
                'wddd_train': Compose([
                    Resize(int(size * 1.5)),
                    CenterCrop(size),
                    ToTensor(),
                ]),
            }
        self.data_transform['dbscreen_test'] = Compose([
            Resize((int(size * 1.5), size)),
            CenterCrop(size),
            ToTensor(),
        ])
        self.data_transform['wddd_test'] = Compose([
            Resize(int(size * 1.5)),
            CenterCrop(size),
            ToTensor(),
        ])
        self.data_transform['common'] = Compose([
            # RandomCrop(resize),
            # CenterCrop(resize),
            # RandomRot90(),
            ToTensor(),
        ])

    def __call__(self, phase='train', **kwargs):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        if self.identical:
            return self.data_transform[phase](kwargs)
        else:
            separated_kwargs = []
            for key in kwargs.keys():
                if 'label' in key:
                    separated_kwargs.append({key: kwargs[key]})
            for i, key in enumerate(kwargs.keys()):
                if 'label' not in key:
                    separated_kwargs[i].setdefault(key, kwargs[key])
            result = {}
            for data in separated_kwargs:
                result.update(self.data_transform[phase](data))
            return result
