import random

from PIL import Image
import cv2
import numpy as np
import torch
from torchvision.transforms import functional as f

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


class Compose:
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, kwargs) -> dict:
        for t in self.transforms:
            kwargs = t(kwargs)
        return kwargs

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, kwargs: dict) -> dict:
        return {
            key: torch.from_numpy(kwargs[key]) if 'label' in key else f.to_tensor(kwargs[key])
            for key in kwargs.keys()
        }

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, kwargs: dict) -> dict:
        """
        :param kwargs:
        :return:
        """
        return {
            key: kwargs[key] if 'label' in key else f.normalize(kwargs[key], self.mean, self.std, self.inplace)
            for key in kwargs.keys()
        }

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class RandomGammaCorrection(object):
    def __init__(self, min_gamma=0, max_gamma=2):
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma

    @staticmethod
    def gamma_correction(img: np.ndarray, gamma):
        max_val = np.max(img)
        img = max_val * (img / max_val) ** (1 / gamma)
        return img.astype(np.uint8)

    def __call__(self, kwargs: dict) -> dict:
        gamma = random.uniform(self.min_gamma, self.max_gamma)
        return {
            key: np.asarray(kwargs[key]) if 'label' in key else self.gamma_correction(np.asarray(kwargs[key]), gamma)
            for key in kwargs.keys()
        }


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        h, w = img.shape[0], img.shape[1]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = (h - th) // 2
        j = (w - tw) // 2
        return i, j, th, tw

    def __call__(self, kwargs: dict) -> dict:
        i, j, h, w = self.get_params(np.asarray(list(kwargs.values())[0]), self.size)
        return {key: np.asarray(kwargs[key])[i:i + h, j:j + w] for key in kwargs.keys()}

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        h, w = img.shape[0], img.shape[1]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, kwargs: dict) -> dict:
        i, j, h, w = self.get_params(np.asarray(list(kwargs.values())[0]), self.size)
        return {key: np.asarray(kwargs[key])[i:i + h, j:j + w] for key in kwargs.keys()}


class RandomCropInterpolation(object):
    """
    使わない
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        h, w = img.shape[0], img.shape[1]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    @staticmethod
    def interpolation(img, label):
        """
        画像を正方形にする（０埋め）
        :param img:
        :param label:
        :return:
        """
        h, w, c = img.shape[0], img.shape[1], img.shape[2]
        th, tw = max(h, w), max(h, w)
        base_img = np.zeros((th, tw, c))
        base_label = np.zeros((th, tw), dtype=np.int)
        i = (th - h) // 2
        j = (tw - w) // 2
        base_img[i:i + h, j:j + w, :] = img
        base_label[i:i + h, j:j + w] = label
        return base_img, base_label

    def __call__(self, img, label):
        img, label = self.interpolation(np.asarray(img), np.asarray(label))
        i, j, h, w = self.get_params(img, self.size)
        img = img[i:i + h, j:j + w]
        label = label[i:i + h, j:j + w]
        return img, label


class RandomRot90(object):
    def __call__(self, kwargs: dict) -> dict:
        # copyはいずれいらなくなる模様
        images = [np.asarray(image) for image in kwargs.values()]
        for _ in range(random.randint(0, 3)):
            for i in range(len(images)):
                images[i] = np.rot90(images[i])

        return {key: images[i].copy() for i, key in enumerate(kwargs.keys())}


class Rot90:
    def __init__(self, count=1):
        self.count = count

    def __call__(self, kwargs: dict) -> dict:
        # copyはいずれいらなくなる模様
        images = [np.asarray(image) for image in kwargs.values()]
        for _ in range(self.count):
            for i in range(len(images)):
                images[i] = np.rot90(images[i])

        return {key: images[i].copy() for i, key in enumerate(kwargs.keys())}


class Resize(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, kwargs: dict) -> dict:
        return {
            key: cv2.resize(
                np.asarray(kwargs[key]).astype(np.uint8), self.size, interpolation=cv2.INTER_NEAREST
            ).astype(np.int)
            if 'label' in key else cv2.resize(np.asarray(kwargs[key]), self.size) for key in kwargs.keys()
        }


class Chimera:
    def __init__(self, rate=0.5, split=(4, 4)):
        self.rate = rate
        if isinstance(split, int):
            split = (int(split), int(split))
        else:
            split = split
        self.split = split

    def get_select_indexes(self):
        num_target = self.split[0] * self.split[1]
        return random.choices(range(num_target), k=int(num_target * self.rate))

    def get_cell_params(self, size: tuple, index: int):
        cell_height = size[0] // self.split[0]
        cell_width = size[1] // self.split[1]
        h_point = index // self.split[0]
        w_point = index % self.split[1]
        h0 = h_point * cell_height
        h1 = h0 + cell_height
        w0 = w_point * cell_width
        w1 = w0 + cell_width
        return h0, h1, w0, w1

    def swap_cell(self, image: np.ndarray, select_indexes: list):
        h, w = image.shape[0], image.shape[1]
        for i in range(0, int(len(select_indexes) - 0.5), 2):
            c0 = self.get_cell_params((h, w), select_indexes[i])
            c1 = self.get_cell_params((h, w), select_indexes[i + 1])
            tmp = image[c0[0]:c0[1], c0[2]:c0[3]].copy()
            image[c0[0]:c0[1], c0[2]:c0[3]] = image[c1[0]:c1[1], c1[2]:c1[3]]
            image[c1[0]:c1[1], c1[2]:c1[3]] = tmp.copy()
        return image

    def __call__(self, kwargs: dict) -> dict:
        select_indexes = self.get_select_indexes()
        return {key: self.swap_cell(np.asarray(kwargs[key]), select_indexes) for key in kwargs.keys()}
