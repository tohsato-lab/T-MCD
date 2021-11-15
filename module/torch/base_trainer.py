import csv

import cv2
import numpy as np
import torch

from module.preview import Preview
from module.torch.logger import Logger


class BaseTrainer:
    def __init__(self, logger: Logger):
        self.logger = logger
        self.config = logger.config
        self.model_path = logger.snapshot_path
        self.validate_path = logger.validate_path
        self.preview = Preview(self.logger.log_root, palette=self.logger.palette)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(self.validate_path, 'a') as log:
            csv.writer(log).writerow(["loss", "metrics"])

    # set requies_grad=Fasle to avoid computation
    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    @staticmethod
    def __formatting_scale__(images: np.ndarray):
        if images.dtype is not np.dtype(np.uint8):
            if np.min(images) < 0:
                images = (images - np.min(images))
            if np.max(images) <= 1:
                images = images * 255
            images = images.astype(np.uint8)
        return images

    def create_on_mask_images(self, images, predicts, labels):
        on_mask_images = []
        # print("create mask images")
        for image, predict, label in zip(images, predicts, labels):
            image = self.__formatting_scale__(np.squeeze(image.cpu().numpy().transpose(1, 2, 0)))
            predict = predict.max(dim=0)[1].cpu().numpy()
            label = label.cpu().numpy()
            color_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

            # label
            for class_id in np.unique(label):
                if class_id == 0:
                    continue
                binary_image = (label == class_id).astype(np.uint8)
                binary_image = binary_image * 255
                result = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(color_image, result[-2], -1, (0, 0, 255), 1)

            # predict
            for class_id in np.unique(predict):
                if class_id == 0:
                    continue
                binary_image = (predict == class_id).astype(np.uint8)
                binary_image = binary_image * 255
                result = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(color_image, result[-2], -1, (255, 0, 0), 1)

            on_mask_images.append(color_image)
        return np.asarray(on_mask_images)

    def evaluate_run(self, data_loaders: list):
        pass
