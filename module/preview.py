import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


class Preview:
    def __init__(self, log_path: str = None, dir_name='output_images', palette: list = None):
        self.palette = palette
        if log_path is not None:
            self.log_path = os.path.join(log_path, dir_name)
            if not os.path.exists(self.log_path):  # ディレクトリ がなければ
                os.makedirs(self.log_path)
            self.num_exist_files = len(os.listdir(self.log_path))
        else:
            self.log_path = None

    def generate_color_pallet(self, color_id: int):
        if self.palette is not None:
            return self.palette[color_id]
        else:
            if color_id == 0:
                return [0, 0, 0]
            base_pallet = (np.asarray([256, 256, 256]) / ((abs((color_id - 1)) // 7) + 1)).astype(np.uint8) - 1
            bit_pallet = np.asarray(list(format(((7 - color_id) % 7) + 1, 'b').zfill(3))).astype(np.uint8)
            return base_pallet * bit_pallet

    @staticmethod
    def __formatting_scale__(images: np.ndarray):
        if images.dtype is not np.dtype(np.uint8):
            if np.min(images) < 0:
                images = (images - np.min(images))
            if np.max(images) <= 1:
                images = images * 255
            else:
                images = (images - np.min(images)) * (255 / np.max(images))
            images = images.astype(np.uint8)
        return images

    @staticmethod
    def __formatting_images__(images: np.ndarray):
        if not (images.shape[-1] == 1 or images.shape[-1] == 3):  # チャンネルがない場合
            print('add channel')
            images = images.transpose()[np.newaxis]
            images = images[[0, 0, 0], :].transpose()  # ３チャンネル化
        if images.ndim == 3:
            print('add dim')
            images = images[np.newaxis]
        return images

    def __formatting_class__(self, images: np.ndarray, mode):
        # one-hot判定
        if mode == 'onehot':
            print('one hot !')
            images = np.argmax(images, axis=-1)
            images = images.transpose()[np.newaxis].transpose()
            for label_ids in np.unique(images):
                images = np.where(images == label_ids, self.generate_color_pallet(label_ids), images)
            images = images.astype(np.uint8)
        else:
            print('sparse !')
            images = images.transpose()[np.newaxis].transpose()
            for label_ids in np.unique(images):
                images = np.where(images == label_ids, self.generate_color_pallet(label_ids), images)
            images = images.astype(np.uint8)
        return images

    @staticmethod
    def __convert_channel__(images, type_name: str, mode: str):
        # Pytorch:Tensor型のChannelFirstを変換
        if type_name == 'torch':
            print('convert torch')
            images = images.cpu().numpy()
            if images.ndim == 3 and not mode == 'sparse':
                print('convert channel first -> last, dim = 3')
                images = images.transpose(1, 2, 0)
            elif images.ndim == 4:
                print('convert channel first -> last, dim = 4')
                images = images.transpose(0, 2, 3, 1)
        return images

    def show(self, images_array, title: str = None, type_name=None, mode=None):
        assert type(images_array) is tuple or \
               type(images_array) is np.ndarray or \
               type(images_array) is list or \
               'torch' in str(type(images_array)), "show_titleの引数は<tuple> <np.ndarray> <list> <torch Tensor>のみ"
        assert mode is None or mode in ['onehot', 'sparse'], 'mode : None or onehot or sparse'
        assert type_name is None or type_name == 'torch', 'type : None or torch'
        if type(images_array) is tuple:
            images_array = list(images_array)
        elif type(images_array) is np.ndarray or 'torch' in str(type(images_array)):
            images_array = [images_array]

        print("----")

        for images_index, images in enumerate(images_array):
            images = self.__convert_channel__(images, type_name, mode)
            if mode in ['onehot', 'sparse']:
                images = self.__formatting_class__(images, mode)
            images = self.__formatting_images__(images)
            images = self.__formatting_scale__(images)
            assert images.ndim == 4, "不正なshapeです:{}".format(str(images.shape))

            length = math.ceil(math.sqrt(images.shape[0]))
            tile_image = np.full((length, length, images.shape[1], images.shape[2], 3), 255, dtype=np.uint8)
            for i in range(images.shape[0]):
                tile_image[i // length, i % length] = images[i]
            integrated_image = cv2.vconcat([cv2.hconcat(h) for h in tile_image])
            plt.title(title if title is not None else str(images_index))

            if self.log_path is not None:
                self.num_exist_files += 1
                plt.imsave(
                    os.path.join(self.log_path, "{}.png".format(self.num_exist_files)),
                    integrated_image
                )
            else:
                plt.imshow(integrated_image)
                plt.show()
