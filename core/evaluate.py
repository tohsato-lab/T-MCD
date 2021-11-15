import csv
import numpy as np
from progressbar import progressbar
import torch
from torch.nn import CrossEntropyLoss

import model
from module.torch import metrics
from module.torch.base_trainer import BaseTrainer
from module.torch.logger import Logger
import params


class Evaluate(BaseTrainer):
    def __init__(self, logger: Logger):
        super().__init__(logger)
        self.model_g, self.model_f1, self.model_f2 = model.get_models(
            net_name=params.model_name, res=params.res, input_ch=params.input_channel,
            n_class=params.num_class, device=self.device, method='MCD', up_mode=params.up_mode,
            junction_point=params.junction_point,
        )
        self.cce_criterion = self.get_criterion(params.num_class)

    def get_criterion(self, n_class):
        weight = torch.ones(n_class)
        criterion = CrossEntropyLoss(weight.to(self.device))
        return criterion

    def build_model(self):
        self.model_g.load_state_dict(torch.load(self.model_path.joinpath(params.model_g_filename)))
        self.model_f1.load_state_dict(torch.load(self.model_path.joinpath(params.model_f1_filename)))
        self.model_g = self.model_g.to(self.device)
        self.model_f1 = self.model_f1.to(self.device)
        self.model_g.eval()
        self.model_f1.eval()
        self.set_requires_grad([self.model_g, self.model_f1, self.model_f2], False)

    def forward(self, image, label):
        predict = self.model_f1(self.model_g(image))
        loss = self.cce_criterion(predict, label)
        hist = metrics.calc_hist(predict, label, params.num_class)
        return loss.item(), hist, predict

    def evaluate(self, image, label):
        loss, hist, predict = self.forward(image, label)
        iou_list = metrics.iou_metrics(hist, mode='none')
        self.preview.show(self.create_on_mask_images(image, predict, label), 'mask')
        with open(self.validate_path, 'a') as log:
            csv.writer(log).writerow(np.asarray(np.hstack([loss, iou_list])))
        return loss, hist

    def evaluate_run(self, data_loaders: list):
        """
        co detectionに合わせるため、最後の画像は評価しない
        :param data_loaders:
        :return:
        """
        self.build_model()
        for dataloader in data_loaders:
            total_loss = []
            total_hist = np.zeros((params.num_class, params.num_class))
            for time, dataset in enumerate(progressbar(dataloader)):
                image = dataset['image'].to(self.device)
                label = dataset['label'].to(self.device)
                if time in params.val_index_list:
                    loss, hist = self.evaluate(image, label)
                    total_loss += [loss]
                    total_hist += hist

            with open(self.validate_path, 'a') as log:
                ious = metrics.iou_metrics(total_hist, mode='none')
                writer = csv.writer(log)
                writer.writerow([])
                writer.writerow([""] + ["iou_class"] + [""] * (len(ious) - 1) + ["miou"])
                writer.writerow(np.hstack([np.mean(total_loss), ious, np.nanmean(ious)]))
                writer.writerow([])
