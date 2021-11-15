import csv

import numpy as np
import torch
from progressbar import progressbar
from torch.nn import CrossEntropyLoss

import model
from module.torch import metrics
from module.torch.base_trainer import BaseTrainer
from module.torch.logger import Logger


class CoEvaluate(BaseTrainer):
    def __init__(self, logger: Logger):
        super().__init__(logger)
        self.model_g, self.model_f1, self.model_f2 = model.get_models(
            models_config=logger.config.models, device=self.device
        )
        self.cce_criterion = self.get_criterion(self.config.models.output_channel)

    def get_criterion(self, n_class):
        weight = torch.ones(n_class)
        criterion = CrossEntropyLoss(weight.to(self.device))
        return criterion

    def build_model(self):
        self.model_g.load_state_dict(torch.load(self.model_path.joinpath(self.config.model_g_filename)))
        self.model_f1.load_state_dict(torch.load(self.model_path.joinpath(self.config.model_f1_filename)))
        self.model_f2.load_state_dict(torch.load(self.model_path.joinpath(self.config.model_f2_filename)))
        self.model_g.to(self.device)
        self.model_f1.to(self.device)
        self.model_f2.to(self.device)
        self.model_g.eval()
        self.model_f1.eval()
        self.model_f2.eval()
        self.set_requires_grad([self.model_g, self.model_f1, self.model_f2], False)

    def forward(self, images: list, label):
        predict2 = self.model_f2(self.model_g(images[0], images[1])[1])
        predict1 = self.model_f1(self.model_g(images[1], images[2])[0])
        predict = predict1 + predict2
        loss = self.cce_criterion(predict, label)
        hist = metrics.calc_hist(predict, label, self.config.models.output_channel)
        return loss.item(), hist, predict

    def evaluate(self, filename: str, images, label):
        loss, hist, predict = self.forward(images, label)
        iou_list = metrics.iou_metrics(hist, mode='none')
        self.preview.show(
            self.create_on_mask_images(images[1], predict, label),
            output_filename='results/{}.png'.format(filename)
        )
        with open(self.validate_path, 'a') as log:
            csv.writer(log).writerow(np.asarray(np.hstack([loss, iou_list])))
        return loss, hist

    def evaluate_run(self, data_loaders: list):
        self.build_model()
        for i, data_loader in enumerate(data_loaders):
            total_loss = []
            total_hist = np.zeros((self.config.models.output_channel, self.config.models.output_channel))
            for time, dataset in enumerate(progressbar(data_loader)):
                images_list = [
                    dataset['image1'].to(self.device),
                    dataset['image2'].to(self.device),
                    dataset['image3'].to(self.device),
                ]
                labels = dataset['label2'].to(self.device)
                times_list = dataset['times']
                if times_list[1] in list(self.config.dataset[i].val_timepoint_list):
                    loss, hist = self.evaluate('{}_{}'.format(i, time), images_list, labels)
                    total_loss.append(loss)
                    total_hist += hist

            with open(self.validate_path, 'a') as log:
                ious = metrics.iou_metrics(total_hist, mode='none')
                writer = csv.writer(log)
                writer.writerow([])
                writer.writerow([""] + ["iou_class"] + [""] * (len(ious) - 1) + ["miou"])
                writer.writerow(np.hstack([np.mean(total_loss), ious, np.nanmean(ious)]))
                writer.writerow([])
            if i == 1:
                print("best target test iou (cell nucleus region only): {}".format(ious[1]))
