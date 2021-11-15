import itertools
import numpy as np
from progressbar import progressbar
import torch
import torch.nn.functional as f
import torch.optim as optim
from torchsummary import summary

from core.evaluate import Evaluate
from module.torch import metrics
from module.torch.logger import Logger
import params


class Adapt(Evaluate):
    def __init__(self, logger: Logger):
        super().__init__(logger)
        self.optimizer_g = optim.Adam(
            itertools.chain(self.model_g.parameters()),
            lr=params.learning_rate,
            weight_decay=params.weight_decay,
            # momentum=params.momentum,
        )
        self.optimizer_f = optim.Adam(
            itertools.chain(self.model_f1.parameters(), self.model_f2.parameters()),
            lr=params.learning_rate,
            weight_decay=params.weight_decay,
            # momentum=params.momentum,
        )
        summary(self.model_g, (1, 256, 256), 5)

    @staticmethod
    def discrepancy(out1, out2):
        return torch.mean(torch.abs(f.softmax(out1, dim=1) - f.softmax(out2, dim=1)))

    def backward_step1(self, images_src, label_src, retain_graph=False):
        feature = self.model_g(images_src)
        predict_class1 = self.model_f1(feature)
        predict_class2 = self.model_f2(feature)
        loss_c = self.cce_criterion(predict_class1, label_src) + self.cce_criterion(predict_class2, label_src)
        loss_c.backward(retain_graph=retain_graph)
        return loss_c.item()

    def backward_step2(self, images_src, label_src, images_tgt, retain_graph=False):
        feature = self.model_g(images_src)
        predict_class1_src = self.model_f1(feature)
        predict_class2_src = self.model_f2(feature)

        feature = self.model_g(images_tgt)
        predict_class1_tgt = self.model_f1(feature)
        predict_class2_tgt = self.model_f2(feature)

        loss_c1 = self.cce_criterion(predict_class1_src, label_src)
        loss_c2 = self.cce_criterion(predict_class2_src, label_src)
        loss_c = loss_c1 + loss_c2
        loss_dis = self.discrepancy(predict_class1_tgt, predict_class2_tgt)
        loss = loss_c - loss_dis
        loss.backward(retain_graph=retain_graph)
        return loss_c.item(), loss_dis.item(), loss.item()

    def backward_step3(self, images_tgt, retain_graph=False):
        feature = self.model_g(images_tgt)
        predict_class1_tgt = self.model_f1(feature)
        predict_class2_tgt = self.model_f2(feature)
        loss_dis = self.discrepancy(predict_class1_tgt, predict_class2_tgt)
        loss_dis.backward(retain_graph=retain_graph)
        return loss_dis.item()

    def run(self, src_train_loader, tgt_train_loader, src_val_loader, tgt_val_loader):
        for epoch in range(params.num_epochs):
            print('\nEpoch {}/{}'.format(epoch + 1, params.num_epochs))
            print('-------------')

            step1_train_losses = []
            step2_train_losses = []
            step3_train_losses = []

            self.model_g.train()
            self.model_f1.train()
            self.model_f2.train()

            for _ in progressbar(range(params.num_iter)):
                src_datasets = next(iter(src_train_loader))
                tgt_datasets = next(iter(tgt_train_loader))
                images_src = src_datasets['image'].to(self.device)
                label_src = src_datasets['label'].to(self.device)
                images_tgt = tgt_datasets['image'].to(self.device)

                ###########################
                #         STEP1           #
                ###########################
                self.set_requires_grad([self.model_g, self.model_f1, self.model_f2], True)
                self.optimizer_g.zero_grad()
                self.optimizer_f.zero_grad()
                step1_train_losses.append(self.backward_step1(images_src, label_src))
                self.optimizer_g.step()
                self.optimizer_f.step()

                ###########################
                #         STEP2           #
                ###########################
                self.set_requires_grad([self.model_g], False)
                self.optimizer_f.zero_grad()
                step2_train_losses.append(self.backward_step2(images_src, label_src, images_tgt))
                self.optimizer_f.step()

                ###########################
                #         STEP3           #
                ###########################
                self.set_requires_grad([self.model_g], True)
                self.set_requires_grad([self.model_f1, self.model_f2], False)
                step3_loss = None
                for _ in range(params.num_k):
                    self.optimizer_g.zero_grad()
                    step3_loss = self.backward_step3(images_tgt)
                    self.optimizer_g.step()
                step3_train_losses.append(step3_loss)

            print("Epoch [{}/{}]: 1th_c_loss={:.3f} 2th_loss={:.3f} 3th_dis_loss={:.3f}".format(
                epoch + 1,
                params.num_epochs,
                np.mean(step1_train_losses),
                np.mean(np.asarray(step2_train_losses)[:, 0]),
                np.mean(step3_train_losses),
            ))

            if np.mean(np.asarray(step2_train_losses)[:, 2]) < 0:
                print("## early stop ##")
                return

            # Validate
            self.model_g.eval()
            self.model_f1.eval()
            self.model_f2.eval()
            self.set_requires_grad([self.model_g, self.model_f1, self.model_f2], False)
            validations = []
            for val_loader in [src_val_loader, tgt_val_loader]:
                total_loss = []
                total_hist = np.zeros((params.num_class, params.num_class))
                for time, dataset in enumerate(progressbar(val_loader)):
                    images = dataset['image'].to(self.device)
                    labels = dataset['label'].to(self.device)
                    if time in params.val_index_list:
                        loss, hist = self.forward(images, labels)[:2]
                        total_loss += [loss]
                        total_hist += hist

                validations.append((np.mean(total_loss), metrics.iou_metrics(total_hist, mode='mean')))

            self.logger.recode_score(
                (epoch + 1),
                {
                    "step1_c_loss": np.mean(step1_train_losses),
                    "step2_c_loss": np.mean(np.asarray(step2_train_losses)[:, 0]),
                    "step2_dis_loss": np.mean(np.asarray(step2_train_losses)[:, 1]),
                    "step2_loss": np.mean(np.asarray(step2_train_losses)[:, 2]),
                    "step3_dis_loss": np.mean(step3_train_losses),
                    "src_val_loss": validations[0][0],
                    "src_val_iou": validations[0][1],
                    "tgt_val_loss": validations[1][0],
                    "tgt_val_iou": validations[1][1],
                }
            )
            self.logger.set_snapshot(
                models={
                    params.model_g_filename: self.model_g,
                    params.model_f1_filename: self.model_f1,
                    params.model_f2_filename: self.model_f2,
                },
                monitor='tgt_val_iou'
            )

            print("tgt", validations[1][0], validations[1][1])
            print("src", validations[0][0], validations[0][1])

        self.logger.save_model(
            models={
                'final_' + params.model_g_filename: self.model_g,
                'final_' + params.model_f1_filename: self.model_f1,
                'final_' + params.model_f2_filename: self.model_f2,
            },
        )
