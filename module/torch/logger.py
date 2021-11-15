import csv
import os
import random
import sys
import zipfile
from datetime import datetime
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from module.preview import Preview
from omegaconf import DictConfig, OmegaConf
from torch.backends import cudnn


class Logger:
    def __init__(self, config: DictConfig, seed=None, palette=None):
        self.config = config
        self.seed = seed
        self.palette = palette
        self.log_root, self.train_mode = self.set_options()
        self.sampling_preview = Preview(str(self.log_root), palette=self.palette)
        self.config_path = self.log_root.joinpath("config.yaml")
        self.snapshot_path = self.log_root.joinpath("snapshots")
        self.save_src_path = self.log_root.joinpath("src.zip")
        self.model_path = self.log_root.joinpath("models.zip")
        self.score_path = self.log_root.joinpath("score.log")
        self.memo_path = self.log_root.joinpath("memo.txt")
        self.validate_path = self.log_root.joinpath("validate.log")
        self.save_model_count = 0
        self.best = None
        self.monitor_logs = None
        self.monitor_op = None
        self.loss_logs = {}
        self.acc_logs = {}
        os.makedirs(self.log_root, exist_ok=True)
        os.makedirs(self.snapshot_path, exist_ok=True)

        self.save_config(self.config, self.config_path)
        self.save_src_files(self.save_src_path)
        self.fix_seed(self.seed)

    @staticmethod
    def add_date(filename: str) -> str:
        return datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + filename

    @staticmethod
    def set_options():
        log_root = os.path.join("log", datetime.now().strftime('%Y%m%d_%H%M%S'))
        train_mode = True
        for i in range(1, len(sys.argv)):
            if os.path.exists(sys.argv[i]):
                log_root = sys.argv[i]
                train_mode = False

        return Path(log_root), train_mode

    def save_config(self, config, config_path):
        print(OmegaConf.to_yaml(config))
        if not self.train_mode:
            config_path = config_path.parent.joinpath(self.add_date(config_path.name))
        OmegaConf.save(config=config, f=config_path, resolve=True)
        return config

    @staticmethod
    def fix_seed(seed):
        if seed is not None:
            # random
            random.seed(seed)
            # Numpy
            np.random.seed(seed)
            # Pytorch
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

    def save_src_files(self, save_src_path):
        if save_src_path.exists():
            save_src_path = save_src_path.parent.joinpath(self.add_date(save_src_path.name))
        with zipfile.ZipFile(save_src_path, 'w', compression=zipfile.ZIP_DEFLATED) as new_zip:
            for filename in glob('**/*.py', recursive=True):
                new_zip.write(filename)
            for filename in glob('**/*.yaml', recursive=True):
                if 'log' not in filename:
                    new_zip.write(filename)

    def save_model(self, models: dict, pack_zip=False, tmp_filename='.tmp.pt'):
        if pack_zip:
            with zipfile.ZipFile(self.model_path, 'a', compression=zipfile.ZIP_DEFLATED) as model_zip:
                self.save_model_count += 1
                for key in models.keys():
                    torch.save(models[key].state_dict(), tmp_filename)
                    model_zip.write(tmp_filename, arcname='{}-{}.pt'.format(key, self.save_model_count))
            os.remove(tmp_filename)
        else:
            print("## save snapshot ##")
            for key in models.keys():
                filename = key if '.pt' not in key else str(key).split('.')[0]
                if os.path.exists(self.snapshot_path.joinpath('{}.pt'.format(filename))):
                    filename = self.add_date(filename)
                torch.save(models[key].state_dict(), self.snapshot_path.joinpath('{}.pt'.format(filename)))

    def set_snapshot(self, models: dict, monitor, mode=None):
        if self.best is None:
            if mode is None:
                mode = 'min' if 'loss' in monitor else 'max'
            if mode == 'min':
                self.best = np.Inf
                self.monitor_op = np.less
            else:
                self.best = -np.Inf
                self.monitor_op = np.greater

            if 'loss' in monitor:
                self.monitor_logs = self.loss_logs[monitor]
            else:
                self.monitor_logs = self.acc_logs[monitor]

        current = self.monitor_logs[-1]
        if self.monitor_op(current, self.best):
            self.best = current
            print("## save snapshot ##")
            for key in models.keys():
                filename = key if '.pt' not in key else str(key).split('.')[0]
                torch.save(models[key].state_dict(), self.snapshot_path.joinpath('{}.pt'.format(filename)))

    def recode_score(self, epoch: int, logs: dict = None):
        for key in logs.keys():
            if 'loss' in key:
                if key not in self.loss_logs:
                    self.loss_logs.setdefault(key, [])
                self.loss_logs[key].append(logs[key])
            else:
                if key not in self.acc_logs:
                    self.acc_logs.setdefault(key, [])
                self.acc_logs[key].append(logs[key])

        with open(self.score_path, 'a') as log:
            writer = csv.writer(log)
            labels = ["epoch"]
            score_list = [epoch]
            labels.extend(self.loss_logs.keys())
            labels.extend(self.acc_logs.keys())
            if os.stat(self.score_path).st_size == 0:
                writer.writerow(labels)
            for key in labels[1:]:
                score_list.append(logs[key])
            writer.writerow(score_list)

        for key, value in self.loss_logs.items():
            plt.plot(value, label=key, marker=".")
        plt.legend()
        plt.grid()
        plt.savefig(self.log_root.joinpath("loss_history.png"))
        plt.clf()

        for key, value in self.acc_logs.items():
            plt.plot(value, label=key, marker=".")
        plt.legend()
        plt.grid()
        plt.savefig(self.log_root.joinpath("acc_history.png"))
        plt.clf()
