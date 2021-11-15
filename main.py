from torch.utils.data.dataloader import DataLoader

from core.co_adapt import CoAdapt
from dataset import get_datasets
from module.torch.config import Config
from module.torch.logger import Logger


def main(logger: Logger):
    src_train_dataset, src_val_dataset, src_test_dataset = get_datasets(config=logger.config, domain='source')
    tgt_train_dataset, tgt_val_dataset, tgt_test_dataset = get_datasets(config=logger.config, domain='target')

    trainer = CoAdapt(logger)
    # train
    if logger.train_mode:
        # load dataset
        src_train_data_loader = DataLoader(
            src_train_dataset, batch_size=logger.config.train.batch_size, shuffle=True, drop_last=True, pin_memory=True,
        )
        tgt_train_data_loader = DataLoader(
            tgt_train_dataset, batch_size=logger.config.train.batch_size, shuffle=True, drop_last=True, pin_memory=True,
        )
        src_val_data_loader = DataLoader(
            src_val_dataset, batch_size=1, shuffle=False, drop_last=True, pin_memory=True,
        )
        tgt_val_data_loader = DataLoader(
            tgt_val_dataset, batch_size=1, shuffle=False, drop_last=True, pin_memory=True,
        )
        trainer.run(src_train_data_loader, tgt_train_data_loader, src_val_data_loader, tgt_val_data_loader)

    # evaluate
    # load dataset
    src_test_data_loader = DataLoader(src_test_dataset, batch_size=1, shuffle=False)
    tgt_test_data_loader = DataLoader(tgt_test_dataset, batch_size=1, shuffle=False)

    trainer.evaluate_run([src_test_data_loader, tgt_test_data_loader])


if __name__ == '__main__':
    config = Config(config_root='config', conf_filename='config.yaml').load()
    for i in range(config.train_count):
        main(Logger(config=config, seed=eval(str(config.seed)), palette=list(config.palette)))
