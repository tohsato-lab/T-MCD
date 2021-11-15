from torch.utils.data.dataloader import DataLoader

import params
from core.adapt import Adapt
from core.co_adapt import CoAdapt
from dataset import get_datasets
from module.torch.logger import Logger


def main(logger: Logger):
    src_train_dataset, src_val_dataset, src_test_dataset = get_datasets(
        params.source, params.datasets, ['dbscreen_train', 'dbscreen_test', 'dbscreen_test'], params.image_size,
        params.augment,
        params.dataset_type, params.train_n, params.n_range, params.augment_identical
    )
    tgt_train_dataset, tgt_val_dataset, tgt_test_dataset = get_datasets(
        params.target, params.datasets, ['wddd_train', 'wddd_test', 'wddd_test'], params.image_size, params.augment,
        params.dataset_type, params.train_n, params.n_range, params.augment_identical
    )

    print(params.model_name)
    print(params.dataset_type)
    trainer = Adapt(logger) if params.dataset_type == 'none' else CoAdapt(logger)

    # train
    if logger.train_mode:
        # load dataset
        src_train_dataloader = DataLoader(
            src_train_dataset, batch_size=params.batch_size, shuffle=True, drop_last=True, pin_memory=True,
        )
        tgt_train_dataloader = DataLoader(
            tgt_train_dataset, batch_size=params.batch_size, shuffle=True, drop_last=True, pin_memory=True,
        )
        src_val_dataloader = DataLoader(
            src_val_dataset, batch_size=1, shuffle=False, drop_last=True, pin_memory=True,
        )
        tgt_val_dataloader = DataLoader(
            tgt_val_dataset, batch_size=1, shuffle=False, drop_last=True, pin_memory=True,
        )
        logger.set_dataset_sample([src_train_dataloader, tgt_train_dataloader, src_val_dataloader, tgt_val_dataloader])
        trainer.run(src_train_dataloader, tgt_train_dataloader, src_val_dataloader, tgt_val_dataloader)

    # evaluate
    # load dataset
    src_test_dataloader = DataLoader(src_test_dataset, batch_size=1, shuffle=False)
    tgt_test_dataloader = DataLoader(tgt_test_dataset, batch_size=1, shuffle=False)

    trainer.evaluate_run([src_test_dataloader, tgt_test_dataloader])


if __name__ == '__main__':
    for i in range(params.train_count):
        main(Logger(seed=params.seed, palette=params.palette))
