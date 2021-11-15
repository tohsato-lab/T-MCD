from utils.custom_constant_time_dataset import CustomConstantTimeDataset
from utils.custom_dataset import CustomDataset
from utils.custom_random_time_dataset import CustomRandomTimeDataset
from utils.image_transform import ImageTransform


def get_datasets(config, domain):
    dataset_config = config.dataset[int(not domain == 'source')]
    dataset_type = config.models.model.dataset_type
    if dataset_type == 'none':
        train_dataset = CustomDataset(
            dataset_root=dataset_config.train,
            transform=ImageTransform(),
            phase="{}_train".format(dataset_config.dataset_name),
        )
        val_dataset = CustomDataset(
            dataset_root=dataset_config.val,
            transform=ImageTransform(),
            phase="{}_test".format(dataset_config.dataset_name),
        )
        test_dataset = CustomDataset(
            dataset_root=dataset_config.test,
            transform=ImageTransform(),
            phase="{}_test".format(dataset_config.dataset_name),
        )
    else:
        if dataset_type == 'random':
            train_dataset = CustomRandomTimeDataset(
                dataset_root=dataset_config.train,
                transform=ImageTransform(),
                phase="{}_train".format(dataset_config.dataset_name),
            )
        elif dataset_type == 'n_random':
            train_dataset = CustomRandomTimeDataset(
                dataset_root=dataset_config.train,
                transform=ImageTransform(),
                phase="{}_train".format(dataset_config.dataset_name),
                n_range=config.models.model.n_range,
            )
        else:
            train_dataset = CustomConstantTimeDataset(
                dataset_root=dataset_config.train,
                transform=ImageTransform(),
                phase="{}_train".format(dataset_config.dataset_name),
                n=0 if dataset_type == 'same' else config.models.model.train_n,
            )
        val_dataset = CustomConstantTimeDataset(
            dataset_root=dataset_config.val,
            transform=ImageTransform(),
            phase="{}_train".format(dataset_config.dataset_name),
            n=0 if dataset_type == 'same' else config.models.model.train_n,
        )
        test_dataset = CustomConstantTimeDataset(
            dataset_root=dataset_config.test,
            transform=ImageTransform(),
            phase="{}_train".format(dataset_config.dataset_name),
            n=0 if dataset_type == 'same' else config.models.model.train_n,
        )

    return train_dataset, val_dataset, test_dataset
