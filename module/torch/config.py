import sys
from pathlib import Path

from omegaconf import ListConfig, OmegaConf, DictConfig


class Config:
    def __init__(self, config_root, conf_filename):
        self.config_root = Path(config_root)
        self.config_filename = conf_filename

    @staticmethod
    def generate_path(root: Path, paths: list) -> Path:
        for path in paths:
            root = root.joinpath(str(path))
        return root

    def search_config_directory(self, config, paths: list):
        if isinstance(config, (tuple, list, ListConfig)):
            conf_list = []
            for item in config:
                conf_list.append(self.search_config_directory(item, paths))
            return conf_list
        elif isinstance(config, (dict, DictConfig)):
            for key in config.keys():
                config[key] = self.search_config_directory(config[key], paths + [key])
            return config
        else:
            search_path = self.generate_path(Path(paths[0]), paths[1:]).joinpath(str(config) + '.yaml')
            if search_path.exists():
                return self.search_config_directory(
                    OmegaConf.load(search_path),
                    paths
                )
            else:
                return config

    def load(self):
        # from files
        base_config = OmegaConf.load(self.config_root.joinpath(self.config_filename))

        # from cli
        args = []
        for arg in sys.argv[1:]:
            if not Path(arg).exists():
                args.append(arg)
            else:
                base_config = OmegaConf.load(Path(arg).joinpath(self.config_filename))

        config = OmegaConf.merge(base_config, OmegaConf.from_cli(args))
        searched_config = self.search_config_directory(config, [self.config_root.name])
        OmegaConf.resolve(searched_config)

        return searched_config
