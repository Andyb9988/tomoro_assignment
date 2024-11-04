import logging
import logging.config
import os
import warnings

import coloredlogs
import yaml

warnings.filterwarnings("ignore", message=".*CUDA is not available.*")


def get_logger(
    name, path="config/logger_config.yaml", default_level=logging.INFO
) -> logging.Logger:
    """Function that generates a logger from a configuration file
    or defaults to a preset configuration if a config file is not found."""

    if os.path.exists(path):
        with open(path, "rt") as f:
            config = yaml.safe_load(f.read())

            file_handlers = config.get("handlers", {})
            for handler_name, handler in file_handlers.items():
                if "filename" in handler:
                    log_dir = os.path.dirname(handler["filename"])
                    if log_dir and not os.path.exists(log_dir):
                        os.makedirs(log_dir, exist_ok=True)

            logging.config.dictConfig(config)
            coloredlogs.install()
            logger = logging.getLogger(name)
    else:
        logging.basicConfig(level=default_level)
        coloredlogs.install(level=default_level)
        logger = logging.getLogger(name)

    return logger
