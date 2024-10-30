import logging
import logging.config
import os
import warnings

import coloredlogs
import yaml
#from google.cloud.logging_v2._helpers import retrieve_metadata_server


warnings.filterwarnings("ignore", message=".*CUDA is not available.*")


def get_logger(name, path="config/logger_config.yaml", default_level=logging.INFO) -> logging.Logger:
    """Function that generates a logger from a configuration file
    or defaults to a preset configuration if a config file is not found."""

    _GCE_INSTANCE_ID = "instance/id"
    """Attribute in metadata server for compute region and instance."""

    gce_instance_name = retrieve_metadata_server(_GCE_INSTANCE_ID)

    if gce_instance_name is None:
        if os.path.exists(path):
            with open(path, "rt") as f:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
                coloredlogs.install()
                logger = logging.getLogger(name)
        else:
            logging.basicConfig(level=default_level)
            coloredlogs.install(level=default_level)
            logger = logging.getLogger(name)

    else:
        logging.basicConfig(level=default_level)
        logger = logging.getLogger(name)

    return logger
