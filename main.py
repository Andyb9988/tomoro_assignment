import os
from logging import Logger
#from dotenv import load_dotenv
#from langchain.globals import set_debug
from config.config import (
    PipelineConfiguration,
    get_pipeline_config,
)

from src.utils.logging_utils import get_logger
#load_dotenv()

APP_CONFIG: PipelineConfiguration = get_pipeline_config()

logger: Logger = get_logger(name=__name__)