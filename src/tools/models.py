import os
from logging import Logger

import dspy
import openai

from config.config import (
    PipelineConfiguration,
    get_pipeline_config,
)
from src.utils.logging_utils import get_logger

APP_CONFIG: PipelineConfiguration = get_pipeline_config()

logger: Logger = get_logger(name=__name__)


class OpenAILM(dspy.LM):
    def __init__(self, model, api_key=None, temperature=0, max_tokens=None, **kwargs):
        """
        Initialize the OpenAILM class.

        Args:
            model (str): The model name to use.
            api_key (str, optional): OpenAI API key. Defaults to None.
            temperature (float, optional): Sampling temperature. Defaults to 0.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        # Set the OpenAI API key
        openai.api_key = os.environ.get("OPENAI_API_KEY") or api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history = []

        # Check if the model is an o1-* model
        if self.model.startswith("o1-"):
            logger.info(f"Detected o1-* model: {self.model}")

            # Enforce temperature=1.0
            if self.temperature != 1.0:
                logger.warning(
                    f"Overriding temperature from {self.temperature} to 1.0 for model {self.model}"
                )
                self.temperature = 1.0

            # Enforce max_tokens >= 5000
            if self.max_tokens is None or self.max_tokens < 5000:
                logger.warning(f"Setting max_tokens to 5000 for model {self.model}")
                self.max_tokens = 5000
            else:
                logger.info(
                    f"Using provided max_tokens={self.max_tokens} for model {self.model}"
                )
        else:
            logger.info(
                f"Using non-o1 model: {self.model} with temperature={self.temperature} and max_tokens={self.max_tokens}"
            )

        # Initialize the superclass with the correct parameters
        super().__init__(
            model, temperature=self.temperature, max_tokens=self.max_tokens, **kwargs
        )
