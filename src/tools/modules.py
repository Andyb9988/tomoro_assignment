from logging import Logger
from typing import (
    Any,
    List,
)

import dspy

from config.config import (
    PipelineConfiguration,
    get_pipeline_config,
)
from src.utils.logging_utils import get_logger
from tools.signatures import GenerateCotAnswer

APP_CONFIG: PipelineConfiguration = get_pipeline_config()

logger: Logger = get_logger(name=__name__)


class OutputFinalAnswer(dspy.Module):
    """
    A module to generate and retrieve a final answer to a single question
    based on provided context using Chain of Thought reasoning.

    Attributes:
        context (str): The context within which the question is answered.
        question (str): The question to answer.
        cot_answer (dspy.ChainOfThought): Chain of Thought instance for generating the answer.
    """

    def __init__(self, context: str, id: str) -> None:
        """
        Initializes OutputFinalAnswerSingle with context and question.

        Args:
            context (str): The context within which the question is answered.
            question (str): The question to be answered.
        """
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(GenerateCotAnswer)
        self.context = context
        self.id = id

    def get_answer(self, question: str) -> Any:
        """
        Generates an answer using Chain of Thought reasoning based on context and question.

        Returns:
            Any: The final answer generated by the Chain of Thought model.
        """
        logger.info(
            f"Generating answer for the single {question} using provided context."
        )

        prediction = self.generate_answer(question=question, context=self.context)
        return prediction
