from logging import Logger
from typing import (
    Any,
    List,
)

import pandas as pd

from src.utils.logging_utils import get_logger

logger: Logger = get_logger(name=__name__)


def get_llm_answer_outcome_df(
    eval_df: List[Any], llm_ans_list: List[Any]
) -> pd.DataFrame:
    """
    Combines evaluation data with LLM answers into a single outcome DataFrame.

    Args:
        eval_df (List[EvalExample]): A list of EvalExample instances containing evaluation data.
        llm_ans_list (List[LLMAnswer]): A list of LLMAnswer instances containing LLM-generated answers.

    Returns:
        pd.DataFrame: A merged DataFrame containing both evaluation data and LLM answers.

    Raises:
        ValueError: If input data is invalid or merging fails.
    """
    eval_data = []
    for example in eval_df:
        eval_data.append(
            {
                "question": example.question,
                "id": example.id,
                "step_list": example.step_list,
                "dialogue_break": example.dialogue_break,
                "answer": example.answer,
                "exe_answer": example.exe_answer,
            }
        )
    # Create a DataFrame from eval_data
    eval_df = pd.DataFrame(eval_data)

    # Prepare data from llm_ans_list
    llm_data = []
    for prediction in llm_ans_list:
        llm_data.append(
            {"reasoning": prediction.reasoning, "llm_answer": prediction.answer}
        )
    # Create a DataFrame from llm_data
    llm_df = pd.DataFrame(llm_data)

    if len(eval_df) != len(llm_df):
        logger.warning(
            "The number of evaluation examples and LLM answers do not match. Proceeding with concatenation."
        )

    outcome_df = pd.concat([eval_df, llm_df], axis=1)

    return outcome_df
