from logging import Logger
from typing import (
    Any,
    List,
)

import pandas as pd

from src.utils.logging_utils import get_logger

logger: Logger = get_logger(name=__name__)


def get_llm_answer_outcome_df(
    eval_df: List[Any],
    llm_ans_list: List[Any],
    reasoning_df: pd.DataFrame = None,
    accuracy_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Combines evaluation data with LLM answers, reasoning, and accuracy into a single outcome DataFrame.

    Args:
        eval_df (List[Any]): A list of EvalExample instances containing evaluation data.
        llm_ans_list (List[Any]): A list of LLMAnswer instances containing LLM-generated answers.
        reasoning_df (pd.DataFrame, optional): DataFrame containing reasoning information with an 'id' column.
        accuracy_df (pd.DataFrame, optional): DataFrame containing accuracy metrics with an 'id' column.

    Returns:
        pd.DataFrame: A merged DataFrame containing evaluation data, LLM answers, reasoning, and accuracy metrics.

    Raises:
        ValueError: If input data is invalid or required columns are missing.
    """
    # Validate inputs
    if not isinstance(eval_df, list):
        logger.error("eval_df must be a list of EvalExample instances.")
        raise ValueError("eval_df must be a list of EvalExample instances.")

    if not isinstance(llm_ans_list, list):
        logger.error("llm_ans_list must be a list of LLMAnswer instances.")
        raise ValueError("llm_ans_list must be a list of LLMAnswer instances.")

    # Convert eval_df to DataFrame
    eval_data = []
    for example in eval_df:
        eval_data.append(
            {
                "id": example.id,
                "question": example.question,
                "context": example.context,
                "dialogue_break": example.dialogue_break,
                "answer": example.answer,
                "exe_answer": example.exe_answer,
            }
        )
    eval_df_pd = pd.DataFrame(eval_data)
    logger.info(f"Converted eval_df to DataFrame with {len(eval_df_pd)} entries.")

    # Convert llm_ans_list to DataFrame
    llm_data = []
    for prediction in llm_ans_list:
        llm_data.append(
            {
                "reasoning": prediction.reasoning,
                "llm_answer": prediction.answer,
            }
        )
    llm_df_pd = pd.DataFrame(llm_data)
    logger.info(f"Converted llm_ans_list to DataFrame with {len(llm_df_pd)} entries.")

    # Check for matching lengths and log a warning if they differ
    if len(eval_df_pd) != len(llm_df_pd):
        logger.warning(
            "The number of evaluation examples and LLM answers do not match. Proceeding with concatenation based on 'id'."
        )
        raise ValueError("Datasets are not equal length.")

    outcome_df = pd.concat([eval_df_pd, llm_df_pd], axis=1)
    logger.info("Concatenated eval_df and llm_ans_list DataFrames based on 'id'.")

    # Merge reasoning_df if provided and not already included
    if reasoning_df is not None and len(reasoning_df) == len(outcome_df):
        try:
            outcome_df = pd.concat([outcome_df, reasoning_df], axis=1)
            logger.info("Concated with reasoning_df.")
        except Exception as e:
            logger.error(f"Error merging with reasoning_df: {e}")
            raise ValueError(f"Error merging with reasoning_df: {e}")

    # Merge accuracy_df if provided
    if accuracy_df is not None and len(accuracy_df) == len(outcome_df):
        try:
            outcome_df = pd.concat([outcome_df, accuracy_df], axis=1)
            logger.info("Concatenated accuracy_df")
        except Exception as e:
            logger.error(f"Error merging with accuracy_df: {e}")
            raise ValueError(f"Error merging with accuracy_df: {e}")

    # Handle any missing merges (optional)
    missing_reasoning = (
        outcome_df["reasoning"].isnull().sum()
        if "reasoning" in outcome_df.columns
        else 0
    )
    missing_accuracy = (
        outcome_df["accuracy_metric"].isnull().sum()
        if "accuracy_metric" in accuracy_df.columns
        else 0
    )

    if missing_reasoning > 0:
        logger.warning(f"{missing_reasoning} entries are missing reasoning data.")
    if missing_accuracy > 0:
        logger.warning(f"{missing_accuracy} entries are missing accuracy data.")

    logger.info(f"Final outcome DataFrame has {len(outcome_df)} entries.")
    return outcome_df
