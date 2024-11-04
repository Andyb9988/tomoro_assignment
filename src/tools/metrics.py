import re
from logging import Logger
from typing import (
    List,
    Optional,
)

import dspy
import pandas as pd

from src.tools.models import OpenAILM
from src.tools.modules import LLMJudge
from src.tools.signatures import AssessReasoning
from src.utils.logging_utils import get_logger

logger: Logger = get_logger(name=__name__)


def get_reasoning_scores(data, llm_ans, model: str = "gpt-4-turbo") -> pd.DataFrame:
    """
    Returns a pandas DataFrame of reasoning scores, each with the associated data_item.id.

    Args:
        data (List): A list of data items, each containing `context`, `dialogue_break`, and `id` attributes.
        llm_ans (List): A list of answer objects, each containing a `reasoning` attribute.
        model (str): The model to use for reasoning assessment.

    Returns:
        pd.DataFrame: A DataFrame with columns `id` and `reasoning_score`.
    """
    reasoning_list = []
    lm = OpenAILM(model=model)
    dspy.configure(lm=lm)
    logger.info(f"Using model: {lm}, {model}")

    for data_item, ans_obj in zip(data, llm_ans):
        try:
            context = data_item.context
            dialogue_break = data_item.dialogue_break
            llm_reasoning = ans_obj.reasoning
        except AttributeError as e:
            logger.error(f"Missing expected attribute: {e}")
            continue

        try:
            assessment = LLMJudge(
                context=context,
                llm_reasoning=llm_reasoning,
                dialogue_break=dialogue_break,
            ).get_answer()
            reasoning_accuracy = assessment.assessment_answer
        except Exception as e:
            logger.error(f"Error during reasoning assessment: {e}")
            continue

        try:
            reasoning_accuracy = float(reasoning_accuracy)
            reasoning_list.append((data_item.id, reasoning_accuracy))
            logger.info(
                f"Reasoning Accuracy for ID {data_item.id}: {reasoning_accuracy}"
            )
        except (ValueError, TypeError) as e:
            logger.error(
                f"Non-numeric reasoning accuracy encountered for ID {data_item.id}: {reasoning_accuracy} ({e})"
            )

    # Convert the list of tuples to a pandas DataFrame
    df_reasoning = pd.DataFrame(reasoning_list, columns=["id", "reasoning_score"])
    return df_reasoning


def calculate_average_reasoning_score(reasoning_df: pd.DataFrame) -> Optional[float]:
    """
    Calculates the average reasoning score from a DataFrame of reasoning scores.

    Args:
        reasoning_df (pd.DataFrame): A DataFrame with columns `id` and `reasoning_score`.

    Returns:
        Optional[float]: The average reasoning score if at least one score is present, otherwise `None`.
    """
    if not reasoning_df.empty:
        average_reasoning_score = reasoning_df["reasoning_score"].mean()
        logger.info(f"Average Reasoning Accuracy: {average_reasoning_score:.2f}")
        return average_reasoning_score
    else:
        logger.info("No reasoning scores to average.")
        return None


# def get_average_reasoning_score(
#     data, llm_ans, model: str = "gpt-4-turbo"
# ) -> Optional[float]:
#     """
#     Calculates the average reasoning accuracy score based on provided data and LLM answers.

#     This function iterates through paired items from `data` and `llm_ans`, evaluates the reasoning
#     accuracy using the `AssessReasoning` predictor, and computes the average accuracy score.

#     Args:
#         data (List): A list of data items, each containing `context` and `dialogue_break` attributes.
#         llm_ans (List): A list of answer objects, each containing a `reasoning` attribute.

#     Returns:
#         Optional[float]: The average reasoning accuracy score between 1-10 if at least one valid score is computed;
#                          otherwise, `None`.

#     Raises:
#         AttributeError: If expected attributes (`context`, `dialogue_break`, `reasoning`) are missing.
#     """
#     reasoning_list: List[float] = []
#     lm = OpenAILM(model=model)
#     dspy.configure(lm=lm)
#     logger.info(f"Using model: {lm}, {model}")

#     # Iterate through the data and LLM answers simultaneously
#     for data_item, ans_obj in zip(data, llm_ans):
#         try:
#             # Extract necessary attributes
#             context = data_item.context
#             dialogue_break = data_item.dialogue_break
#             llm_reasoning = ans_obj.reasoning
#         except AttributeError as e:
#             logger.error(f"Missing expected attribute: {e}")
#             continue

#         # Assess reasoning accuracy
#         try:
#             assessment = dspy.Predict(AssessReasoning)(
#                 actual_reasoning=dialogue_break,
#                 llm_reasoning=llm_reasoning,
#                 context=context,
#             )
#             reasoning_accuracy = assessment.assessment_answer
#         except Exception as e:
#             logger.error(f"Error during reasoning assessment: {e}")
#             continue

#         # Convert reasoning accuracy to float and add to list
#         try:
#             reasoning_accuracy = float(reasoning_accuracy)
#             reasoning_list.append(reasoning_accuracy)
#             logger.info(f"Reasoning Accuracy: {reasoning_accuracy}")
#         except (ValueError, TypeError) as e:
#             logger.error(
#                 f"Non-numeric reasoning accuracy encountered: {reasoning_accuracy} ({e})"
#             )
#         # Calculate and return the average reasoning accuracy if available

#     if reasoning_list:
#         average_reasoning_score = sum(reasoning_list) / len(reasoning_list)
#         logger.info(f"Average Reasoning Accuracy: {average_reasoning_score:.2f}")
#         return average_reasoning_score
#     else:
#         logger.info("No reasoning scores to average.")
#         return None


def get_answer_accuracy_df(
    data: List[object], llm_answers: List[object]
) -> pd.DataFrame:
    """
    Compares predicted answers with actual answers and returns a DataFrame with results.

    Args:
        data (List[object]): List of actual answer objects with an 'answer' attribute.
        llm_answers (List[object]): List of predicted answer objects with an 'answer' attribute.

    Returns:
        pd.DataFrame: DataFrame containing 'id' and 'result' columns.
                      'result' is 'correct' or 'incorrect' based on the comparison.
    """
    if not llm_answers or not data:
        logger.warning("One or both input lists are empty.")
        return pd.DataFrame(columns=["id", "result"])

    if len(llm_answers) != len(data):
        logger.warning(
            "Input lists have different lengths. Processing up to the shortest length."
        )

    # Define a regex pattern to remove specified symbols
    pattern = r"[\(\)\$£%]"

    results = []
    min_length = min(len(llm_answers), len(data))

    for idx in range(min_length):
        predicted_obj = llm_answers[idx]
        actual_obj = data[idx]

        predicted_raw = getattr(predicted_obj, "answer", "")
        actual_raw = getattr(actual_obj, "answer", "")

        # Clean the answers by removing specified symbols
        predicted_clean = re.sub(pattern, "", predicted_raw)
        actual_clean = re.sub(pattern, "", actual_raw)

        try:
            predicted_value = float(predicted_clean)
            actual_value = float(actual_clean)
            difference = abs(predicted_value - actual_value)
            logger.info(
                f"Parsed Values [{idx}]: Predicted={predicted_value}, Actual={actual_value}, Difference={difference}"
            )

            if difference <= 1:
                result = "correct"
                logger.debug(f"Result [{idx}]: Correct")
            else:
                result = "incorrect"
                logger.debug(f"Result [{idx}]: Incorrect")

        except ValueError:
            # If conversion fails, consider the answer incorrect
            result = "incorrect"
            logger.debug(f"Result [{idx}]: Incorrect (ValueError during conversion)")

        results.append({"result": result})

    df = pd.DataFrame(results)
    logger.info(f"Generated DataFrame with {len(df)} entries.")
    return df


def calculate_accuracy_from_df(df: pd.DataFrame) -> float:
    """
    Calculates the accuracy percentage from a DataFrame of results.

    Args:
        df (pd.DataFrame): DataFrame containing a 'result' column with 'correct' or 'incorrect' entries.

    Returns:
        float: Accuracy percentage of correct answers.
    """
    if df.empty:
        logger.warning("The DataFrame is empty. Accuracy is set to 0.0%.")
        return 0.0

    total = len(df)
    correct = df["result"].str.lower().eq("correct").sum()

    accuracy = (correct / total) * 100
    logger.info(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy


# def get_answer_accuracy(data: List[object], llm_answers: List[object]) -> float:
#     """
#     Calculates the accuracy as the percentage of correct answers.

#     An answer is considered correct if the absolute difference between
#     the predicted and actual values (after cleaning) is within 1.

#     Args:
#         llm_answers (List[object]): List of predicted answer objects with an 'answer' attribute.
#         data (List[object]): List of actual answer objects with an 'answer' attribute.

#     Returns:
#         float: Accuracy percentage of correct answers.
#     """

#     if not llm_answers or not data:
#         logger.warning("One or both input lists are empty.")
#         return 0.0

#     if len(llm_answers) != len(data):
#         logger.warning(
#             "Input lists have different lengths. Processing up to the shortest length."
#         )

#     # Define a regex pattern to remove specified symbols
#     pattern = r"[\(\)\$£%]"

#     correct = 0
#     total = 0
#     min_length = min(len(llm_answers), len(data))

#     for idx in range(min_length):
#         predicted_obj = llm_answers[idx]
#         actual_obj = data[idx]

#         predicted_raw = getattr(predicted_obj, "answer", "")
#         actual_raw = getattr(actual_obj, "answer", "")

#         # Clean the answers by removing specified symbols
#         predicted_clean = re.sub(pattern, "", predicted_raw)
#         actual_clean = re.sub(pattern, "", actual_raw)

#         try:
#             predicted_value = float(predicted_clean)
#             actual_value = float(actual_clean)
#             difference = abs(predicted_value - actual_value)
#             logger.info(
#                 f"Parsed Values [{idx}]: Predicted={predicted_value}, Actual={actual_value}, Difference={difference}"
#             )

#             if difference <= 1:
#                 correct += 1
#                 logger.debug(f"Result [{idx}]: Correct")
#             else:
#                 logger.debug(f"Result [{idx}]: Incorrect")

#         except ValueError:
#             # If conversion fails, consider the answer incorrect
#             logger.debug(f"Result [{idx}]: Incorrect (ValueError during conversion)")
#             pass

#         total += 1

#     if total == 0:
#         logger.warning("No valid answers to process.")
#         return 0.0

#     accuracy = (correct / total) * 100
#     logger.info(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
#     return accuracy
